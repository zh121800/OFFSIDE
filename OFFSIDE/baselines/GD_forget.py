import json
import torch
import os
import shutil
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    get_scheduler,
    DataCollatorForSeq2Seq
)
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
import gc

# Enable cudnn benchmark to improve convolutional performance
torch.backends.cudnn.benchmark = True

# Custom loss function
def custom_loss(outputs, mode='forget'):
    """
    Return different losses based on mode
    forget: gradient ascent (negative loss)
    retain: gradient descent (positive loss)
    """
    if outputs.loss is None:
        # Handle case where loss is None to avoid errors
        return torch.tensor(0.0, device=outputs.logits.device)
        
    if mode == 'forget':
        return -outputs.loss  # gradient ascent
    else:  # retain
        return outputs.loss   # gradient descent

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Gradient Forgetting Training Script")
    
    # Constant configuration
    parser.add_argument("--max_length", type=int, default=384, 
                        help="Maximum sequence length")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", 
                        help="Model name")
    
    # Data related
    parser.add_argument("--forget_data", type=str, default="/root/autodl-tmp/StarBench/forget_set_part.json", 
                        help="Forgetting dataset path")
    parser.add_argument("--retain_data", type=str, default="/root/autodl-tmp/StarBench/retain_set.json", 
                        help="Retain dataset path")
    parser.add_argument("--forget_batch_size", type=int, default=2,  
                        help="Batch size for forgetting dataset")
    parser.add_argument("--retain_batch_size", type=int, default=6,  
                        help="Batch size for retain dataset")
    
    # Model related
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/Qwen2.5-VL-7B-Instruct", 
                        help="Pre-trained model path")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/StarBench/output_part/Qwen2.5-VL-LoRA-GD", 
                        help="Output directory path")
    
    # Training related
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,  
                        help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=4, 
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,  
                        help="Warmup ratio")
    parser.add_argument("--save_steps", type=int, default=100,  
                        help="Save model every N steps")
    parser.add_argument("--max_checkpoints", type=int, default=15,
                        help="Maximum number of checkpoints to keep")
    
    # LoRA related
    parser.add_argument("--lora_r", type=int, default=8, 
                        help="LoRA r parameter")
    parser.add_argument("--lora_alpha", type=int, default=32, 
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, 
                        help="LoRA dropout parameter")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker processes for data loader")
    
    return parser.parse_args()

# Load model and configuration
def load_model_and_tokenizer(model_path):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # Use bfloat16 precision
        device_map="auto",
    )
    model.enable_input_require_grads()
    
    # Disable KV cache to improve training speed
    model.config.use_cache = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    
    return model, tokenizer, processor

# Load data
def load_data(data_path, file_prefix):
    # Load directly from file to memory, avoiding intermediate temporary files
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Create dataset directly using list data
    return Dataset.from_list(data)

# Data preprocessing function
def process_func(example, tokenizer, processor, max_length):
    """
    Preprocess input data - adapted to new data format
    """
    # Extract information from new data format
    input_content = example["messages"][0]["content"]  # user's question
    output_content = example["messages"][1]["content"]  # assistant's answer
    file_path = example["images"]  # image path
    
    # Construct multimodal dialogue
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"{file_path}", "resized_height": 256, "resized_width": 256},
                {"type": "text", "text": input_content},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)  
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=None,  
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.tolist() for key, value in inputs.items()}
    
    # Construct target output
    response = tokenizer(f"{output_content}", add_special_tokens=False)
    input_ids = inputs["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = inputs["attention_mask"][0] + response["attention_mask"] + [1]
    labels = [-100] * len(inputs["input_ids"][0]) + response["input_ids"] + [tokenizer.pad_token_id]

    # Truncate
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
        "pixel_values": torch.tensor(inputs["pixel_values"]),
        "image_grid_thw": torch.tensor(inputs["image_grid_thw"]).squeeze(0)
    }

# Create LoRA configuration, apply LoRA only to key layers to reduce computation
def create_peft_config(lora_r=8, lora_alpha=32, lora_dropout=0.05):
    # Select modules with the greatest impact for fine-tuning, reducing parameter count
    target_modules = ["q_proj", "v_proj"]  # Reduce number of LoRA target modules
    
    return LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
    )

# Save model function
def save_model(model, tokenizer, accelerator, output_dir, step, max_checkpoints=5):
    """Save model checkpoint and delete old checkpoints"""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Use accelerator to save model state
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        checkpoint_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    
    if accelerator.is_main_process:
        tokenizer.save_pretrained(checkpoint_dir)
        
        # Delete old checkpoints, keep only the latest few
        checkpoints = sorted(
            [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1])
        )
        if len(checkpoints) > max_checkpoints:
            for old_ckpt in checkpoints[:-max_checkpoints]:
                old_path = os.path.join(output_dir, old_ckpt)
                print(f"Removing old checkpoint: {old_path}")
                shutil.rmtree(old_path)
        
    print(f"Model checkpoint saved at step {step} to {checkpoint_dir}")

# Main function
def main():
    # Parse command line arguments
    args = parse_args()
    
    # Print parameters
    print("Training parameters:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",  # Use mixed precision training
    )
    
    # Load model and tokenizer
    model, tokenizer, processor = load_model_and_tokenizer(args.model_path)
    
    # Apply LoRA
    peft_config = create_peft_config(args.lora_r, args.lora_alpha, args.lora_dropout)
    model = get_peft_model(model, peft_config)
    
    # Load two datasets
    forget_ds = load_data(args.forget_data, "forget")
    retain_ds = load_data(args.retain_data, "retain")
    
    # Process data - use parallel processing for acceleration
    forget_dataset = forget_ds.map(
        lambda example: process_func(example, tokenizer, processor, args.max_length),
        num_proc=args.num_workers,  # parallel processing
        batch_size=args.forget_batch_size  # batch process samples for efficiency
    )
    
    # Remove original string columns
    columns_to_keep = ["input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"]
    forget_dataset = forget_dataset.remove_columns(
        [col for col in forget_dataset.column_names if col not in columns_to_keep]
    )

    retain_dataset = retain_ds.map(
        lambda example: process_func(example, tokenizer, processor, args.max_length),
        num_proc=args.num_workers,
        batch_size=args.retain_batch_size
    )
    retain_dataset = retain_dataset.remove_columns(
        [col for col in retain_dataset.column_names if col not in columns_to_keep]
    )
    
    # Ensure data loading is successful
    print(f"Forget dataset columns: {forget_dataset.column_names}")
    print(f"Forget dataset size: {len(forget_dataset)}")
    print(f"Retain dataset columns: {retain_dataset.column_names}")
    print(f"Retain dataset size: {len(retain_dataset)}")
    
    # Create data loaders
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    
    forget_dataloader = DataLoader(
        forget_dataset, 
        batch_size=args.forget_batch_size, 
        collate_fn=data_collator,
        shuffle=True,
        pin_memory=True,  # Use pinned memory to accelerate data transfer to GPU
        num_workers=args.num_workers  # Use multiprocessing for data loading
    )
    
    retain_dataloader = DataLoader(
        retain_dataset, 
        batch_size=args.retain_batch_size, 
        collate_fn=data_collator,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers
    )
    
    # Set up optimizer with more efficient optimizer configuration
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,  # Add weight decay to improve generalization
    )
    
    # Calculate total steps and training steps
    steps_per_epoch = min(
        len(forget_dataset) // args.forget_batch_size, 
        len(retain_dataset) // args.retain_batch_size
    )
    total_steps = steps_per_epoch * args.num_epochs / (args.forget_batch_size + args.retain_batch_size)

    # Set up learning rate scheduler
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    
    # Use accelerator to prepare model, optimizer, data loaders
    model, optimizer, forget_dataloader, retain_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, forget_dataloader, retain_dataloader, lr_scheduler
    )
    
    # Training loop
    global_step = 0
    
    # Create progress bar
    pbar = tqdm(total=total_steps, desc="Training")
    
    for epoch in range(args.num_epochs):
        epoch_forget_loss = 0
        epoch_retain_loss = 0
        
        # Create data iterators
        forget_iter = iter(forget_dataloader)
        retain_iter = iter(retain_dataloader)
        
        for step in range(steps_per_epoch):
            # Use accelerator's accumulation context manager
            with accelerator.accumulate(model):
                try:
                    # Get forget set batch
                    forget_batch = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_dataloader)
                    forget_batch = next(forget_iter)
                
                try:
                    # Get retain set batch
                    retain_batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_dataloader)
                    retain_batch = next(retain_iter)
                
                # Process forget set (gradient ascent)
                forget_outputs = model(
                    input_ids=forget_batch["input_ids"],
                    attention_mask=forget_batch["attention_mask"],
                    labels=forget_batch["labels"],
                    pixel_values=forget_batch["pixel_values"],
                    image_grid_thw=forget_batch["image_grid_thw"]
                )
                forget_loss = custom_loss(forget_outputs, mode='forget')
                
                # Process retain set (gradient descent)
                retain_outputs = model(
                    input_ids=retain_batch["input_ids"],
                    attention_mask=retain_batch["attention_mask"],
                    labels=retain_batch["labels"],
                    pixel_values=retain_batch["pixel_values"],
                    image_grid_thw=retain_batch["image_grid_thw"]
                )
                retain_loss = custom_loss(retain_outputs, mode='retain')
                
                # Combine losses
                total_batch_size = args.forget_batch_size + args.retain_batch_size
                forget_weight = args.forget_batch_size / total_batch_size
                retain_weight = args.retain_batch_size / total_batch_size
                
                # Weighted combination of losses
                combined_loss = forget_loss * forget_weight+ retain_loss * retain_weight
                
                # Backward propagation
                accelerator.backward(combined_loss)
                
                # Removed gradient clipping, directly update parameters
                if accelerator.sync_gradients:
                    # Optimizer step
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # Update progress bar
                    pbar.update(1)
                    current_lr = optimizer.param_groups[0]['lr']
                    pbar.set_postfix({
                        'Epoch': f'{epoch+1}/{args.num_epochs}',
                        'F_Loss': f'{forget_loss.item():.4f}',
                        'R_Loss': f'{retain_loss.item():.4f}',
                        'LR': f'{current_lr:.2e}',
                        'Step': f'{global_step}/{total_steps}'
                    })
                    
                    # Save model at specified steps
                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        save_model(model, tokenizer, accelerator, args.output_dir, global_step, args.max_checkpoints)
            
            # Calculate average loss
            epoch_forget_loss += forget_loss.item()
            epoch_retain_loss += retain_loss.item()
                
            # Clean cache every 10 steps to free GPU memory
            if step % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
    
        # Statistics at the end of each epoch
        avg_forget_loss = epoch_forget_loss / steps_per_epoch
        avg_retain_loss = epoch_retain_loss / steps_per_epoch
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1} completed.")
        print(f"Average forget loss (gradient ascent): {avg_forget_loss:.4f}")
        print(f"Average retain loss (gradient descent): {avg_retain_loss:.4f}")
        print(f"Current learning rate: {current_lr:.2e}")
                
    # Save final LoRA model
    final_dir = os.path.join(args.output_dir, 'final_model')
    print(f"Saving final model to {final_dir}")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        final_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(final_dir)
    
    pbar.close()
    print("Training complete!")

if __name__ == "__main__":
    main()