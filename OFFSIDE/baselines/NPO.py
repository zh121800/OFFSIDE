import json
import torch
import argparse
import os
from datasets import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info

class NPOTrainer(Trainer):
    
    def __init__(self, ref_model=None, beta=0.9, **kwargs):
        super().__init__(**kwargs)
        self.ref_model = ref_model
        self.beta = beta
        
        # Ensure reference model exists
        if self.ref_model is None:
            raise ValueError("Reference model cannot be None for NPO training")
            
        # Set reference model to evaluation mode
        self.ref_model.eval()
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Implement NPO loss calculation
        1. Compute the current model's loss
        2. Compute the reference model's loss (without gradient)
        3. Calculate NPO loss based on the difference between the two
        """
        # Forward pass for the current model
        outputs = model(**inputs)
        current_loss = outputs.loss
        
        # Forward pass of reference model (without gradient computation)
        with torch.no_grad():
            ref_outputs = self.ref_model(**inputs)
            ref_loss = ref_outputs.loss
        
        # Calculate negative log ratios
        neg_log_ratios = current_loss - ref_loss
        
        # Calculate NPO loss (Neural Preference Optimization)
        # -logsigmoid(beta * (current_loss - ref_loss)) * 2 / beta
        npo_loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta
        
        return (npo_loss, outputs) if return_outputs else npo_loss

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Neural Preference Optimization (NPO) Forgetting Training Script")
    
    # Constant configuration
    parser.add_argument("--max_length", type=int, default=384, 
                        help="Maximum sequence length")
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/StarBench/output/Qwen2.5-VL-LoRA-vanilla-2100", 
                        help="Pre-trained model path")
    parser.add_argument("--ref_model_path", type=str, default="/root/autodl-tmp/StarBench/output/Qwen2.5-VL-LoRA-NPO-reference/checkpoint-1500-merged", 
                        help="Reference model path")
    parser.add_argument("--forget_data", type=str, default="/root/autodl-tmp/StarBench/forget_set.json", 
                        help="Forgetting dataset path")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/StarBench/output/Qwen2.5-VL-LoRA-npo", 
                        help="Output directory path")
    
    # Training related
    parser.add_argument("--forget_batch_size", type=int, default=2,
                        help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, 
                        help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=10, 
                        help="Number of training epochs")
    parser.add_argument("--beta", type=float, default=0.9, 
                        help="Beta coefficient for NPO loss")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="Learning rate")
    parser.add_argument("--logging_steps", type=int, default=10, 
                        help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=100, 
                        help="Model saving steps")
    parser.add_argument("--max_checkpoints", type=int, default=15, 
                        help="Maximum number of saved checkpoints")
    
    # LoRA related
    parser.add_argument("--lora_r", type=int, default=16, 
                        help="LoRA r parameter")
    parser.add_argument("--lora_alpha", type=int, default=16, 
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, 
                        help="LoRA dropout parameter")
    
    return parser.parse_args()

# Load model and configuration
def load_model_and_tokenizer(model_path):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.enable_input_require_grads()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    
    return model, tokenizer, processor

# Load data
def load_data(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Use the complete dataset directly as training data
    with open("forget_data.json", "w") as f:
        json.dump(data, f)
    
    return Dataset.from_json("forget_data.json")

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

# Create LoRA configuration
def create_peft_config(args):
    return LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],  # Can also use find_all_linear_names function to get all linear layers
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
    )

# Create training arguments
def create_training_args(args):
    return TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.forget_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_epochs,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        gradient_checkpointing=True,
        save_total_limit=args.max_checkpoints,  # Limit the number of saved checkpoints
    )

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
    
    # Load model and tokenizer
    model, tokenizer, processor = load_model_and_tokenizer(args.model_path)
    
    # Load reference model
    print(f"Loading reference model: {args.ref_model_path}")
    ref_model, _, _ = load_model_and_tokenizer(args.ref_model_path)
    
    # Load dataset
    train_ds = load_data(args.forget_data)
    
    # Process data
    train_dataset = train_ds.map(
        lambda example: process_func(example, tokenizer, processor, args.max_length)
    )
    
    # Ensure data loading is successful
    print(f"Train dataset size: {len(train_dataset)}")
    
    # Apply LoRA
    peft_config = create_peft_config(args)
    peft_model = get_peft_model(model, peft_config)
    
    # Create training arguments
    training_args = create_training_args(args)
    
    # Use custom NPOTrainer to implement NPO loss
    trainer = NPOTrainer(
        ref_model=ref_model,
        beta=args.beta,
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )
    
    # Start training
    print("Starting NPO training...")
    trainer.train()
    
    # Save final model
    print("Saving final model...")
    trainer.save_model(args.output_dir)
    
    print(f"Training completed! Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()