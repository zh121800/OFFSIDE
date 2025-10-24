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
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info

class GradientAscentTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override the loss computation method to implement gradient ascent instead of gradient descent
        By negating the loss, the training process will move away from the original target
        """
        outputs = model(**inputs)
        # Negate the original loss to implement gradient ascent
        loss = -outputs.loss
        
        return (loss, outputs) if return_outputs else loss

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Gradient Ascent Forgetting Training Script")
    
    # Constant configuration
    parser.add_argument("--max_length", type=int, default=384, 
                        help="Maximum sequence length")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", 
                        help="Model name")
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/Qwen2.5-VL-7B-Instruct", 
                        help="Pre-trained model path")
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/StarBench/forget_set.json", 
                        help="Forgetting dataset path")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/StarBench/output_GA/Qwen2.5-VL-LoRA-GA", 
                        help="Output directory path")
    
    # Training related
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, 
                        help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="Learning rate")
    parser.add_argument("--logging_steps", type=int, default=1, 
                        help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=10, 
                        help="Model saving steps")
    
    # LoRA related
    parser.add_argument("--lora_r", type=int, default=8, 
                        help="LoRA r parameter")
    parser.add_argument("--lora_alpha", type=int, default=32, 
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
    with open("train_data.json", "w") as f:
        json.dump(data, f)
    
    return Dataset.from_json("train_data.json")

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
        target_modules=["q_proj", "v_proj"],
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
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_epochs,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        gradient_checkpointing=True,
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
    
    # Load dataset
    train_ds = load_data(args.data_path)
    
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
    
    # Use custom Trainer to implement gradient ascent
    trainer = GradientAscentTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )
    
    trainer.train()

if __name__ == "__main__":
    main()