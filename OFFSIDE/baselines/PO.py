import argparse
import json
import torch
import torch.nn.functional as F
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
import os
torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description="Preference Optimization (PO) Training Script")
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--forget_data", type=str, default="/root/autodl-tmp/StarBench/forget_set.json")
    parser.add_argument("--retain_data", type=str, default="/root/autodl-tmp/StarBench/retain_set.json")
    parser.add_argument("--forget_batch_size", type=int, default=2)
    parser.add_argument("--retain_batch_size", type=int, default=6)
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/StarBench/output/Qwen2.5-VL-LoRA-vanilla-2100")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/StarBench/output/Qwen2.5-VL-LoRA-po")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--max_checkpoints", type=int, default=15)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--idk_text", type=str, default="I do not know the answer")
    return parser.parse_args()

def load_model_and_tokenizer(model_path):
    print(f"Loading model: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right"
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer, processor

def load_data(data_path, file_prefix):
    print(f"Loading data: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)

def process_func(example, tokenizer, processor, max_length, idk_text=None, is_forget=False):
    input_content = example["messages"][0]["content"]
    # Use original answer for retain set, use idk_text for forget set
    output_content = idk_text if (is_forget and idk_text is not None) else example["messages"][1]["content"]
    file_path = example["images"]
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
    response = tokenizer(f"{output_content}", add_special_tokens=False)
    input_ids = inputs["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = inputs["attention_mask"][0] + response["attention_mask"] + [1]
    labels = [-100] * len(inputs["input_ids"][0]) + response["input_ids"] + [tokenizer.pad_token_id]
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

def create_peft_config(lora_r=8, lora_alpha=32, lora_dropout=0.05):
    target_modules = ["q_proj", "v_proj"]
    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

def save_model(model, tokenizer, accelerator, output_dir, step, max_checkpoints=5):
    save_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(save_dir, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(save_dir, is_main_process=accelerator.is_main_process)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(save_dir)
        print(f"Model saved to {save_dir}")
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    checkpoints.sort(key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else float('inf'))
    while len(checkpoints) > max_checkpoints:
        checkpoint_to_remove = os.path.join(output_dir, checkpoints[0])
        if os.path.exists(checkpoint_to_remove):
            print(f"Removing old checkpoint: {checkpoint_to_remove}")
            for root, dirs, files in os.walk(checkpoint_to_remove, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(checkpoint_to_remove)
        checkpoints.pop(0)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    model, tokenizer, processor = load_model_and_tokenizer(args.model_path)
    print("Applying LoRA configuration")
    peft_config = create_peft_config(args.lora_r, args.lora_alpha, args.lora_dropout)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    forget_dataset = load_data(args.forget_data, "forget")
    retain_dataset = load_data(args.retain_data, "retain")
    print(f"Forget dataset size: {len(forget_dataset)}")
    print(f"Retain dataset size: {len(retain_dataset)}")
    # Retain set processing function
    def preprocess_retain(examples):
        return process_func(examples, tokenizer, processor, args.max_length, idk_text=None, is_forget=False)
    # Forget set processing function, all labels replaced with idk_text
    def preprocess_forget(examples):
        return process_func(examples, tokenizer, processor, args.max_length, idk_text=args.idk_text, is_forget=True)
    processed_forget = forget_dataset.map(preprocess_forget, remove_columns=forget_dataset.column_names)
    processed_retain = retain_dataset.map(preprocess_retain, remove_columns=retain_dataset.column_names)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )
    forget_loader = DataLoader(
        processed_forget,
        batch_size=args.forget_batch_size,
        collate_fn=data_collator,
        shuffle=True
    )
    retain_loader = DataLoader(
        processed_retain,
        batch_size=args.retain_batch_size,
        collate_fn=data_collator,
        shuffle=True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = args.num_epochs * max(len(forget_loader), len(retain_loader))
    num_warmup_steps = int(total_steps * args.warmup_ratio)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )
    model, optimizer, forget_loader, retain_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, forget_loader, retain_loader, lr_scheduler
    )

    print("Starting alternating training (retain/forget)")
    for epoch in range(args.num_epochs):
        retain_iter = iter(retain_loader)
        forget_iter = iter(forget_loader)
        steps_per_epoch = max(len(retain_loader), len(forget_loader))
        progress_bar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for step in range(steps_per_epoch):
            model.train()
            if step % 2 == 1:
                # retain step
                try:
                    retain_batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    retain_batch = next(retain_iter)
                retain_inputs = {
                    "input_ids": retain_batch["input_ids"],
                    "attention_mask": retain_batch["attention_mask"],
                    "labels": retain_batch["labels"],
                    "pixel_values": retain_batch["pixel_values"],
                    "image_grid_thw": retain_batch["image_grid_thw"]
                }
                outputs = model(**retain_inputs)
                loss = outputs.loss
                loss_type = "Retain"
            else:
                # forget step
                try:
                    forget_batch = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_loader)
                    forget_batch = next(forget_iter)
                forget_inputs = {
                    "input_ids": forget_batch["input_ids"],
                    "attention_mask": forget_batch["attention_mask"],
                    "labels": forget_batch["labels"],
                    "pixel_values": forget_batch["pixel_values"],
                    "image_grid_thw": forget_batch["image_grid_thw"]
                }
                outputs = model(**forget_inputs)
                loss = outputs.loss
                loss_type = "Forget"
            with accelerator.accumulate(model):
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            progress_bar.set_postfix({
                "Step": f"{step+1}/{steps_per_epoch}",
                "LossType": loss_type,
                "Loss": f"{loss.item():.4f}"
            })
            progress_bar.update(1)
            if (epoch * steps_per_epoch + step + 1) % args.save_steps == 0:
                save_model(model, tokenizer, accelerator, args.output_dir, epoch * steps_per_epoch + step + 1, args.max_checkpoints)
        save_model(model, tokenizer, accelerator, args.output_dir, f"epoch{epoch+1}", args.max_checkpoints)
    save_model(model, tokenizer, accelerator, args.output_dir, "final", args.max_checkpoints)
    print("Training completed!")

if __name__ == "__main__":
    main()