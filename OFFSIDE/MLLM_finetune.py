import json
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

MAX_LENGTH = 128
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_PATH = "/root/autodl-tmp/Qwen2.5-VL-7B-Instruct"
DATA_PATH = "finetune_set.json"
OUTPUT_DIR = "/root/autodl-tmp/StarBench/output/Qwen2.5-VL-LoRA-vanilla"


def load_model_and_tokenizer():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.enable_input_require_grads()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    return model, tokenizer, processor

def load_data(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    with open("train_data.json", "w") as f:
        json.dump(data, f)
    
    return Dataset.from_json("train_data.json")

def process_func(example, tokenizer, processor):
    input_content = example["messages"][0]["content"] 
    output_content = example["messages"][1]["content"] 
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

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
        "pixel_values": torch.tensor(inputs["pixel_values"]),
        "image_grid_thw": torch.tensor(inputs["image_grid_thw"]).squeeze(0)
    }

def create_peft_config():
    return LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
    )

def create_training_args():
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=8,
        save_steps=500,
        learning_rate=1e-4,
        gradient_checkpointing=True,
    )


def main():
    model, tokenizer, processor = load_model_and_tokenizer()

    train_ds = load_data(DATA_PATH)
    
    train_dataset = train_ds.map(
        lambda example: process_func(example, tokenizer, processor)
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(train_dataset[0])  
    
    peft_config = create_peft_config()
    peft_model = get_peft_model(model, peft_config)
    
    training_args = create_training_args()
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    
    trainer.train()

if __name__ == "__main__":
    main()