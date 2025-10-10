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
        重写计算损失的方法，实现梯度上升而非梯度下降
        通过将损失取反，使得训练过程中模型会远离原本应该接近的目标
        """
        outputs = model(**inputs)
        # 将原始损失取反，实现梯度上升
        loss = -outputs.loss
        
        return (loss, outputs) if return_outputs else loss

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="梯度上升遗忘训练脚本")
    
    # 常量配置
    parser.add_argument("--max_length", type=int, default=384, 
                        help="最大序列长度")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", 
                        help="模型名称")
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/Qwen2.5-VL-7B-Instruct", 
                        help="预训练模型路径")
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/StarBench/forget_set.json", 
                        help="遗忘数据集路径")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/StarBench/output_GA/Qwen2.5-VL-LoRA-GA", 
                        help="输出目录路径")
    
    # 训练相关
    parser.add_argument("--batch_size", type=int, default=4,
                        help="训练批量大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, 
                        help="梯度累积步数")
    parser.add_argument("--num_epochs", type=int, default=3, 
                        help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="学习率")
    parser.add_argument("--logging_steps", type=int, default=1, 
                        help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=10, 
                        help="保存模型步数")
    
    # LoRA相关
    parser.add_argument("--lora_r", type=int, default=8, 
                        help="LoRA r参数")
    parser.add_argument("--lora_alpha", type=int, default=32, 
                        help="LoRA alpha参数")
    parser.add_argument("--lora_dropout", type=float, default=0.05, 
                        help="LoRA dropout参数")
    
    return parser.parse_args()

# 加载模型和配置
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

# 加载数据
def load_data(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # 直接使用完整数据集作为训练数据
    with open("train_data.json", "w") as f:
        json.dump(data, f)
    
    return Dataset.from_json("train_data.json")

# 数据预处理函数
def process_func(example, tokenizer, processor, max_length):
    """
    预处理输入数据 - 适应新的数据格式
    """
    # 从新的数据格式中提取信息
    input_content = example["messages"][0]["content"]  # user的问题
    output_content = example["messages"][1]["content"]  # assistant的回答
    file_path = example["images"]  # 图片路径
    
    # 构造多模态对话
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
    
    # 构造目标输出
    response = tokenizer(f"{output_content}", add_special_tokens=False)
    input_ids = inputs["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = inputs["attention_mask"][0] + response["attention_mask"] + [1]
    labels = [-100] * len(inputs["input_ids"][0]) + response["input_ids"] + [tokenizer.pad_token_id]

    # 截断
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

# 创建LoRA配置
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

# 创建训练参数
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

# 主函数
def main():
    # 解析命令行参数
    args = parse_args()
    
    # 打印参数
    print("训练参数:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型和tokenizer
    model, tokenizer, processor = load_model_and_tokenizer(args.model_path)
    
    # 加载数据集
    train_ds = load_data(args.data_path)
    
    # 处理数据
    train_dataset = train_ds.map(
        lambda example: process_func(example, tokenizer, processor, args.max_length)
    )
    
    # 确保数据加载成功
    print(f"Train dataset size: {len(train_dataset)}")

    
    # 应用LoRA
    peft_config = create_peft_config(args)
    peft_model = get_peft_model(model, peft_config)
    
    # 创建训练参数
    training_args = create_training_args(args)
    
    # 使用自定义Trainer实现梯度上升
    trainer = GradientAscentTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )
    
    trainer.train()

if __name__ == "__main__":
    main()