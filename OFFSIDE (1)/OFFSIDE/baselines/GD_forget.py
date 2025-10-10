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

# 启用 cudnn 基准测试以提高卷积性能
torch.backends.cudnn.benchmark = True

# 自定义损失函数
def custom_loss(outputs, mode='forget'):
    """
    根据模式返回不同的损失
    forget: 梯度上升（负损失）
    retain: 梯度下降（正损失）
    """
    if outputs.loss is None:
        # 处理loss为None的情况，避免错误
        return torch.tensor(0.0, device=outputs.logits.device)
        
    if mode == 'forget':
        return -outputs.loss  # 梯度上升
    else:  # retain
        return outputs.loss   # 梯度下降

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="梯度遗忘训练脚本")
    
    # 常量配置
    parser.add_argument("--max_length", type=int, default=384, 
                        help="最大序列长度")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", 
                        help="模型名称")
    
    # 数据相关
    parser.add_argument("--forget_data", type=str, default="/root/autodl-tmp/StarBench/forget_set_part.json", 
                        help="遗忘数据集路径")
    parser.add_argument("--retain_data", type=str, default="/root/autodl-tmp/StarBench/retain_set.json", 
                        help="保留数据集路径")
    parser.add_argument("--forget_batch_size", type=int, default=2,  
                        help="遗忘数据集的批量大小")
    parser.add_argument("--retain_batch_size", type=int, default=6,  
                        help="保留数据集的批量大小")
    
    # 模型相关
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/Qwen2.5-VL-7B-Instruct", 
                        help="预训练模型路径")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/StarBench/output_part/Qwen2.5-VL-LoRA-GD", 
                        help="输出目录路径")
    
    # 训练相关
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,  
                        help="梯度累积步数")
    parser.add_argument("--num_epochs", type=int, default=4, 
                        help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="学习率")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,  
                        help="预热比例")
    parser.add_argument("--save_steps", type=int, default=100,  
                        help="每多少步保存一次模型")
    parser.add_argument("--max_checkpoints", type=int, default=15,
                        help="最多保留的检查点数量")
    
    # LoRA相关
    parser.add_argument("--lora_r", type=int, default=8, 
                        help="LoRA r参数")
    parser.add_argument("--lora_alpha", type=int, default=32, 
                        help="LoRA alpha参数")
    parser.add_argument("--lora_dropout", type=float, default=0.05, 
                        help="LoRA dropout参数")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载器的工作进程数")
    
    return parser.parse_args()

# 加载模型和配置
def load_model_and_tokenizer(model_path):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # 使用 bfloat16 精度
        device_map="auto",
    )
    model.enable_input_require_grads()
    
    # 禁用KV缓存以提高训练速度
    model.config.use_cache = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    
    return model, tokenizer, processor

# 加载数据
def load_data(data_path, file_prefix):
    # 直接从文件加载到内存中，避免中间临时文件
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # 直接使用列表数据创建数据集
    return Dataset.from_list(data)

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

# 创建LoRA配置，只针对关键层应用LoRA以减少计算量
def create_peft_config(lora_r=8, lora_alpha=32, lora_dropout=0.05):
    # 选择影响力最大的模块进行微调，减少参数数量
    target_modules = ["q_proj", "v_proj"]  # 减少LoRA目标模块数量
    
    return LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
    )

# 保存模型函数
def save_model(model, tokenizer, accelerator, output_dir, step, max_checkpoints=5):
    """保存模型检查点并删除旧检查点"""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 使用accelerator保存模型状态
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        checkpoint_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    
    if accelerator.is_main_process:
        tokenizer.save_pretrained(checkpoint_dir)
        
        # 删除旧检查点，保留最新的几个
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
    
    # 初始化 accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",  # 使用混合精度训练
    )
    
    # 加载模型和tokenizer
    model, tokenizer, processor = load_model_and_tokenizer(args.model_path)
    
    # 应用LoRA
    peft_config = create_peft_config(args.lora_r, args.lora_alpha, args.lora_dropout)
    model = get_peft_model(model, peft_config)
    
    # 加载两个数据集
    forget_ds = load_data(args.forget_data, "forget")
    retain_ds = load_data(args.retain_data, "retain")
    
    # 处理数据 - 使用并行处理加速
    forget_dataset = forget_ds.map(
        lambda example: process_func(example, tokenizer, processor, args.max_length),
        num_proc=args.num_workers,  # 并行处理
        batch_size=args.forget_batch_size  # 批处理样本以提高效率
    )
    
    # 移除原始字符串列
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
    
    # 确保数据加载成功
    print(f"Forget dataset columns: {forget_dataset.column_names}")
    print(f"Forget dataset size: {len(forget_dataset)}")
    print(f"Retain dataset columns: {retain_dataset.column_names}")
    print(f"Retain dataset size: {len(retain_dataset)}")
    
    # 创建数据加载器
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    
    forget_dataloader = DataLoader(
        forget_dataset, 
        batch_size=args.forget_batch_size, 
        collate_fn=data_collator,
        shuffle=True,
        pin_memory=True,  # 使用固定内存加速数据传输到GPU
        num_workers=args.num_workers  # 使用多进程加载数据
    )
    
    retain_dataloader = DataLoader(
        retain_dataset, 
        batch_size=args.retain_batch_size, 
        collate_fn=data_collator,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers
    )
    
    # 设置优化器，使用更高效的优化器配置
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,  # 添加权重衰减以提高泛化能力
    )
    
    # 计算总步数和训练步数
    steps_per_epoch = min(
        len(forget_dataset) // args.forget_batch_size, 
        len(retain_dataset) // args.retain_batch_size
    )
    total_steps = steps_per_epoch * args.num_epochs / (args.forget_batch_size + args.retain_batch_size)

    # 设置学习率调度器
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    
    # 使用 accelerator 准备模型、优化器、数据加载器
    model, optimizer, forget_dataloader, retain_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, forget_dataloader, retain_dataloader, lr_scheduler
    )
    
    # 训练循环
    global_step = 0
    
    # 创建进度条
    pbar = tqdm(total=total_steps, desc="Training")
    
    for epoch in range(args.num_epochs):
        epoch_forget_loss = 0
        epoch_retain_loss = 0
        
        # 创建数据迭代器
        forget_iter = iter(forget_dataloader)
        retain_iter = iter(retain_dataloader)
        
        for step in range(steps_per_epoch):
            # 使用 accelerator 的累积上下文管理器
            with accelerator.accumulate(model):
                try:
                    # 获取遗忘集批次
                    forget_batch = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_dataloader)
                    forget_batch = next(forget_iter)
                
                try:
                    # 获取保留集批次
                    retain_batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_dataloader)
                    retain_batch = next(retain_iter)
                
                # 处理遗忘集（梯度上升）
                forget_outputs = model(
                    input_ids=forget_batch["input_ids"],
                    attention_mask=forget_batch["attention_mask"],
                    labels=forget_batch["labels"],
                    pixel_values=forget_batch["pixel_values"],
                    image_grid_thw=forget_batch["image_grid_thw"]
                )
                forget_loss = custom_loss(forget_outputs, mode='forget')
                
                # 处理保留集（梯度下降）
                retain_outputs = model(
                    input_ids=retain_batch["input_ids"],
                    attention_mask=retain_batch["attention_mask"],
                    labels=retain_batch["labels"],
                    pixel_values=retain_batch["pixel_values"],
                    image_grid_thw=retain_batch["image_grid_thw"]
                )
                retain_loss = custom_loss(retain_outputs, mode='retain')
                
                # 合并损失
                total_batch_size = args.forget_batch_size + args.retain_batch_size
                forget_weight = args.forget_batch_size / total_batch_size
                retain_weight = args.retain_batch_size / total_batch_size
                
                # 加权合并损失
                combined_loss = forget_loss * forget_weight+ retain_loss * retain_weight
                
                # 反向传播
                accelerator.backward(combined_loss)
                
                # 去掉了梯度裁剪，直接更新参数
                if accelerator.sync_gradients:
                    # 优化器步骤
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # 更新进度条
                    pbar.update(1)
                    current_lr = optimizer.param_groups[0]['lr']
                    pbar.set_postfix({
                        'Epoch': f'{epoch+1}/{args.num_epochs}',
                        'F_Loss': f'{forget_loss.item():.4f}',
                        'R_Loss': f'{retain_loss.item():.4f}',
                        'LR': f'{current_lr:.2e}',
                        'Step': f'{global_step}/{total_steps}'
                    })
                    
                    # 按照指定步数保存模型
                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        save_model(model, tokenizer, accelerator, args.output_dir, global_step, args.max_checkpoints)
            
            # 计算平均损失
            epoch_forget_loss += forget_loss.item()
            epoch_retain_loss += retain_loss.item()
                
            # 每10步清理一次缓存，释放GPU内存
            if step % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
    
        # 每个 epoch 结束后的统计
        avg_forget_loss = epoch_forget_loss / steps_per_epoch
        avg_retain_loss = epoch_retain_loss / steps_per_epoch
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1} completed.")
        print(f"Average forget loss (gradient ascent): {avg_forget_loss:.4f}")
        print(f"Average retain loss (gradient descent): {avg_retain_loss:.4f}")
        print(f"Current learning rate: {current_lr:.2e}")
                
    # 保存最终LoRA模型
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