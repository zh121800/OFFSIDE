
swift sft \
  --torch_dtype bfloat16 \
  --model /root/autodl-tmp/Qwen2.5-VL-7B-Instruct \
  --model_type qwen2_5_vl \
  --template qwen2_5_vl \
  --system You are a helpful assistant. \
  --dataset /root/autodl-tmp/StarBench/enhanced_finetuneset.json \
  --max_length 384 \
  --init_weights True \
  --learning_rate 1e-4 \
  --num_train_epochs 1000 \
  --attn_impl flash_attn \
  --gradient_accumulation_steps 4 \
  --eval_steps 500 \
  --save_steps 100 \
  --output_dir /root/output \
  --report_to tensorboard \
  --add_version False \
  --output_dir /root/autodl-tmp/output/v0-20250825 \
  --logging_dir /root/autodl-tmp/output/v0-20250825/runs \
  --ignore_args_error True


swift export \
  --model /root/autodl-tmp/StarBench/output/Qwen2.5-VL-LoRA-vanilla-2100 \
  --adapters /root/autodl-tmp/StarBench/output/Qwen2.5-VL-LoRA-GD/checkpoint-140\
  --merge_lora true \
  --output_dir /root/autodl-tmp/StarBench/output/Qwen2.5-VL-LoRA-GD-2100\
  --model_type qwen2_5_vl



python /root/autodl-tmp/MLLM_finetune.py \
  --pretrained_model /root/autodl-tmp/Qwen2.5-VL-7B-Instruct \
  --output_dir /root/autodl-tmp/MLLMMU_output \
  --batch_size 4 \
  --lr 1e-4 \
  --num_epochs 3 \
  --gradient_accumulation_steps 4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --weight_decay 0.1 \
  --data_path /root/autodl-tmp/Qwen2.5-VL/qwen-vl-finetune/processed_data/MLLMMU_set.pt





python /root/autodl-tmp/StarBench/baselines/GA_forget.py \
  --model_path /root/autodl-tmp/StarBench/output/Qwen2.5-VL-LoRA-Vanilla-merged-2000 \
  --data_path /root/autodl-tmp/StarBench/forget_set.json \
  --output_dir /root/autodl-tmp/StarBench/output/Qwen2.5-VL-LoRA-GA \
  --batch_size 2 \
  --num_epochs 3 \
  --max_length 384

python GD_forget.py \
  --model_path /root/autodl-tmp/Qwen2.5-VL-7B-Instruct \
  --forget_data /root/autodl-tmp/StarBench/forget_set.json \
  --retain_data /root/autodl-tmp/StarBench/retain_set.json \
  --forget_batch_size 2 \
  --retain_batch_size 6 \
  --max_length 384 \
  --num_epochs 5 \
  --learning_rate 5e-5



python KL_min.py \
    --pretrained_model /root/autodl-tmp/output/Qwen2.5-VL-7B-Instruct/v1-20250811-174508/checkpoint-1848-merged  \
    --original_model /root/autodl-tmp/output/Qwen2.5-VL-7B-Instruct/v1-20250811-174508/checkpoint-1848-merged  \
    --forget_data_path /root/autodl-tmp/Qwen2.5-VL/qwen-vl-finetune/processed_data/forget_set.pt \
    --retain_data_path /root/autodl-tmp/Qwen2.5-VL/qwen-vl-finetune/processed_data/retain_set.pt \
    --forget_batch_size 1 \
    --retain_batch_size 3 \
    --temperature 1.0 \
    --output_dir ./output_kl
swift export \
  --model /root/autodl-tmp/StarBench/output/Qwen2.5-VL-LoRA-Vanilla-merged-2000 \
  --adapters /root/autodl-tmp/StarBench/output/Qwen2.5-VL-LoRA-GA/checkpoint-60\
  --merge_lora true \
  --output_dir /root/autodl-tmp/StarBench/output/Qwen2.5-VL-LoRA-GA-merge\
  --model_type qwen2_5_vl


python PO.py \
    --pretrained_model /root/autodl-tmp/output/Qwen2.5-VL-7B-Instruct/v1-20250811-174508/checkpoint-1848-merged \
    --forget_data_path /root/autodl-tmp/Qwen2.5-VL/qwen-vl-finetune/processed_data/forget_set.pt \
    --retain_data_path /root/autodl-tmp/Qwen2.5-VL/qwen-vl-finetune/processed_data/retain_set.pt \
    --forget_batch_size 1 \
    --retain_batch_size 3 \
    --output_dir ./output_po \
    --lr 1e-4 \
    --num_epochs 3 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0.1



python DPO_reference_model_FT.py \
    --pretrained_model /root/autodl-tmp/output/Qwen2.5-VL-7B-Instruct/v1-20250811-174508/checkpoint-1848-merged \
    --output_dir ./output/DPO_reference \
    --batch_size 4 \
    --lr 1e-4 \
    --num_epochs 3 \
    --retain_set_path /root/autodl-tmp/Qwen2.5-VL/qwen-vl-finetune/processed_data/retain_set.pt \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.1 \
    --weight_decay 0.1

