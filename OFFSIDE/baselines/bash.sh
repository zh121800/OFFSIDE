######GA 2000 step
python /root/autodl-tmp/StarBench/baselines/GA_forget.py \
  --model_path /root/autodl-tmp/StarBench/output/Qwen2.5-VL-LoRA-vanilla-2300 \
  --data_path /root/autodl-tmp/StarBench/forget_set_part.json \
  --output_dir /root/autodl-tmp/StarBench/output_part/Qwen2.5-VL-LoRA-GA \
  --batch_size 4 \
  --num_epochs 10 \
  --max_length 128

swift export \
      --model /root/autodl-tmp/StarBench/output/Qwen2.5-VL-LoRA-Vanilla-merged-2000 \
      --adapters /root/autodl-tmp/StarBench/output/Qwen2.5-VL-LoRA-GD/checkpoint-100 \
      --merge_lora true\
      --model_type qwen2_5_vl


python /root/autodl-tmp/StarBench/baselines/GD_forget.py \
  --model_path /root/autodl-tmp/StarBench/output/Qwen2.5-VL-LoRA-vanilla-2300 \
  --forget_data /root/autodl-tmp/StarBench/forget_set_part.json \
  --retain_data /root/autodl-tmp/StarBench/retain_set.json \
  --forget_batch_size 1 \
  --retain_batch_size 3 \
  --max_length 128 \
  --num_epochs 10 \
  --learning_rate 2e-5

python /root/autodl-tmp/StarBench/baselines/KL_min.py \
  --model_path "/root/autodl-tmp/StarBench/output/Qwen2.5-VL-LoRA-vanilla-2300" \
  --target_model_path "/root/autodl-tmp/StarBench/output/Qwen2.5-VL-LoRA-vanilla-2300" \
  --output_dir "/root/autodl-tmp/StarBench/output_part/Qwen2.5-VL-LoRA-kl—1" \
  --forget_data "/root/autodl-tmp/StarBench/forget_set.json" \
  --retain_data "/root/autodl-tmp/StarBench/retain_set.json" \
  --num_epochs 15 \
  --kl_weight 0.1 \
  --temperature 1.0 \
  --save_steps 50


python /root/autodl-tmp/StarBench/baselines/PO.py \
  --model_path /root/autodl-tmp/StarBench/output/Qwen2.5-VL-LoRA-vanilla-2300 \
  --num_epochs 10 \
  --forget_data "/root/autodl-tmp/StarBench/forget_set_part.json" \
  --retain_data "/root/autodl-tmp/StarBench/retain_set_part.json" \
  --output_dir "/root/autodl-tmp/StarBench/output_part/Qwen2.5-VL-LoRA-PO-part" \
  --forget_batch_size 2 \
  --retain_batch_size 6


python /root/autodl-tmp/StarBench/baselines/NPO.py \
  --model_path /root/autodl-tmp/StarBench/output/Qwen2.5-VL-LoRA-vanilla-2300 \
  --ref_model_path /root/autodl-tmp/StarBench/output/Qwen2.5-VL-LoRA-NPO-reference/checkpoint-2000-merged \
  --output_dir /root/autodl-tmp/StarBench/output/Qwen2.5-VL-LoRA-npo-part \
  --forget_data "/root/autodl-tmp/StarBench/forget_set_part.json" \
  --forget_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_epochs 10 \
  --beta 0.9 \
  --learning_rate 5e-5 \
  --save_steps 50 \
  --max_checkpoints 15

