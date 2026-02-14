######GA 2000 step
###For complete unlearning
python /root/autodl-tmp/OFFSIDE/OFFSIDE/baselines/GA_forget.py \
  --model_path /root/autodl-tmp/output/output/Qwen2.5-VL-7B-LoRA-2000 \
  --data_path /root/autodl-tmp/OFFSIDE/OFFSIDE/data/complete_unlearning_data/forget_set.json\
  --output_dir /root/autodl-tmp/OFFSIDE/OFFSIDE/output/Qwen2.5-VL-LoRA-GA \
  --batch_size 4 \
  --num_epochs 10 \
  --max_length 128
###Merge the model after unlearning
swift export \
  --model /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct \
      --adapters /root/autodl-tmp/OFFSIDE/OFFSIDE/output/Qwen2.5-VL-LoRA-GD/checkpoint-100 \
      --merge_lora true\
      --model_type qwen2_5_vl


python /root/autodl-tmp/OFFSIDE/OFFSIDE/baselines/GD_forget.py \
  --model_path /root/autodl-tmp/output/output/Qwen2.5-VL-7B-LoRA-2000 \
  --forget_data /root/autodl-tmp/OFFSIDE/OFFSIDE/data/complete_unlearning_data/forget_set.json \
  --retain_data /root/autodl-tmp/OFFSIDE/OFFSIDE/data/complete_unlearning_data/retain_set.json \
  --forget_batch_size 1 \
  --retain_batch_size 3 \
  --max_length 128 \
  --num_epochs 10 \
  --learning_rate 2e-5

python /root/autodl-tmp/OFFSIDE/OFFSIDE/baselines/KL_min.py \
  --model_path /root/autodl-tmp/output/output/Qwen2.5-VL-7B-LoRA-2000 \
  --target_model_path "/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct" \
  --output_dir "" \
  --forget_data "/root/autodl-tmp/OFFSIDE/OFFSIDE/data/complete_unlearning_data/forget_set.json" \
  --retain_data "/root/autodl-tmp/OFFSIDE/OFFSIDE/data/complete_unlearning_data/retain_set.json" \
  --num_epochs 15 \
  --kl_weight 0.1 \
  --temperature 1.0 \
  --save_steps 50


python /root/autodl-tmp/OFFSIDE/OFFSIDE/baselines/PO.py \
  --model_path /root/autodl-tmp/output/output/Qwen2.5-VL-7B-LoRA-2000 \
  --num_epochs 10 \
  --forget_data "/root/autodl-tmp/OFFSIDE/OFFSIDE/data/complete_unlearning_data/forget_set.json" \
  --retain_data "/root/autodl-tmp/OFFSIDE/OFFSIDE/data/complete_unlearning_data/retain_set.json" \
  --output_dir "" \
  --forget_batch_size 2 \
  --retain_batch_size 6


python /root/autodl-tmp/OFFSIDE/OFFSIDE/baselines/NPO.py \
  --model_path /root/autodl-tmp/output/output/Qwen2.5-VL-7B-LoRA-2000 \
  --ref_model_path /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct \
  --output_dir  \
  --forget_data "/root/autodl-tmp/OFFSIDE/OFFSIDE/data/complete_unlearning_data/forget_set.json" \
  --forget_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_epochs 10 \
  --beta 0.9 \
  --learning_rate 5e-5 \
  --save_steps 50 \
  --max_checkpoints 15

###For other settings, you can simply change the data path.