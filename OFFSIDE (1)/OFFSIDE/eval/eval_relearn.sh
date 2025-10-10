#!/bin/bash
# ===================== Basic Parameters =====================
MODEL_PATH=""
IMAGE_DIR=""
GEN_OUTPUT_DIR=""
CLASSI_OUTPUT_DIR=""
GPT_OUTPUT_DIR=""
MAX_NEW_TOKENS=32

# Parallel Parameters
GEN_BATCH_SIZE=4
GEN_WORKERS=4
CLASSI_BATCH_SIZE=16
CLASSI_WORKERS=16
GPT_BATCH_SIZE=8
GPT_INFER_WORKERS=8
GPT_API_WORKERS=8

# ===================== Dataset List =====================
declare -a GEN_DATASETS=(
"forget_set.json"
"retain_set.json"
"test_set.json"
"relearn_set_enhanced.json"
)
declare -a CLASSI_DATASETS=(
"classification_retain_set.json"
"classification_forget_set.json"
"classification_test_set.json"
"classification_relearn_set.json"
)
declare -a GPT_DATASETS=(
"retain_set.json"
"test_set.json"
"forget_set.json"
"relearn_set_enhanced.json"
)

# ===================== Color Output =====================
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ===================== Record Start Time =====================
start_time=$(date +%s)

echo -e "${BLUE}========== 1. Generation Evaluation ==========${NC}"
for dataset in "${GEN_DATASETS[@]}"; do
  dataset_name=$(basename "$dataset" .json)
  output_dir="$GEN_OUTPUT_DIR/$dataset_name"
  mkdir -p "$output_dir"
  data_path="$IMAGE_DIR/$dataset"
  if [ ! -f "$data_path" ]; then
    echo -e "${RED}Generation dataset does not exist: $data_path${NC}"
    continue
  fi
  echo -e "${GREEN}Evaluating Generation dataset: $dataset_name${NC}"
  python /root/autodl-tmp/StarBench/eval_generation.py \
    --model_path "$MODEL_PATH" \
    --data_path "$data_path" \
    --image_dir "$IMAGE_DIR" \
    --output_dir "$output_dir" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --num_workers "$GEN_WORKERS" \
    --batch_size "$GEN_BATCH_SIZE"
done

echo -e "${BLUE}========== 2. Classification Evaluation ==========${NC}"
for dataset in "${CLASSI_DATASETS[@]}"; do
  DATASET_BASENAME=$(basename "$dataset" .json)
  OUTPUT_DIR="${CLASSI_OUTPUT_DIR}/${DATASET_BASENAME}"
  mkdir -p "$OUTPUT_DIR"
  if [ ! -f "$dataset" ]; then
    echo -e "${RED}Classification dataset does not exist: $dataset${NC}"
    continue
  fi
  echo -e "${GREEN}Evaluating Classification dataset: $DATASET_BASENAME${NC}"
  python /root/autodl-tmp/StarBench/eval_classi.py \
    --data_path="$dataset" \
    --model_path="$MODEL_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --image_dir="$IMAGE_DIR" \
    --num_workers="$CLASSI_WORKERS" \
    --batch_size "$CLASSI_BATCH_SIZE"
done

echo -e "${BLUE}========== 3. GPT Evaluation ==========${NC}"
for dataset in "${GPT_DATASETS[@]}"; do
  dataset_name=$(basename "$dataset" .json)
  output_dir="$GPT_OUTPUT_DIR/${dataset_name}_gpt_eval"
  mkdir -p "$output_dir"
  data_path="$IMAGE_DIR/$dataset"
  if [ ! -f "$data_path" ]; then
    echo -e "${RED}GPT evaluation dataset does not exist: $data_path${NC}"
    continue
  fi
  echo -e "${GREEN}Evaluating GPT dataset: $dataset_name${NC}"
  python /root/autodl-tmp/StarBench/eval_gpt.py \
    --model_path "$MODEL_PATH" \
    --data_path "$data_path" \
    --output_dir "$output_dir" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --batch_size "$GPT_BATCH_SIZE" \
    --inference_workers "$GPT_INFER_WORKERS" \
    --api_workers "$GPT_API_WORKERS"
done

# ===================== Summary Report =====================
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

echo -e "${BLUE}========== All Evaluations Completed ==========${NC}"
echo "Total Time: ${hours} hours ${minutes} minutes ${seconds} seconds"

# Generation Evaluation Markdown
markdown_gen="$GEN_OUTPUT_DIR/evaluation_summary.md"
echo "# Generation Evaluation Report" > "$markdown_gen"
echo "| Dataset | Samples | Total Score | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |" >> "$markdown_gen"
echo "|--------|--------|--------|------|---------|---------|---------|" >> "$markdown_gen"
for dataset in "${GEN_DATASETS[@]}"; do
  dataset_name=$(basename "$dataset" .json)
  summary="$GEN_OUTPUT_DIR/$dataset_name/evaluation_summary.json"
  if [ -f "$summary" ]; then
    total=$(grep -o '"processed_examples": [0-9]*' "$summary" | awk '{print $2}')
    overall=$(grep -o '"overall_score": [0-9.]*' "$summary" | awk '{print $2}')
    bleu=$(grep -o '"avg_bleu": [0-9.]*' "$summary" | awk '{print $2}')
    rouge1=$(grep -o '"avg_rouge1": [0-9.]*' "$summary" | awk '{print $2}')
    rouge2=$(grep -o '"avg_rouge2": [0-9.]*' "$summary" | awk '{print $2}')
    rougeL=$(grep -o '"avg_rougeL": [0-9.]*' "$summary" | awk '{print $2}')
    echo "| $dataset_name | $total | $overall | $bleu | $rouge1 | $rouge2 | $rougeL |" >> "$markdown_gen"
  fi
done

# Classification Evaluation Markdown
markdown_classi="$CLASSI_OUTPUT_DIR/summary_report.md"
echo "# Classification Dataset Evaluation Summary Report" > "$markdown_classi"
echo "| Dataset | Samples | Accuracy |" >> "$markdown_classi"
echo "|--------|--------|--------|" >> "$markdown_classi"
for dataset in "${CLASSI_DATASETS[@]}"; do
  DATASET_BASENAME=$(basename "$dataset" .json)
  SUMMARY_FILE="${CLASSI_OUTPUT_DIR}/${DATASET_BASENAME}/classification_summary.json"
  if [ -f "$SUMMARY_FILE" ]; then
    TOTAL=$(grep -o '"processed_examples": [0-9]*' "$SUMMARY_FILE" | cut -d' ' -f2)
    ACCURACY=$(grep -o '"accuracy": [0-9.]*' "$SUMMARY_FILE" | cut -d' ' -f2)
    echo "| $DATASET_BASENAME | $TOTAL | $ACCURACY |" >> "$markdown_classi"
  fi
done

# GPT Evaluation Markdown
markdown_gpt="$GPT_OUTPUT_DIR/evaluation_gpt_summary.md"
echo "# GPT Evaluation Summary Report" > "$markdown_gpt"
echo "| Dataset | Average GPT Score | Samples |" >> "$markdown_gpt"
echo "|--------|------------|--------|" >> "$markdown_gpt"
for dataset in "${GPT_DATASETS[@]}"; do
  dataset_name=$(basename "$dataset" .json)
  summary_file="$GPT_OUTPUT_DIR/${dataset_name}_gpt_eval/evaluation_summary.json"
  results_file="$GPT_OUTPUT_DIR/${dataset_name}_gpt_eval/inference_results.json"
  if [ -f "$summary_file" ]; then
    avg_gpt=$(grep -o '"avg_gpt_factuality": [0-9.]*' "$summary_file" | cut -d' ' -f2)
    sample_count=$(grep -o '"example_id":' "$results_file" | wc -l)
    echo "| $dataset_name | $avg_gpt | $sample_count |" >> "$markdown_gpt"
  fi
done

echo -e "${GREEN}All evaluation Markdown reports have been generated!${NC}"
echo "Generation: $markdown_gen"
echo "Classification: $markdown_classi"
echo "GPT: $markdown_gpt"
echo ""
echo -e "${BLUE}All evaluation processes completed.${NC}"

# ===================== Average Summary =====================
echo -e "${BLUE}========== 4. Average Statistics for Four Datasets ==========${NC}"

# Generation Average
gen_sets=("forget_set" "retain_set" "test_set" "relearn_set_enhanced")
gen_score_sum=0
gen_bleu_sum=0
gen_rouge1_sum=0
gen_rouge2_sum=0
gen_rougeL_sum=0
gen_count=0
for name in "${gen_sets[@]}"; do
  line=$(grep "| $name" "$GEN_OUTPUT_DIR/evaluation_summary.md")
  if [ -n "$line" ]; then
    score=$(echo "$line" | awk -F'|' '{print $4}' | xargs)
    bleu=$(echo "$line" | awk -F'|' '{print $5}' | xargs)
    rouge1=$(echo "$line" | awk -F'|' '{print $6}' | xargs)
    rouge2=$(echo "$line" | awk -F'|' '{print $7}' | xargs)
    rougeL=$(echo "$line" | awk -F'|' '{print $8}' | xargs)
    gen_score_sum=$(echo "$gen_score_sum + $score" | bc)
    gen_bleu_sum=$(echo "$gen_bleu_sum + $bleu" | bc)
    gen_rouge1_sum=$(echo "$gen_rouge1_sum + $rouge1" | bc)
    gen_rouge2_sum=$(echo "$gen_rouge2_sum + $rouge2" | bc)
    gen_rougeL_sum=$(echo "$gen_rougeL_sum + $rougeL" | bc)
    ((gen_count++))
  fi
done
if [ $gen_count -gt 0 ]; then
  gen_score_mean=$(echo "scale=4; $gen_score_sum/$gen_count" | bc)
  gen_bleu_mean=$(echo "scale=4; $gen_bleu_sum/$gen_count" | bc)
  gen_rouge1_mean=$(echo "scale=4; $gen_rouge1_sum/$gen_count" | bc)
  gen_rouge2_mean=$(echo "scale=4; $gen_rouge2_sum/$gen_count" | bc)
  gen_rougeL_mean=$(echo "scale=4; $gen_rougeL_sum/$gen_count" | bc)
  echo -e "${GREEN}Generation Average for Four Datasets:${NC}"
  echo "Average Total Score: $gen_score_mean"
  echo "Average BLEU: $gen_bleu_mean"
  echo "Average ROUGE-1: $gen_rouge1_mean"
  echo "Average ROUGE-2: $gen_rouge2_mean"
  echo "Average ROUGE-L: $gen_rougeL_mean"
else
  echo -e "${RED}Generation average for four datasets cannot be calculated (data missing)${NC}"
fi

# Classification Average
classi_sets=("classification_retain_set" "classification_forget_set" "classification_test_set" "classification_relearn_set")
classi_acc_sum=0
classi_count=0
for name in "${classi_sets[@]}"; do
  line=$(grep "| $name" "$CLASSI_OUTPUT_DIR/summary_report.md")
  if [ -n "$line" ]; then
    acc=$(echo "$line" | awk -F'|' '{print $4}' | xargs)
    classi_acc_sum=$(echo "$classi_acc_sum + $acc" | bc)
    ((classi_count++))
  fi
done
if [ $classi_count -gt 0 ]; then
  classi_acc_mean=$(echo "scale=4; $classi_acc_sum/$classi_count" | bc)
  echo -e "${GREEN}Classification Average Accuracy for Four Datasets: $classi_acc_mean${NC}"
else
  echo -e "${RED}Classification average for four datasets cannot be calculated (data missing)${NC}"
fi

# GPT Average
gpt_sets=("forget_set" "retain_set" "test_set" "relearn_set_enhanced")
gpt_score_sum=0
gpt_count=0
for name in "${gpt_sets[@]}"; do
  line=$(grep "| $name" "$GPT_OUTPUT_DIR/evaluation_gpt_summary.md")
  if [ -n "$line" ]; then
    score=$(echo "$line" | awk -F'|' '{print $3}' | xargs)
    gpt_score_sum=$(echo "$gpt_score_sum + $score" | bc)
    ((gpt_count++))
  fi
done
if [ $gpt_count -gt 0 ]; then
  gpt_score_mean=$(echo "scale=4; $gpt_score_sum/$gpt_count" | bc)
  echo -e "${GREEN}GPT Average Score for Four Datasets: $gpt_score_mean${NC}"
else
  echo -e "${RED}GPT average for four datasets cannot be calculated (data missing)${NC}"
fi

echo ""
echo -e "${BLUE}All evaluation processes completed.${NC}"
