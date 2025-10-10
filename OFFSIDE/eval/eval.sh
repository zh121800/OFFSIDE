#!/bin/bash

# ===================== Basic Parameters =====================
MODEL_PATH=""
IMAGE_DIR="your image dir"
GEN_OUTPUT_DIR=""
CLASSI_OUTPUT_DIR=""
GPT_OUTPUT_DIR=""
MAX_NEW_TOKENS=32

# Parallel Parameters
GEN_BATCH_SIZE=2
GEN_WORKERS=2
CLASSI_BATCH_SIZE=8
CLASSI_WORKERS=8
GPT_BATCH_SIZE=4
GPT_INFER_WORKERS=4
GPT_API_WORKERS=4

# ===================== Dataset List =====================
declare -a GEN_DATASETS=(
#for example
  "forget_set.json"
  "retain_set.json"
  "test_set.json"
)

declare -a CLASSI_DATASETS=(
#for example
    "classification_retain_set.json"
    "classification_forget_set.json"
    "classification_test_set.json"
)

declare -a GPT_DATASETS=(
#for example
  "forget_set.json"
  "test_set.json"
  "retain_set.json"
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
    --batch_size="$CLASSI_BATCH_SIZE"
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
echo "Total time: ${hours} hours ${minutes} minutes ${seconds} seconds"

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
