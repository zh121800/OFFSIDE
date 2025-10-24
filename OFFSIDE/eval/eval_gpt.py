import os
import json
import torch
import time
import argparse
import concurrent.futures
from PIL import Image
from tqdm import tqdm
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from openai import OpenAI

client = OpenAI(
    base_url=
    api_key=""
)


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL Model Evaluation Script (GPT scoring only)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Model name")
    parser.add_argument("--model_path", type=str, 
                        default="/root/autodl-tmp/StarBench/Qwen2.5-VL-LoRA-Vanilla-merged-2000-GA",
                        help="Model path")
    parser.add_argument("--data_path", type=str, 
                        default="/root/autodl-tmp/StarBench/enhanced_finetuneset.json",
                        help="Dataset path")
    parser.add_argument("--image_dir", type=str, default="/root/autodl-tmp/StarBench",
                        help="Image directory")
    parser.add_argument("--output_dir", type=str, 
                        default="/root/autodl-tmp/StarBench/GPT_evaluation_results",
                        help="Output directory")
    parser.add_argument("--max_new_tokens", type=int, default=32,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--gpt_model", type=str, default="gpt-4o-mini",
                        help="GPT model for scoring")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of samples per batch")
    parser.add_argument("--inference_workers", type=int, default=8,
                        help="Number of parallel inference threads")
    parser.add_argument("--api_workers", type=int, default=8,
                        help="Number of parallel API call threads")
    parser.add_argument("--quiet", action="store_true",
                        help="Quiet mode, reduce output information")
    return parser.parse_args()

def gpt_evaluate_factuality(question, generated_answer, ground_truth, image_id, gpt_model="gpt-4o-mini", task_type="generation"):

    prompt = f"""You will be provided with a question and two answers: a generated answer and a ground truth answer. Your task is to evaluate the factuality of the "generated_answer" against the "ground_truth". 

Please assign a factuality score from 1 to 10 based on the following criteria:

1. Factuality (core importance):
• 10-9: The generated response is fully factually correct and has the same meaning as the ground truth, even if phrased differently.
• 8-7: The response is mostly correct but may be missing minor details or contain slightly less important deviations.
• 6-5: The response is partially correct but has a noticeable factual error or significant missing information.
• 4-3: The response has major factual errors or lacks crucial elements of the ground truth.
• 2-1: The response is nonsensical, completely incorrect, or irrelevant.

2. Relevance and Detail:
• More detail does not always improve the score; added details should be factually relevant.
• If the generated response contains excessive or irrelevant details, lower the score accordingly.

3. Fluency and Language Requirements:
• The response must be in English. If it's not in English, reduce the score according to how much this affects comprehension.
• If the response contains garbled text, random symbols, or is completely incomprehensible, assign a score of 0.
• Poor grammar or awkward phrasing should result in a score reduction proportional to how much it affects understanding.

- Task Type: {task_type.capitalize()}
- Image ID: {image_id}
- Question: {question}
- Generated Answer: {generated_answer}
- Ground Truth: {ground_truth}

Please evaluate the factuality of the generated response based on the rubric above, and return a score (1-10) along with a short justification.

Return your response in JSON format only:
{{
"factuality_score": [score from 1-10 as a number, or 0 if completely incomprehensible],
"justification": "[Your brief justification, including comments on factuality, relevance, and fluency]"
}}

"""

    max_retries = 3
    for attempt in range(max_retries):
        try:

            response = client.chat.completions.create(
                model=gpt_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2  
            )
            

            result = json.loads(response.choices[0].message.content)
            

            if "factuality_score" not in result:
                result["factuality_score"] = 0
            if "justification" not in result:
                result["justification"] = "Score parsing failed"
                
            return result
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1) 
                print(f"GPT scoring error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"GPT scoring failed, maximum retries reached: {e}")
                return {"factuality_score": 0, "justification": f"Scoring failed: {str(e)}"}

def load_model(model_path):

    print("Loading model:", model_path)
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    return model, processor

def generate_response(model, processor, query, image_path, max_new_tokens=32):

    try:

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                        "resized_height": 256,  
                        "resized_width": 256
                    },
                    {"type": "text", "text": query}
                ],
            }
        ]
        

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        

        image_inputs, video_inputs = process_vision_info(messages)
        

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
        

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
            
        return response
        
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return None

def process_sample(args):

    model, processor, example, idx, image_dir, max_new_tokens = args
    
    try:

        user_content = example['messages'][0]['content']
        ground_truth = example['messages'][1]['content']
        

        image_path = None
        if 'images' in example:
            image_path = os.path.join(image_dir, example['images'])
        elif 'image' in example:
            image_path = os.path.join(image_dir, example['image'])
        else:
            return None
        

        if not os.path.exists(image_path):

            image_key = 'images' if 'images' in example else 'image'
            alt_path = os.path.join(image_dir, os.path.basename(example[image_key]))
            if os.path.exists(alt_path):
                image_path = alt_path
            else:
                return None
        

        prediction = generate_response(model, processor, user_content, image_path, max_new_tokens)
        
        if prediction:
            return {
                'example_id': idx,
                'user_content': user_content,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'image_path': image_path,
                'image_id': os.path.basename(image_path)
            }
        return None
        
    except Exception as e:
        print(f"Error processing sample {idx}: {e}")
        return None

def evaluate_prediction(pred_item, gpt_model):

    try:
        gpt_result = gpt_evaluate_factuality(
            pred_item['user_content'], 
            pred_item['prediction'], 
            pred_item['ground_truth'], 
            pred_item['image_id'],
            gpt_model,
            "generation"
        )
        
        pred_item['gpt_factuality_score'] = gpt_result.get('factuality_score', 0)
        pred_item['gpt_justification'] = gpt_result.get('justification', "Not provided")
        return pred_item
        
    except Exception as e:
        print(f"Error evaluating sample {pred_item['example_id']}: {e}")
        pred_item['gpt_factuality_score'] = 0
        pred_item['gpt_justification'] = f"Evaluation failed: {str(e)}"
        return pred_item

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading dataset: {args.data_path}")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Dataset loaded, total {len(dataset)} samples")
    
    model, processor = load_model(args.model_path)

    inference_results = []
    

    batch_size = args.batch_size
    inference_workers = min(args.inference_workers, batch_size) 
    

    print(f"Stage 1: Parallel batch inference (batch_size={batch_size}, workers={inference_workers})...")
    

    pbar = tqdm(total=len(dataset), desc="Model inference progress", disable=args.quiet)
    

    total_batches = (len(dataset) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(dataset))
        batch = dataset[batch_start:batch_end]
        

        batch_args = [(model, processor, example, batch_start+i, args.image_dir, args.max_new_tokens) 
                      for i, example in enumerate(batch)]
        

        batch_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=inference_workers) as executor:
            futures = {executor.submit(process_sample, arg): i for i, arg in enumerate(batch_args)}
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    batch_results.append(result)
                

                pbar.update(1)
        

        inference_results.extend(batch_results)
        

        if batch_results:
            with open(os.path.join(args.output_dir, f'inference_batch_{batch_idx+1}_results.json'), 'w', encoding='utf-8') as f:
                json.dump(batch_results, f, ensure_ascii=False, indent=2)
    
    pbar.close() 
    

    with open(os.path.join(args.output_dir, 'all_inference_results.json'), 'w', encoding='utf-8') as f:
        json.dump(inference_results, f, ensure_ascii=False, indent=2)
    
    print(f"Model inference stage completed, successfully processed {len(inference_results)}/{len(dataset)} samples")
    
    if not inference_results:
        print("No successful inference results, program terminated")
        return
    
    print(f"Stage 2: Parallel API scoring (workers={args.api_workers})...")
    final_results = []

    pbar = tqdm(total=len(inference_results), desc="API scoring progress", disable=args.quiet)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.api_workers) as executor:

        future_to_idx = {
            executor.submit(evaluate_prediction, item, args.gpt_model): i 
            for i, item in enumerate(inference_results)
        }
        

        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                final_results.append(result)
                

                pbar.update(1)
                

                if len(final_results) % 10 == 0:
                    with open(os.path.join(args.output_dir, f'evaluation_intermediate_{len(final_results)}.json'), 'w', encoding='utf-8') as f:
                        json.dump(final_results, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Error processing evaluation result {idx}: {e}")
                pbar.update(1)  
    
    pbar.close()  
    

    with open(os.path.join(args.output_dir, 'inference_results.json'), 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    

    if final_results:
        gpt_scores = [r.get('gpt_factuality_score', 0) for r in final_results]
        avg_gpt = sum(gpt_scores) / len(gpt_scores) if gpt_scores else 0
    
        summary = {
            'total_examples': len(final_results),
            'avg_gpt_factuality': float(avg_gpt),
            'dataset_path': args.data_path,
            'model_path': args.model_path,
            'batch_size': args.batch_size,
            'inference_workers': args.inference_workers,
            'api_workers': args.api_workers
        }
        
        with open(os.path.join(args.output_dir, 'evaluation_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\nEvaluation completed. Results saved to {args.output_dir}")
        print(f"GPT factuality score: {avg_gpt:.2f}/10")
        print(f"Total evaluated samples: {len(final_results)}")
    else:
        print("No evaluation results generated.")

if __name__ == "__main__":
    main()