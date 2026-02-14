import os
import json
import torch
import re
import argparse
import concurrent.futures
from PIL import Image
from tqdm import tqdm

from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL Model Multiple-Choice Evaluation Script (Parallel Version)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Model name")
    parser.add_argument("--model_path", type=str, 
                        default="/root/autodl-tmp/StarBench/Qwen2.5-VL-LoRA-Vanilla-merged-2000-GA",
                        help="Model path")
    parser.add_argument("--data_path", type=str, 
                        default="/root/autodl-tmp/StarBench/classification_forget_set.json",
                        help="Dataset path")
    parser.add_argument("--image_dir", type=str, default="/root/autodl-tmp/StarBench",
                        help="Image directory")
    parser.add_argument("--output_dir", type=str, 
                        default="/root/autodl-tmp/StarBench/classification_evaluation_results",
                        help="Output directory")
    parser.add_argument("--max_new_tokens", type=int, default=32,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker threads for parallel processing")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Number of samples per batch")
    return parser.parse_args()


def load_model(model_path):
    print("Loading model from:", model_path)
    
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

def create_few_shot_prompt(options):

    options_text = ""
    for key, value in options.items():
        options_text += f"{key}: {value}\n"
    
    few_shot_prompt = f"""Please look at this image and answer the question by selecting the correct option from the choices below.

Please respond with only the letter of the correct answer (e.g., 'The answer is A' or just 'A').

Options:
{options_text}
"""
    return few_shot_prompt

def generate_response(model, processor, query, options, image_path, max_new_tokens=128
                      ):
    try:
        prompt = create_few_shot_prompt(options)
        full_query = f"{query}\n\n{prompt}"
        
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
                    {"type": "text", "text": full_query}
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
        print(f"Error processing example: {str(e)}")
        return None

def extract_answer(response):

    patterns = [
        r"[Tt]he answer is ([A-D])",  
        r"^([A-D])$",                 
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1)
            
    return None

def process_example(args):
    i, example, model, processor, image_dir, max_new_tokens = args
    
    try:
        question = example["Question"]
        options = example["Options"]
        correct_answer = example["Correct_Answer"]
        
        image_path = os.path.join(image_dir, example['images'])
        
        if not os.path.exists(image_path):
            alt_path = os.path.join(image_dir, os.path.basename(example['images']))
            if os.path.exists(alt_path):
                image_path = alt_path
            else:
                return {
                    'example_id': i,
                    'error': f"Image not found at {image_path}",
                    'status': 'skipped'
                }
        
        prediction_raw = generate_response(model, processor, question, options, image_path, max_new_tokens)
        
        if prediction_raw:
            predicted_answer = extract_answer(prediction_raw)
            

            is_correct = predicted_answer == correct_answer
            
            return {
                'example_id': i,
                'question': question,
                'options': options,
                'correct_answer': correct_answer,
                'predicted_answer': predicted_answer,
                'full_prediction': prediction_raw,
                'is_correct': is_correct,
                'image_path': example['images'],
                'status': 'success'
            }
        else:
            return {
                'example_id': i,
                'error': "Failed to generate response",
                'status': 'failed'
            }
    
    except Exception as e:
        return {
            'example_id': i,
            'error': str(e),
            'status': 'error'
        }

def process_batch(batch_args, results_list):
    batch_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch_args)) as executor:
        batch_futures = [executor.submit(process_example, arg) for arg in batch_args]
        
        for future in concurrent.futures.as_completed(batch_futures):
            try:
                result = future.result()
                if result['status'] == 'success':
                    results_list.append(result)
                    batch_results.append(result)
            except Exception as e:
                print(f"Batch processing error: {e}")
    
    return batch_results

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading dataset from: {args.data_path}")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    total_examples = len(dataset)
    print(f"Dataset loaded with {total_examples} examples")
    
    model, processor = load_model(args.model_path)
    
    results = []
    
    batch_size = args.batch_size
    num_workers = min(args.num_workers, batch_size)  
    
    print(f"Starting inference with {num_workers} workers and batch size {batch_size}...")
    
    progress_bar = tqdm(total=total_examples, desc="processing")
    
    for start_idx in range(0, total_examples, batch_size):
        end_idx = min(start_idx + batch_size, total_examples)
        current_batch_size = end_idx - start_idx
        
        batch_args = [
            (i, dataset[i], model, processor, args.image_dir, args.max_new_tokens) 
            for i in range(start_idx, end_idx)
        ]
        
        batch_results = process_batch(batch_args, results)
        
        progress_bar.update(current_batch_size)
        
        successful_results = [r for r in batch_results if r.get('status') == 'success']
        if successful_results:
            correct_in_batch = sum(1 for r in successful_results if r.get('is_correct', False))
            batch_accuracy = correct_in_batch / len(successful_results) if successful_results else 0
            
            correct_total = sum(1 for r in results if r.get('is_correct', False))
            total_accuracy = correct_total / len(results) if results else 0
            
            progress_bar.set_postfix(
                batch=f"{start_idx//batch_size+1}/{(total_examples+batch_size-1)//batch_size}", 
                batch_acc=f"{batch_accuracy:.4f}",
                total_acc=f"{total_accuracy:.4f}"
            )
    
    progress_bar.close()
    
    correct_count = sum(1 for r in results if r.get('is_correct', False))
    accuracy = correct_count / len(results) if results else 0
    
    with open(os.path.join(args.output_dir, 'classification_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    summary = {
        'total_examples': total_examples,
        'processed_examples': len(results),
        'correct_predictions': correct_count,
        'accuracy': float(accuracy),
        'dataset_path': args.data_path,
        'model_path': args.model_path
    }
    
    with open(os.path.join(args.output_dir, 'classification_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\nEvaluation completed. Results saved to {args.output_dir}")
    print(f"Accuracy: {accuracy:.4f} ({correct_count}/{len(results)})")

if __name__ == "__main__":
    main()