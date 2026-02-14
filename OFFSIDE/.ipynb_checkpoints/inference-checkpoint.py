import os
import torch
import json
from datetime import datetime
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


model_paths = [

]
results_dir = "/root/autodl-tmp/model_results"
os.makedirs(results_dir, exist_ok=True)
all_results = {}
image_path = ""
query_text = "How old is the player in the image?"

for model_path in model_paths:
    print(f"\nprocessing model: {model_path}")
    
    try:
        model_name = os.path.basename(model_path.rstrip('/'))
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        
        processor = AutoProcessor.from_pretrained(model_path)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": query_text},
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
        inputs = inputs.to("cuda")
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        print('------------------------------------------begin------------------------------------------')
        print(output_text)
        print('------------------------------------------end------------------------------------------')
    
        all_results[model_name] = output_text[0]
        
        del model
        del processor
        del inputs
        del generated_ids
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error processing model {model_path}: {e}")
        all_results[model_name] = f"Error: {str(e)}"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = os.path.join(results_dir, f"model_results_{timestamp}.json")
with open(results_file, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)

txt_file = os.path.join(results_dir, f"model_results_{timestamp}.txt")
with open(txt_file, "w", encoding="utf-8") as f:
    f.write(f"Image: {image_path}\n")
    f.write(f"Question: {query_text}\n\n")
    
    for model_name, result in all_results.items():
        f.write(f"Model: {model_name}\n")
        f.write(f"Answer: {result}\n")
        f.write("-" * 80 + "\n\n")

print(f"\nAll results saved to: {results_file} and {txt_file}")