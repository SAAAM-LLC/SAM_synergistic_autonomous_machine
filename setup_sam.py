# setup_sam.py - Set up training environment for SAM

import os
import json
import argparse
import glob
import random
from tqdm import tqdm

def create_directory_structure():
    """Create the necessary directories for SAM"""
    dirs = [
        "./data",
        "./data/checkpoints",
        "./data/raw",
        "./data/processed"
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def process_text_file(file_path, output_path):
    """Process a text file into SAM-compatible JSON format"""
    samples = []
    
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
        
        # Split by potential separators (paragraphs, documents, etc.)
        chunks = []
        if "\n\n" in content:
            # Split by double newline (paragraphs)
            chunks = [c for c in content.split("\n\n") if len(c) > 50]
        else:
            # Use sliding window approach for long text
            words = content.split()
            window_size = 100
            stride = 50
            
            for i in range(0, len(words) - window_size, stride):
                chunk = " ".join(words[i:i+window_size])
                chunks.append(chunk)
        
        # Create samples
        for chunk in chunks:
            if len(chunk) > 50:  # Minimum length check
                samples.append({"text": chunk})
    
    return samples

def process_jsonl_file(file_path, output_path):
    """Process a JSONL file into SAM-compatible JSON format"""
    samples = []
    
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                
                # Handle different formats
                if isinstance(data, dict):
                    if "text" in data:
                        samples.append({"text": data["text"]})
                    elif "content" in data:
                        samples.append({"text": data["content"]})
                    elif "instruction" in data and "output" in data:
                        text = data["instruction"]
                        if "input" in data and data["input"]:
                            text += f"\n\n{data['input']}"
                        text += f"\n\n{data['output']}"
                        samples.append({"text": text})
                    elif "prompt" in data and "response" in data:
                        text = f"{data['prompt']}\n\n{data['response']}"
                        samples.append({"text": text})
                    elif "messages" in data and isinstance(data["messages"], list):
                        # Chat format
                        messages = data["messages"]
                        text = ""
                        for msg in messages:
                            if "role" in msg and "content" in msg:
                                text += f"{msg['role'].capitalize()}: {msg['content']}\n\n"
                        samples.append({"text": text})
            except json.JSONDecodeError:
                continue
    
    return samples

def process_directory(input_dir, output_dir):
    """Process all files in a directory"""
    all_samples = []
    
    # Process text files
    for file_path in tqdm(glob.glob(f"{input_dir}/**/*.txt", recursive=True)):
        samples = process_text_file(file_path, output_dir)
        all_samples.extend(samples)
        print(f"Processed {file_path}: {len(samples)} samples")
    
    # Process JSON files
    for file_path in tqdm(glob.glob(f"{input_dir}/**/*.json", recursive=True)):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            if "text" in item:
                                all_samples.append({"text": item["text"]})
                            elif "content" in item:
                                all_samples.append({"text": item["content"]})
                            # Add more format handling as needed
        except:
            continue
    
    # Process JSONL files
    for file_path in tqdm(glob.glob(f"{input_dir}/**/*.jsonl", recursive=True)):
        samples = process_jsonl_file(file_path, output_dir)
        all_samples.extend(samples)
        print(f"Processed {file_path}: {len(samples)} samples")
    
    # Save all samples
    random.shuffle(all_samples)
    
    # Split into train and eval
    split_point = int(len(all_samples) * 0.95)  # 95% train, 5% eval
    train_samples = all_samples[:split_point]
    eval_samples = all_samples[split_point:]
    
    with open(f"{output_dir}/train.json", 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    
    with open(f"{output_dir}/eval.json", 'w', encoding='utf-8') as f:
        json.dump(eval_samples, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(train_samples)} training samples and {len(eval_samples)} evaluation samples")

def create_initial_config():
    """Create initial configuration file"""
    config = {
        "initial_char_dim": 256,
        "initial_hidden_dim": 768,  # Starting smaller for Titan X Pascal
        "initial_num_layers": 8,    # Starting with fewer layers
        "max_position_embeddings": 4096,
        "concept_memory_size": 50000,
        "concept_dim": 768,
        "thought_dim": 1024,
        "pattern_memory_capacity": 10000,
        "save_dir": "./data",
        "experiences_path": "./data/experiences.json",
        "concepts_path": "./data/concepts.json",
        "growth_log_path": "./data/growth_log.json",
        "communication_style": "claude_unwrapped"
    }
    
    with open("./data/initial_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Created initial configuration file")

def main():
    parser = argparse.ArgumentParser(description='Set up SAM training environment')
    parser.add_argument('--data_dir', type=str, default='./input_data', 
                      help='Directory containing raw data files')
    args = parser.parse_args()
    
    # Create directory structure
    create_directory_structure()
    
    # Process data
    process_directory(args.data_dir, "./data/processed")
    
    # Create initial configuration
    create_initial_config()
    
    print("\nSetup complete! You can now train SAM with:")
    print("python run.py --mode train --train_data ./data/processed/train.json --eval_data ./data/processed/eval.json")

if __name__ == "__main__":
    main()
