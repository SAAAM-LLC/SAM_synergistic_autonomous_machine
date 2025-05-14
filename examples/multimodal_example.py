# examples/multimodal_example.py
# Example demonstrating SAM's multimodal capabilities

import os
import sys
import logging
import torch
import numpy as np
from PIL import Image
import io
import time

# Add parent directory to path to import SAM
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sam import SAM, SAMConfig, create_sam_model

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SAM-Multimodal-Example")

def load_image(image_path):
    """Load image and convert to tensor"""
    try:
        # Load image
        image = Image.open(image_path)
        
        # Resize to standard size
        image = image.resize((224, 224))
        
        # Convert to numpy array and normalize
        image_array = np.array(image) / 255.0
        
        # Convert to tensor
        image_tensor = torch.tensor(image_array, dtype=torch.float32)
        
        # Add batch dimension if needed
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Flatten features for simple example
        # In a real implementation, you would use a proper image encoder
        features = image_tensor.flatten()
        
        return features
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        return None

def create_multimodal_sam():
    """Create a SAM instance with multimodal capabilities"""
    # Create config with multimodal enabled
    config = SAMConfig()
    config.multimodal_enabled = True
    config.image_dim = 150528  # 224x224x3 flattened
    config.audio_dim = 1024
    config.save_dir = "./data/multimodal"
    
    # Create multimodal model
    model, _ = create_sam_model(
        config_overrides=vars(config),
        multimodal=True
    )
    
    return model

def main():
    """Main function to demonstrate multimodal capabilities"""
    logger.info("Creating multimodal SAM instance...")
    
    # Create or load model
    if os.path.exists("./data/multimodal/model"):
        model = SAM.load("./data/multimodal/model")
        logger.info("Loaded existing multimodal model")
    else:
        model = create_multimodal_sam()
        logger.info("Created new multimodal model")
    
    # Create example directory for images
    os.makedirs("./examples/images", exist_ok=True)
    
    # Check if we have a test image
    test_image_path = "./examples/images/test_image.jpg"
    if not os.path.exists(test_image_path):
        logger.info("No test image found. Please add an image to ./examples/images/test_image.jpg")
    
    # Simple interaction loop
    print("\nMultimodal SAM Demo")
    print("-------------------")
    print("Available commands:")
    print("  'text [your message]' - Text-only mode")
    print("  'image [path]' - Process image at path")
    print("  'both [path] [your message]' - Process both image and text")
    print("  'stats' - Show model statistics")
    print("  'exit' - Exit the demo")
    
    while True:
        user_input = input("\nCommand: ")
        
        if user_input.lower() == "exit":
            break
        
        parts = user_input.split(" ", 1)
        command = parts[0].lower()
        
        if command == "text" and len(parts) > 1:
            # Text-only mode
            text = parts[1]
            
            # Process with SAM
            print("\nProcessing text...")
            response = model.generate(
                input_text=text,
                max_length=250,
                temperature=0.8,
                modality="text"
            )
            
            print(f"\nSAM: {response}")
            
        elif command == "image" and len(parts) > 1:
            # Image mode
            image_path = parts[1]
            
            if not os.path.exists(image_path):
                print(f"Error: Image not found at {image_path}")
                continue
            
            # Load image
            print("\nLoading image...")
            image_tensor = load_image(image_path)
            
            if image_tensor is None:
                print("Error processing image")
                continue
            
            # Process with SAM
            print("Processing image...")
            
            # For demo purposes, we'll just generate some text about the image
            # In a real implementation, you would do actual multimodal processing
            response = model.generate(
                input_text="This is an image",
                max_length=250,
                temperature=0.8,
                modality="image",
                image_data=image_tensor
            )
            
            print(f"\nSAM: {response}")
            
        elif command == "both" and len(parts) > 1:
            # Both image and text
            both_parts = parts[1].split(" ", 1)
            
            if len(both_parts) < 2:
                print("Error: Format should be 'both [image_path] [text message]'")
                continue
            
            image_path = both_parts[0]
            text = both_parts[1]
            
            if not os.path.exists(image_path):
                print(f"Error: Image not found at {image_path}")
                continue
            
            # Load image
            print("\nLoading image...")
            image_tensor = load_image(image_path)
            
            if image_tensor is None:
                print("Error processing image")
                continue
            
            # Process with SAM
            print("Processing image and text together...")
            response = model.generate(
                input_text=text,
                max_length=250,
                temperature=0.8,
                modality="multimodal",
                image_data=image_tensor
            )
            
            print(f"\nSAM: {response}")
            
        elif command == "stats":
            # Show model statistics
            status = model.get_status()
            print("\nSAM Multimodal Status:")
            print(f"  Model size: {status['model_size']['hidden_dim']} hidden dims, {status['model_size']['num_layers']} layers")
            
            if status.get('multimodal'):
                print("  Multimodal stats:")
                for modality, count in status['multimodal']['modality_counts'].items():
                    print(f"    - {modality}: {count} concepts")
                print(f"  Current modality: {status['multimodal']['current_modality']}")
        else:
            print("Unrecognized command. Try 'text', 'image', 'both', 'stats', or 'exit'.")
    
    # Save model before exit
    os.makedirs("./data/multimodal", exist_ok=True)
    model.save("./data/multimodal/model")
    logger.info("Model saved. Exiting.")

if __name__ == "__main__":
    main()
