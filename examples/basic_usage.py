# examples/basic_usage.py
# Basic usage example for SAM

import os
import sys
import logging
import torch

# Add parent directory to path to import SAM
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sam import SAM, create_sam_model

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SAM-Example")

def main():
    # Check if model exists
    if os.path.exists("./data/models/basic_sam"):
        logger.info("Loading existing model...")
        model = SAM.load("./data/models/basic_sam")
    else:
        logger.info("Creating new model with auto-configuration...")
        # Create new model with automatic hardware configuration
        model = SAM.create_with_auto_config()[0]
        
        # Initialize with vocabulary
        logger.info("Loading vocabulary...")
        model.load_claude_vocabulary()
    
    # Print model status
    status = model.get_status()
    logger.info(f"Model ready with {status['model_size']['hidden_dim']} hidden dimensions "
               f"and {status['model_size']['num_layers']} layers")
    logger.info(f"Device: {status['config']['device']}")
    
    # Example interaction
    print("\nSAM is ready. Type your message or 'exit' to quit.")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == "exit":
            break
        
        # Generate response
        response = model.generate(
            input_text=user_input,
            max_length=250,
            temperature=0.8
        )
        
        print(f"\nSAM: {response}")
    
    # Save model
    save_path = model.save("./data/models/basic_sam")
    logger.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
