# run.py - Entry point for SAM

import os
import argparse
from sam import SAM, SAMConfig, create_sam_model, run_sam, SAMTrainer

def main():
    parser = argparse.ArgumentParser(description='Run or train SAM')
    parser.add_argument('--mode', choices=['interact', 'train'], default='interact', help='Mode to run SAM in')
    parser.add_argument('--load_path', type=str, default=None, help='Path to load model from')
    parser.add_argument('--train_data', type=str, default=None, help='Path to training data')
    parser.add_argument('--eval_data', type=str, default=None, help='Path to evaluation data')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs for training')
    args = parser.parse_args()
    
    if args.mode == 'train':
        if args.load_path and os.path.exists(args.load_path):
            # Load existing model
            model = SAM.load(args.load_path)
            print(f"Loaded model from {args.load_path}")
        else:
            # Create new model
            model, _ = create_sam_model(config_overrides={
                "initial_hidden_dim": 1536,  # Start moderate size for training
                "initial_num_layers": 16     # Begin with reasonable depth
            })
            print("Created new model")
        
        # Initialize trainer
        trainer = SAMTrainer(
            model=model,
            train_data_path=args.train_data,
            eval_data_path=args.eval_data,
            batch_size=args.batch_size,
            num_epochs=args.epochs
        )
        
        # Train model
        print(f"Starting training on {args.train_data}")
        trainer.train()
    else:
        # Interactive mode
        run_sam(load_path=args.load_path)

if __name__ == "__main__":
    main()
