# run.py - Entry point for SAM with enhanced error handling

import os
import sys
import argparse
import logging
import traceback
from sam import SAM, create_sam_model, SAMTrainer, run_sam

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('sam_runner.log')
    ]
)
logger = logging.getLogger("SAM-Runner")

def main():
    """Main entry point with comprehensive error handling"""
    parser = argparse.ArgumentParser(description='Run or train SAM')
    parser.add_argument('--mode', choices=['interact', 'train'], default='interact', 
                       help='Mode to run SAM in')
    parser.add_argument('--load_path', type=str, default=None, 
                       help='Path to load model from')
    parser.add_argument('--train_data', type=str, default=None, 
                       help='Path to training data')
    parser.add_argument('--eval_data', type=str, default=None,
                       help='Path to evaluation data')
    parser.add_argument('--batch_size', type=int, default=4, 
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3, 
                       help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (uses model default if not specified)')
    parser.add_argument('--multimodal', action='store_true',
                       help='Enable multimodal training')
    parser.add_argument('--hive_mind', action='store_true',
                       help='Enable hive mind capabilities')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save models and logs')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'train':
            success = run_training_mode(args)
            if not success:
                logger.error("Training failed")
                sys.exit(1)
        else:
            run_interactive_mode(args)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

def run_training_mode(args):
    """Run SAM in training mode with robust error handling"""
    try:
        logger.info("Starting SAM in training mode")
        
        # Validate training data path
        if not args.train_data:
            logger.error("No training data path provided")
            return False
            
        if not os.path.exists(args.train_data):
            logger.error(f"Training data file not found: {args.train_data}")
            return False
        
        # Load or create model
        model = None
        if args.load_path and os.path.exists(args.load_path):
            try:
                logger.info(f"Loading existing model from {args.load_path}")
                model = SAM.load(args.load_path)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model from {args.load_path}: {e}")
                logger.info("Creating new model instead")
                model = None
        
        if model is None:
            try:
                logger.info("Creating new SAM model")
                
                # Prepare configuration overrides
                config_overrides = {
                    "initial_hidden_dim": 1536,
                    "initial_num_layers": 8
                }
                
                # Add save directory if specified
                if args.save_dir:
                    os.makedirs(args.save_dir, exist_ok=True)
                    config_overrides["save_dir"] = args.save_dir
                
                # Create model with specified options
                model, _ = create_sam_model(
                    config_overrides=config_overrides,
                    load_vocab=True,
                    hive_mind=args.hive_mind,
                    multimodal=args.multimodal
                )
                
                logger.info(f"Created new SAM model with {sum(p.numel() for p in model.parameters()):,} parameters")
                
                # Load expanded vocabulary safely
                try:
                    if hasattr(model, 'concept_bank') and hasattr(model.concept_bank, 'load_vocabulary'):
                        vocab_path = os.path.join(model.config.save_dir, "sam_expanded_vocab.txt")
                        if os.path.exists(vocab_path):
                            model.concept_bank.load_vocabulary(vocab_path)
                            logger.info("Loaded expanded vocabulary")
                        else:
                            logger.info("No expanded vocabulary file found, using default")
                    else:
                        logger.warning("ConceptBank vocabulary loading not available")
                except Exception as e:
                    logger.warning(f"Could not load expanded vocabulary: {e}")
                
            except Exception as e:
                logger.error(f"Failed to create model: {e}")
                return False

        # Initialize trainer with enhanced configuration
        try:
            logger.info("Initializing SAM trainer")
            
            trainer_config = {
                "model": model,
                "train_data_path": args.train_data,
                "eval_data_path": args.eval_data,
                "batch_size": args.batch_size,
                "num_epochs": args.epochs,
                "multimodal_train": args.multimodal,
                "gradient_clip_val": 1.0,
                "save_every_n_steps": 1000,
                "eval_every_n_steps": 1000,
                "log_every_n_steps": 100
            }
            
            # Add learning rate if specified
            if args.learning_rate:
                trainer_config["learning_rate"] = args.learning_rate
            
            trainer = SAMTrainer(**trainer_config)
            logger.info("Trainer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize trainer: {e}")
            return False

        # Start training
        try:
            logger.info(f"Starting training with:")
            logger.info(f"  - Training data: {args.train_data}")
            logger.info(f"  - Evaluation data: {args.eval_data or 'None'}")
            logger.info(f"  - Batch size: {args.batch_size}")
            logger.info(f"  - Epochs: {args.epochs}")
            logger.info(f"  - Multimodal: {args.multimodal}")
            logger.info(f"  - Hive mind: {args.hive_mind}")
            
            # Run training
            results = trainer.train()
            
            if results:
                logger.info("Training completed successfully!")
                logger.info(f"Final results:")
                logger.info(f"  - Steps completed: {results['steps']}")
                logger.info(f"  - Epochs completed: {results['epochs']}")
                logger.info(f"  - Final loss: {results.get('final_loss', 'N/A')}")
                logger.info(f"  - Best loss: {results['best_loss']}")
                logger.info(f"  - Best step: {results['best_step']}")
                logger.info(f"  - Training duration: {results.get('training_duration', 0):.2f} seconds")
                
                # Save trained model
                try:
                    if args.save_dir:
                        save_path = args.save_dir
                    else:
                        save_path = os.path.join(model.config.save_dir, "trained_model")
                    
                    model.save(save_path)
                    logger.info(f"Trained model saved to: {save_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to save trained model: {e}")
                    return False
                
                return True
            else:
                logger.error("Training returned no results")
                return False
                
        except Exception as e:
            logger.error(f"Training error: {e}")
            logger.error(traceback.format_exc())
            return False
            
    except Exception as e:
        logger.error(f"Error in training mode: {e}")
        logger.error(traceback.format_exc())
        return False

def run_interactive_mode(args):
    """Run SAM in interactive mode"""
    try:
        logger.info("Starting SAM in interactive mode")
        
        # Configure for interaction
        config_overrides = {}
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            config_overrides["save_dir"] = args.save_dir
        
        # Run SAM interactively
        if args.load_path and os.path.exists(args.load_path):
            logger.info(f"Loading model from {args.load_path} for interaction")
            run_sam(load_path=args.load_path)
        else:
            logger.info("Creating new model for interaction")
            
            # Prepare hive config if requested
            hive_config = None
            if args.hive_mind:
                hive_config = {
                    "hive_enabled": True,
                    "hive_identity": "interactive_instance"
                }
            
            run_sam(
                config=type('Config', (), config_overrides)() if config_overrides else None,
                hive_config=hive_config,
                multimodal=args.multimodal
            )
            
    except Exception as e:
        logger.error(f"Error in interactive mode: {e}")
        logger.error(traceback.format_exc())
        raise

def validate_environment():
    """Validate that the environment is properly set up"""
    try:
        # Check for required imports
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Check if SAM module is importable
        from sam import SAM, SAMConfig
        logger.info("SAM module imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"Missing required dependency: {e}")
        return False
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return False

if __name__ == "__main__":
    # Validate environment first
    if not validate_environment():
        logger.error("Environment validation failed")
        sys.exit(1)
    
    # Run main function
    main()
