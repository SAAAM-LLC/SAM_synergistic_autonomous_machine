# examples/hive_example.py
# Example demonstrating SAM's hive mind capabilities

import os
import sys
import logging
import torch
import time
import threading

# Add parent directory to path to import SAM
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sam import SAM, SAMConfig, create_sam_model

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SAM-Hive-Example")

def run_server():
    """Run SAM in server mode"""
    logger.info("Creating server instance...")
    
    # Create server config
    server_config = SAMConfig()
    server_config.hive_enabled = True
    server_config.hive_server_mode = True
    server_config.hive_identity = "hive_server"
    server_config.save_dir = "./data/hive/server"
    
    # Create smaller model for server
    server_config.initial_hidden_dim = 768
    server_config.initial_num_layers = 8
    
    # Create server model
    server_model, _ = create_sam_model(
        config_overrides=vars(server_config),
        hive_mind=True
    )
    
    # Save server model
    os.makedirs(server_config.save_dir, exist_ok=True)
    server_model.save(os.path.join(server_config.save_dir, "initial"))
    
    # Start server services
    server_model.start_services()
    
    logger.info("Server started. Press Ctrl+C to stop.")
    
    try:
        # Keep running
        while True:
            time.sleep(60)
            
            # Print status
            if hasattr(server_model, 'hive_synchronizer'):
                stats = server_model.hive_synchronizer.get_sync_stats()
                logger.info(f"Server status: {stats['connected_instances']} connected instances, "
                           f"{stats['sync_count']} syncs")
    except KeyboardInterrupt:
        logger.info("Stopping server...")
        server_model.stop_services()
        logger.info("Server stopped.")

def run_client(client_id):
    """Run SAM in client mode"""
    logger.info(f"Creating client instance {client_id}...")
    
    # Create client config
    client_config = SAMConfig()
    client_config.hive_enabled = True
    client_config.hive_server_mode = False
    client_config.hive_server_url = "http://localhost:8765"
    client_config.hive_identity = f"client_{client_id}"
    client_config.save_dir = f"./data/hive/client_{client_id}"
    
    # Create client model
    client_model, _ = create_sam_model(
        config_overrides=vars(client_config),
        hive_mind=True
    )
    
    # Save client model
    os.makedirs(client_config.save_dir, exist_ok=True)
    client_model.save(os.path.join(client_config.save_dir, "initial"))
    
    # Start client services
    client_model.start_services()
    
    logger.info(f"Client {client_id} started. Enter messages or 'exit' to quit.")
    
    # Simple interaction loop
    private_mode = False
    
    try:
        while True:
            mode_str = " (private)" if private_mode else ""
            user_input = input(f"\nYou{mode_str}: ")
            
            if user_input.lower() == "exit":
                break
            elif user_input.lower() == "private":
                private_mode = not private_mode
                mode_str = "enabled" if private_mode else "disabled"
                print(f"\nSAM: Private mode {mode_str}.")
                continue
            elif user_input.lower() == "sync":
                print("\nSAM: Forcing sync with hive mind...")
                if hasattr(client_model, 'hive_synchronizer'):
                    client_model.hive_synchronizer._sync_with_server()
                continue
            elif user_input.lower() == "stats":
                status = client_model.get_status()
                print("\nSAM: Current stats:")
                print(f"  Concepts: {status['concept_stats']['total_concepts']}")
                
                if status['hive_mind']:
                    print("\nHive Mind Status:")
                    print(f"  Identity: {status['hive_mind']['identity']}")
                    print(f"  Last sync: {time.ctime(status['hive_mind']['last_sync'])}")
                    print(f"  Sync count: {status['hive_mind']['sync_count']}")
                continue
            
            # Generate response
            response = client_model.generate(
                input_text=user_input,
                max_length=250,
                temperature=0.8,
                private_context=private_mode,
                use_hive_mind=not private_mode
            )
            
            print(f"\nSAM: {response}")
    
    except KeyboardInterrupt:
        pass
    finally:
        logger.info(f"Stopping client {client_id}...")
        client_model.stop_services()
        client_model.save(os.path.join(client_config.save_dir, "final"))
        logger.info(f"Client {client_id} stopped and saved.")

def main():
    """Main function to run the hive mind example"""
    print("SAM Hive Mind Example")
    print("=====================")
    print("1. Start server")
    print("2. Start client 1")
    print("3. Start client 2")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ")
    
    if choice == "1":
        run_server()
    elif choice == "2":
        run_client(1)
    elif choice == "3":
        run_client(2)
    elif choice == "4":
        print("Exiting...")
    else:
        print("Invalid choice. Exiting...")

if __name__ == "__main__":
    main()
