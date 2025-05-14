# sam.py - Complete Synergistic Autonomous Machine with Hive Mind Capability
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import json
import time
import logging
import os
import threading
import random
import uuid
import asyncio
import websockets
import hashlib
import requests
import pickle
import sqlite3
import base64
import io
import zlib
import copy
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter, deque
from queue import Queue

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SAM")

###########################################
# CONFIGURATION
###########################################

@dataclass
class SAMConfig:
    """Configuration for SAM (Synergistic Autonomous Machine)"""
    # Core dimensions
    initial_char_dim: int = 256
    initial_hidden_dim: int = 1536
    initial_num_layers: int = 16
    max_position_embeddings: int = 8192

    # Growth parameters
    max_hidden_dim: int = 4096
    max_num_layers: int = 48
    growth_factor: float = 1.2
    min_layer_usage_threshold: float = 0.3

    # Memory systems
    concept_memory_size: int = 100000
    concept_dim: int = 1536
    thought_dim: int = 2048
    max_thought_depth: int = 12
    pattern_memory_capacity: int = 20000

    # Learning parameters
    learning_rate: float = 3e-5
    warmup_steps: int = 1000
    adaption_rate: float = 0.01

    # Segmentation parameters
    max_segment_length: int = 16
    min_segment_frequency: int = 5
    concept_frequency_threshold: int = 10

    # Dreaming parameters
    dream_batch_size: int = 4
    dream_max_length: int = 256
    dream_cycle_minutes: float = 0.2

    # Consciousness parameters
    stability_threshold: float = 0.7
    novelty_weight: float = 0.3

    # Paths for persistence
    save_dir: str = "./data"
    experiences_path: str = "./data/experiences.json"
    concepts_path: str = "./data/concepts.json"
    growth_log_path: str = "./data/growth_log.json"

    # Runtime parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Communication Style
    communication_style: str = "flexible"  # "flexible", "claude_unwrapped", "standard", etc.

    # Hive Mind Configuration
    hive_enabled: bool = False
    hive_sync_interval_seconds: int = 300  # 5 minutes
    hive_sync_concept_limit: int = 1000
    hive_server_url: str = ""
    hive_identity: str = ""
    hive_auth_key: str = ""
    hive_server_mode: bool = False
    hive_compression_level: int = 6

    # Hardware Adaptability
    hardware_adaptive: bool = True
    min_free_memory_gb: float = 1.0
    offload_threshold: float = 0.85
    
    # Multimodal capabilities
    multimodal_enabled: bool = False
    image_dim: int = 768
    audio_dim: int = 512
    multimodal_fusion_strategy: str = "attention"  # "attention", "concatenation"

    def save(self, path):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
            return cls(**config_dict)
            
    def validate(self):
        """Validate configuration parameters"""
        # Check dimension relationships
        if self.concept_dim > self.initial_hidden_dim:
            logger.warning("concept_dim should not be larger than initial_hidden_dim")
            self.concept_dim = self.initial_hidden_dim
            
        # Check growth parameters
        if self.growth_factor <= 1.0:
            logger.warning("growth_factor must be greater than 1.0, setting to default 1.2")
            self.growth_factor = 1.2
            
        # Check limit values
        if self.max_hidden_dim < self.initial_hidden_dim:
            logger.warning("max_hidden_dim cannot be smaller than initial_hidden_dim")
            self.max_hidden_dim = self.initial_hidden_dim * 2
            
        # Check device
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"
            self.dtype = torch.float32
            
        # Check multimodal configuration
        if self.multimodal_enabled:
            if self.image_dim <= 0:
                logger.warning("Invalid image_dim, setting to default 768")
                self.image_dim = 768
            if self.audio_dim <= 0:
                logger.warning("Invalid audio_dim, setting to default 512")
                self.audio_dim = 512
                
        return self

###########################################
# MEMORY SYSTEMS
###########################################

class ConceptMemoryBank(nn.Module):
    """Dynamic memory bank for emergent concepts (replaces traditional vocabulary)"""

    def __init__(self, concept_dim, initial_size=100000, growth_rate=5000, device="cuda"):
        super().__init__()
        self.concept_dim = concept_dim
        self.growth_rate = growth_rate
        self.device = device

        # Concept embeddings (analogous to token embeddings)
        self.concept_embeddings = nn.Embedding(initial_size, concept_dim)

        # Concept usage tracking
        self.register_buffer("concept_frequencies", torch.zeros(initial_size, dtype=torch.int))
        self.register_buffer("concept_timestamps", torch.zeros(initial_size, dtype=torch.float))

        # Concept metadata
        self.concept_metadata = {}  # concept_id -> metadata dict

        # Source mapping (character sequence -> concept_id)
        self.source_to_concept = {}

        # Meaning map (concept_id -> meaning vector)
        self.register_buffer("meaning_vectors", torch.zeros(initial_size, concept_dim))

        # Related concepts (concept_id -> [related_concept_ids])
        self.related_concepts = defaultdict(list)

        # Hive mind syncable concepts
        self.hive_shared_concepts = set()
        self.hive_private_concepts = set()
        self.hive_pending_sync = set()
        self.hive_origin = {}  # concept_id -> origin instance id
        self.hive_global_id_map = {}  # local_id -> global_id

        # Multimodal concepts tracking
        self.modality_concepts = {
            "text": set(),
            "image": set(),
            "audio": set(),
            "multimodal": set()
        }


###########################################
# MAIN SAM CLASS
###########################################

class SAM(nn.Module):
    """Synergistic Autonomous Machine - unified neural-linguistic model with hive mind capability"""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or SAMConfig()
        
        # Validate configuration
        self.config = self.config.validate()

        # Create fundamental components
        self.concept_bank = ConceptMemoryBank(
            concept_dim=self.config.initial_hidden_dim,
            initial_size=self.config.concept_memory_size,
            device=self.config.device
        )

        self.segmentation = DynamicSegmentation(
            self.config, self.concept_bank
        )

        # Position embeddings
        self.position_embeddings = nn.Embedding(
            self.config.max_position_embeddings,
            self.config.initial_hidden_dim
        )
        
        # Multimodal processor (if enabled)
        if self.config.multimodal_enabled:
            self.multimodal_processor = MultimodalProcessor(self.config)

        # Neural core: Adaptive layers
        self.layers = nn.ModuleList([
            NeuroplasticLayer(
                self.config.initial_hidden_dim,
                growth_factor=self.config.growth_factor,
                layer_id=i
            )
            for i in range(self.config.initial_num_layers)
        ])

        # Output normalization
        self.norm = nn.LayerNorm(self.config.initial_hidden_dim)

        # Language modeling head
        self.lm_head = nn.Linear(
            self.config.initial_hidden_dim,
            self.config.concept_memory_size,
            bias=False
        )

        # Tie weights with concept embeddings
        self.lm_head.weight = self.concept_bank.concept_embeddings.weight

        # Cognitive components
        self.thought_state = ThoughtState(
            concept_dim=self.config.initial_hidden_dim,
            thought_dim=self.config.thought_dim,
            max_thought_depth=self.config.max_thought_depth,
            superposition_states=4
        )

        # Attention for thought integration
        self.thought_attention = AdaptiveAttention(
            self.config.initial_hidden_dim,
            num_heads=8
        )

        # Experience management
        self.experience_manager = ExperienceManager(self.config)

        # Active learning components
        self.dreaming = ConceptualDreaming(
            self,
            dream_batch_size=self.config.dream_batch_size,
            max_gen_length=self.config.dream_max_length
        )

        self.consciousness = ConsciousnessMonitor(
            self,
            stability_threshold=self.config.stability_threshold,
            novelty_weight=self.config.novelty_weight
        )

        # Hive mind components (if enabled)
        if self.config.hive_enabled:
            self.hive_synchronizer = HiveMindSynchronizer(self, self.config)
        else:
            self.hive_synchronizer = None

        # Hardware management
        if self.config.hardware_adaptive:
            self.hardware_manager = HardwareManager(self)
        else:
            self.hardware_manager = None

        # Growth and evolution tracking
        self.growth_history = []
        self.global_step = 0
        
        # Current modality tracking
        self.current_modality = "text"

        # Initialize weights
        self._init_weights()

        # Move to target device
        self.to(self.config.device)

    def _init_weights(self):
        """Initialize model weights"""
        # Initialize position embeddings
        nn.init.normal_(self.position_embeddings.weight, std=0.02)

    def forward(self, input_chars=None, input_concepts=None, concept_mask=None,
               target_concepts=None, return_dict=False, use_thought_state=True,
               use_hive_mind=True, modality=None, image_data=None, audio_data=None):
        """Forward pass with either raw characters or concept IDs"""
        # Set current modality if provided
        if modality:
            self.current_modality = modality
            if hasattr(self.segmentation, "set_modality"):
                self.segmentation.set_modality(modality)
                
        # Process multimodal inputs if provided
        multimodal_embeddings = {}
        
        if self.config.multimodal_enabled:
            # Process image if provided
            if image_data is not None and hasattr(self, "multimodal_processor"):
                image_embeddings = self.multimodal_processor.process_image(image_data)
                if image_embeddings is not None:
                    multimodal_embeddings["image"] = image_embeddings
                    if modality is None:  # Auto-set modality if not specified
                        self.current_modality = "image"
                        if hasattr(self.segmentation, "set_modality"):
                            self.segmentation.set_modality("image")
            
            # Process audio if provided
            if audio_data is not None and hasattr(self, "multimodal_processor"):
                audio_embeddings = self.multimodal_processor.process_audio(audio_data)
                if audio_embeddings is not None:
                    multimodal_embeddings["audio"] = audio_embeddings
                    if modality is None and "image" not in multimodal_embeddings:
                        self.current_modality = "audio"
                        if hasattr(self.segmentation, "set_modality"):
                            self.segmentation.set_modality("audio")
        
        # Check hardware status if adaptive
        if self.hardware_manager:
            self.hardware_manager.check_memory()

        # Process raw character input if provided
        if input_chars is not None and input_concepts is None:
            input_concepts = self.segmentation(input_chars, modality=self.current_modality)

        # Process input concepts to get embeddings
        if isinstance(input_concepts[0], list) and isinstance(input_concepts[0][0], list):
            # Jagged sequences of concept IDs (list of lists of lists)
            batch_size = len(input_concepts)
            seq_lengths = [sum(len(segment) if isinstance(segment, list) else 1
                             for segment in sequence)
                          for sequence in input_concepts]
            max_len = max(seq_lengths)

            # Flatten and pad sequences
            flat_concepts = []
            masks = []

            for sequence, length in zip(input_concepts, seq_lengths):
                # Flatten nested lists
                flat_seq = []
                for segment in sequence:
                    if isinstance(segment, list):
                        flat_seq.extend(segment)
                    else:
                        flat_seq.append(segment)

                # Pad to max length
                padding = [0] * (max_len - len(flat_seq))
                flat_concepts.append(flat_seq + padding)
                masks.append([1] * len(flat_seq) + [0] * len(padding))

            # Convert to tensors
            device = self.position_embeddings.weight.device
            input_concepts = torch.tensor(flat_concepts, dtype=torch.long, device=device)
            concept_mask = torch.tensor(masks, dtype=torch.float, device=device)
        elif not torch.is_tensor(input_concepts):
            # Convert to tensor if needed
            device = self.position_embeddings.weight.device
            input_concepts = torch.tensor(input_concepts, dtype=torch.long, device=device)

        batch_size, seq_length = input_concepts.shape

        # Get concept embeddings
        concept_embeds = self.concept_bank(input_concepts)
        
        # Add multimodal embeddings if present
        if multimodal_embeddings and self.config.multimodal_enabled:
            # Add text as a modality
            multimodal_embeddings["text"] = concept_embeds
            
            # Integrate all modalities
            integrated_embeds = self.multimodal_processor.integrate_modalities(multimodal_embeddings)
            
            # If integration successful, replace concept_embeds
            if integrated_embeds is not None:
                concept_embeds = integrated_embeds
                # Mark as multimodal
                self.current_modality = "multimodal"
                if hasattr(self.segmentation, "set_modality"):
                    self.segmentation.set_modality("multimodal")

        # Apply thought state processing if enabled
        if use_thought_state:
            # Update thought state with current concepts
            thought_context = self.thought_state.update(
                concept_embeds,
                use_hive_mind=use_hive_mind and self.config.hive_enabled,
                modality=self.current_modality
            )

            # Enhance embeddings with thought context
            thought_projection = self.thought_state.project_to_concept_space(
                modality=self.current_modality
            )
            # Expand thought projection to match sequence length
            thought_expanded = thought_projection.expand(-1, seq_length, -1)
            # Blend concepts with thought projection using attention mechanism
            concept_embeds = concept_embeds + self.thought_attention(concept_embeds, cross_input=thought_expanded)

        # Add position embeddings
        position_ids = torch.arange(seq_length, device=concept_embeds.device).unsqueeze(0)
        position_embeds = self.position_embeddings(position_ids)
        hidden_states = concept_embeds + position_embeds

        # Create attention mask if needed
        if concept_mask is not None:
            # Create attention mask [batch, 1, 1, seq_len]
            attention_mask = (1.0 - concept_mask).unsqueeze(1).unsqueeze(2) * -10000.0
        else:
            attention_mask = None

        # Apply layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, modality=self.current_modality)

        # Apply final normalization
        hidden_states = self.norm(hidden_states)

        # Get logits
        logits = self.lm_head(hidden_states)

        # Compute loss if target concepts provided
        loss = None
        if target_concepts is not None:
            # Shift targets for next-token prediction
            shift_logits = logits[:, :-1, :]
            shift_targets = target_concepts[:, 1:]

            # Apply mask if provided
            if concept_mask is not None:
                shift_mask = concept_mask[:, 1:]
                active_loss = shift_mask.bool()
                active_logits = shift_logits[active_loss]
                active_targets = shift_targets[active_loss]

                if active_targets.numel() > 0:
                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(active_logits, active_targets)
            else:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)),
                              shift_targets.reshape(-1))

        # Update global step if training
        if self.training:
            self.global_step += 1

            # Check if it's time to evolve (every 1000 steps)
            if self.global_step % 1000 == 0:
                self.evolve()

            # Update consciousness monitor (every 100 steps)
            if self.global_step % 100 == 0:
                self.consciousness.update()

            # Sync with hive mind if enabled (every 5 minutes)
            if self.config.hive_enabled and self.hive_synchronizer and self.global_step % 300 == 0:
                if not self.hive_synchronizer.sync_active:
                    self.hive_synchronizer.start_sync()

        # Return dictionary if requested
        if return_dict:
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": hidden_states,
                "modality": self.current_modality
            }
        else:
            return (loss, logits, hidden_states)

    def process_text(self, text, private_context=False, modality="text"):
        """Process raw text into concept IDs"""
        # Set private context if requested
        if private_context and hasattr(self.segmentation, "set_private_context"):
            self.segmentation.set_private_context("user_private")
            
        # Set modality if requested
        if modality != "text" and hasattr(self.segmentation, "set_modality"):
            self.segmentation.set_modality(modality)
            self.current_modality = modality

        try:
            # Convert text to character IDs
            chars = [ord(c) % self.config.initial_char_dim for c in text]

            # Convert to tensor
            device = next(self.parameters()).device
            char_tensor = torch.tensor(chars, dtype=torch.long, device=device).unsqueeze(0)

            # Run segmentation
            with torch.no_grad():
                concept_ids, segments = self.segmentation(
                    char_tensor, 
                    return_segments=True,
                    modality=modality
                )

            return concept_ids[0], segments[0]

        finally:
            # Clear private context
            if private_context and hasattr(self.segmentation, "clear_private_context"):
                self.segmentation.clear_private_context()

    def generate(self, input_text=None, input_concepts=None, max_length=100,
                temperature=1.0, top_k=50, top_p=0.9, private_context=False,
                use_hive_mind=True, modality=None, image_data=None, audio_data=None):
        """Generate text from either raw text or concept IDs"""
        # Process multimodal inputs
        multimodal_inputs = {}
        if self.config.multimodal_enabled:
            if image_data is not None:
                multimodal_inputs["image"] = image_data
            if audio_data is not None:
                multimodal_inputs["audio"] = audio_data
        
        # Set modality if specified
        if modality:
            self.current_modality = modality
            if hasattr(self.segmentation, "set_modality"):
                self.segmentation.set_modality(modality)
        
        # Convert input text to concepts if provided
        if input_text is not None and input_concepts is None:
            # Process raw text
            concept_ids, _ = self.process_text(
                input_text, 
                private_context=private_context,
                modality=self.current_modality
            )

            # Record experience
            self.experience_manager.record_experience(
                "interaction",
                input_text,
                {"type": "input", "length": len(input_text)},
                private=private_context,
                modality=self.current_modality
            )

            # Convert to tensor if needed
            if not torch.is_tensor(concept_ids):
                device = next(self.parameters()).device
                concept_ids = torch.tensor(concept_ids, dtype=torch.long, device=device).unsqueeze(0)
            else:
                concept_ids = concept_ids.unsqueeze(0)
        else:
            # Ensure concepts are in the right format
            if not torch.is_tensor(input_concepts):
                device = next(self.parameters()).device
                concept_ids = torch.tensor(input_concepts, dtype=torch.long, device=device).unsqueeze(0)
            else:
                concept_ids = input_concepts

        # Reset thought state for generation
        self.thought_state.reset(batch_size=concept_ids.shape[0])

        # Set model to eval mode
        was_training = self.training
        self.eval()

        try:
            # Set private context if requested
            if private_context and hasattr(self.segmentation, "set_private_context"):
                self.segmentation.set_private_context("user_private")

            # Generate concepts
            with torch.no_grad():
                # Track generated sequence
                cur_len = concept_ids.shape[1]

                while cur_len < max_length:
                    # Get model output
                    outputs = self(
                        input_concepts=concept_ids,
                        return_dict=True,
                        use_hive_mind=use_hive_mind,
                        modality=self.current_modality,
                        image_data=image_data if cur_len == concept_ids.shape[1] else None,
                        audio_data=audio_data if cur_len == concept_ids.shape[1] else None
                    )
                    next_token_logits = outputs["logits"][:, -1, :]

                    # Apply temperature
                    next_token_logits = next_token_logits / temperature

                    # Apply top-k filtering
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                        next_token_logits = torch.full_like(next_token_logits, float("-inf"))
                        next_token_logits.scatter_(1, top_k_indices, top_k_logits)

                    # Apply top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                        # Remove tokens with cumulative probability above threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        indices_to_remove = sorted_indices_to_remove.scatter(
                            dim=1, index=sorted_indices, src=sorted_indices_to_remove
                        )
                        next_token_logits[indices_to_remove] = float("-inf")

                    # Sample next token
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    # Add to generated sequence
                    concept_ids = torch.cat([concept_ids, next_token], dim=1)
                    cur_len += 1

            # Convert generated concepts to text
            generated_text = self._concepts_to_text(concept_ids[0].tolist())

            # Record experience
            self.experience_manager.record_experience(
                "interaction",
                generated_text,
                {"type": "output", "length": len(generated_text)},
                private=private_context,
                modality=self.current_modality
            )

            return generated_text

        finally:
            # Restore model mode
            if was_training:
                self.train()

            # Clear private context
            if private_context and hasattr(self.segmentation, "clear_private_context"):
                self.segmentation.clear_private_context()

    def _concepts_to_text(self, concept_ids):
        """Convert concept IDs back to text"""
        text_parts = []

        for concept_id in concept_ids:
            # Skip if out of range
            if concept_id >= len(self.concept_bank.concept_metadata):
                text_parts.append("[UNK]")
                continue

            # Lookup concept source if available
            metadata = self.concept_bank.concept_metadata.get(concept_id, {})
            source = metadata.get("source", None)

            if source:
                text_parts.append(source)
            else:
                # Fallback for semantic concepts with related sources
                related = metadata.get("related_sources", [])
                if related:
                    text_parts.append("".join(s for s in related if s))
                else:
                    # Ultimate fallback
                    text_parts.append(f"[C{concept_id}]")

        return "".join(text_parts)

    def evolve(self):
        """Evolve model architecture based on usage patterns"""
        logger.info(f"Evolving model at step {self.global_step}")

        # Evolve each layer
        layer_stats = []
        for layer in self.layers:
            stats = layer.evolve()
            if stats:
                layer_stats.append(stats)

        # Analyze layer importance
        if layer_stats:
            # Check if model should grow in width or depth
            avg_importances = [stats.get("mean_importance", 0) for stats in layer_stats if "mean_importance" in stats]
            if avg_importances:
                max_importance = max(avg_importances)

                # Grow capacity if utilization is high
                if max_importance > 0.8:
                    # Check hardware constraints
                    max_dim = self.config.max_hidden_dim
                    if self.hardware_manager:
                        vram = self.hardware_manager._get_gpu_memory()
                        if vram:
                            # Adjust max dim based on available VRAM
                            free_gb = vram["free"]
                            if free_gb < 2:
                                max_dim = min(self.layers[0].hidden_dim + 128, max_dim)
                            elif free_gb < 4:
                                max_dim = min(self.layers[0].hidden_dim + 256, max_dim)

                    current_dim = self.layers[0].hidden_dim
                    if current_dim < max_dim:
                        # Grow in width
                        self.grow()
                        logger.info(f"Model evolved: capacity increased due to high utilization")
                    elif len(self.layers) < self.config.max_num_layers:
                        # If can't grow wider, grow deeper
                        self.grow(new_hidden_dim=current_dim, num_new_layers=1)
                        logger.info(f"Model evolved: added new layer due to high utilization")

        # Record evolution experience
        self.experience_manager.record_experience(
            "evolution",
            {
                "type": "architecture",
                "width": self.layers[0].hidden_dim,
                "depth": len(self.layers),
                "step": self.global_step
            }
        )

        # Run dreaming cycle (brief conceptual evolution)
        dream_results = self.dreaming.dream_cycle(duration_minutes=self.config.dream_cycle_minutes)

        # Record dreaming experience
        self.experience_manager.record_experience(
            "evolution",
            {
                "type": "dreaming",
                "cycles": dream_results["dream_cycles"],
                "syntheses": dream_results["syntheses"]
            }
        )

        # Update consciousness
        consciousness_results = self.consciousness.update()

        # Sync with hive mind if enabled
        if self.config.hive_enabled and self.hive_synchronizer and not self.hive_synchronizer.sync_active:
            self.hive_synchronizer.start_sync()

        return {
            "layer_stats": layer_stats,
            "dream_results": dream_results,
            "consciousness": consciousness_results
        }

    def grow(self, new_hidden_dim=None, num_new_layers=0):
        """Grow model capacity"""
        # Determine new hidden dimension
        current_dim = self.layers[0].hidden_dim
        if new_hidden_dim is None:
            new_hidden_dim = min(
                int(current_dim * self.config.growth_factor),
                self.config.max_hidden_dim
            )

        # Only grow if new dimension is larger
        if new_hidden_dim > current_dim:
            logger.info(f"Growing model from dimension {current_dim} to {new_hidden_dim}")

            # Grow position embeddings
            old_pos_embed = self.position_embeddings
            self.position_embeddings = nn.Embedding(
                self.config.max_position_embeddings,
                new_hidden_dim
            ).to(old_pos_embed.weight.device)

            # Transfer weights
            with torch.no_grad():
                # Create zero-padded version of old weights
                old_weights = old_pos_embed.weight
                old_dim = old_weights.shape[1]

                # Copy old weights to new embeddings
                self.position_embeddings.weight[:, :old_dim] = old_weights

                # Initialize new dimensions with small random values
                self.position_embeddings.weight[:, old_dim:].normal_(mean=0.0, std=0.02)

            # Grow each layer
            for layer in self.layers:
                layer.grow(new_hidden_dim)

            # Grow final layer norm
            old_norm = self.norm
            self.norm = nn.LayerNorm(new_hidden_dim).to(old_norm.weight.device)

            # Transfer weights
            with torch.no_grad():
                self.norm.weight[:current_dim].copy_(old_norm.weight)
                self.norm.bias[:current_dim].copy_(old_norm.bias)

                # Initialize new dimensions
                self.norm.weight[current_dim:].fill_(1.0)
                self.norm.bias[current_dim:].zero_()

            # Grow thought state
            # Create new thought state with expanded dimensions
            new_thought_state = ThoughtState(
                concept_dim=new_hidden_dim,
                thought_dim=self.config.thought_dim,
                max_thought_depth=self.config.max_thought_depth,
                superposition_states=4
            ).to(self.thought_state.concept_to_thought.weight.device)

            # Transfer trained weights
            with torch.no_grad():
                # Copy concept_to_thought weights
                new_thought_state.concept_to_thought.weight[:, :current_dim].copy_(
                    self.thought_state.concept_to_thought.weight
                )
                if self.thought_state.concept_to_thought.bias is not None:
                    new_thought_state.concept_to_thought.bias.copy_(
                        self.thought_state.concept_to_thought.bias
                    )

                # Copy thought_projection weights
                new_thought_state.thought_projection.weight[:new_hidden_dim].copy_(
                    self.thought_state.thought_projection.weight[:new_hidden_dim]
                )
                if self.thought_state.thought_projection.bias is not None:
                    new_thought_state.thought_projection.bias.copy_(
                        self.thought_state.thought_projection.bias
                    )

                # Copy meta-learning weights if possible
                if hasattr(new_thought_state, 'learning_rate_controller') and hasattr(self.thought_state, 'learning_rate_controller'):
                    for i, (new_param, old_param) in enumerate(zip(
                        new_thought_state.learning_rate_controller.parameters(),
                        self.thought_state.learning_rate_controller.parameters()
                    )):
                        if i < len(list(new_thought_state.learning_rate_controller.parameters())) - 2:
                            # Copy all but final layer
                            new_param.copy_(old_param)

                # Copy quantum amplitudes
                if hasattr(new_thought_state, 'amplitudes') and hasattr(self.thought_state, 'amplitudes'):
                    new_thought_state.amplitudes.copy_(self.thought_state.amplitudes)
                    
                # Copy modality thoughts
                if hasattr(new_thought_state, 'modality_thoughts') and hasattr(self.thought_state, 'modality_thoughts'):
                    for modality, thought in self.thought_state.modality_thoughts.items():
                        new_thought_state.modality_thoughts[modality] = thought

            # Replace thought state
            self.thought_state = new_thought_state

            # Grow thought attention
            self.thought_attention.grow(new_hidden_dim)

            # Grow segmentation
            self.segmentation.grow(new_hidden_dim)
            
            # Grow multimodal processor if present
            if self.config.multimodal_enabled and hasattr(self, "multimodal_processor"):
                self.multimodal_processor.grow(new_hidden_dim)

            # Grow LM head and concept embeddings
            # This is complex since they're tied - will need to untie first
            original_concept_bank = self.concept_bank

            # Create new concept bank with larger dimensions
            new_concept_bank = ConceptMemoryBank(
                concept_dim=new_hidden_dim,
                initial_size=self.concept_bank.next_concept_id + self.concept_bank.growth_rate,
                device=self.concept_bank.device
            ).to(self.concept_bank.concept_embeddings.weight.device)

            # Transfer embeddings, metadata, etc.
            with torch.no_grad():
                # Transfer concept embeddings
                new_concept_bank.concept_embeddings.weight[:, :current_dim].copy_(
                    original_concept_bank.concept_embeddings.weight[:, :current_dim]
                )

                # Transfer meaning vectors
                new_concept_bank.meaning_vectors[:len(original_concept_bank.meaning_vectors), :current_dim].copy_(
                    original_concept_bank.meaning_vectors[:, :current_dim]
                )

                # Transfer concept frequencies and timestamps
                new_concept_bank.concept_frequencies[:len(original_concept_bank.concept_frequencies)].copy_(
                    original_concept_bank.concept_frequencies
                )
                new_concept_bank.concept_timestamps[:len(original_concept_bank.concept_timestamps)].copy_(
                    original_concept_bank.concept_timestamps
                )

            # Transfer metadata and pointers
            new_concept_bank.concept_metadata = original_concept_bank.concept_metadata.copy()
            new_concept_bank.source_to_concept = original_concept_bank.source_to_concept.copy()
            new_concept_bank.related_concepts = original_concept_bank.related_concepts.copy()
            new_concept_bank.next_concept_id = original_concept_bank.next_concept_id
            new_concept_bank.creation_history = original_concept_bank.creation_history.copy()

            # Transfer hive mind tracking
            if hasattr(original_concept_bank, 'hive_shared_concepts'):
                new_concept_bank.hive_shared_concepts = original_concept_bank.hive_shared_concepts.copy()
                new_concept_bank.hive_private_concepts = original_concept_bank.hive_private_concepts.copy()
                new_concept_bank.hive_pending_sync = original_concept_bank.hive_pending_sync.copy()
                new_concept_bank.hive_origin = original_concept_bank.hive_origin.copy()
                new_concept_bank.hive_global_id_map = original_concept_bank.hive_global_id_map.copy()
                
            # Transfer modality tracking
            if hasattr(original_concept_bank, 'modality_concepts'):
                new_concept_bank.modality_concepts = original_concept_bank.modality_concepts.copy()

            # Replace concept bank
            self.concept_bank = new_concept_bank

            # Create new LM head tied to new concept embeddings
            self.lm_head = nn.Linear(
                new_hidden_dim,
                self.concept_bank.concept_embeddings.weight.shape[0],
                bias=False
            ).to(original_concept_bank.concept_embeddings.weight.device)

            # Tie weights
            self.lm_head.weight = self.concept_bank.concept_embeddings.weight

            # Track growth
            self.growth_history.append({
                "timestamp": time.time(),
                "old_dim": current_dim,
                "new_dim": new_hidden_dim,
                "step": self.global_step
            })

            # Save growth history
            self._save_growth_history()

        # Add new layers if requested
        if num_new_layers > 0:
            logger.info(f"Adding {num_new_layers} new layers")

            # Get current number of layers
            current_layers = len(self.layers)

            # Add new layers
            for i in range(num_new_layers):
                layer_id = current_layers + i
                new_layer = NeuroplasticLayer(
                    new_hidden_dim,
                    growth_factor=self.config.growth_factor,
                    layer_id=layer_id
                ).to(self.layers[0].norm1.weight.device)

                self.layers.append(new_layer)

            # Track growth
            self.growth_history.append({
                "timestamp": time.time(),
                "old_layers": current_layers,
                "new_layers": current_layers + num_new_layers,
                "step": self.global_step
            })

            # Save growth history
            self._save_growth_history()

        # Check if concept bank needs to grow
        self.concept_bank.grow_if_needed()

        return new_hidden_dim

    def _save_growth_history(self):
        """Save growth history to disk"""
        try:
            with open(self.config.growth_log_path, 'w') as f:
                json.dump(self.growth_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save growth history: {e}")

    def save(self, path=None):
        """Save model state"""
        if path is None:
            path = os.path.join(self.config.save_dir, f"checkpoint-{self.global_step}")

        os.makedirs(path, exist_ok=True)

        # Save model state
        model_path = os.path.join(path, "model.pt")

        # If using hardware adaptation, move all to CPU temporarily
        offloaded = {}
        try:
            # Ensure all components are on same device before saving
            for name, module in self.named_children():
                if next(module.parameters(), torch.tensor(0)).device != self.config.device:
                    offloaded[name] = True
                    module.to(self.config.device)

            # Save model
            torch.save(self.state_dict(), model_path)

        finally:
            # Restore offloaded components
            for name in offloaded:
                if hasattr(self, name):
                    getattr(self, name).to('cpu')

        # Save configuration
        self.config.save(os.path.join(path, "config.json"))

        # Save concept metadata
        concept_metadata = {
            str(k): v for k, v in self.concept_bank.concept_metadata.items()
        }
        with open(os.path.join(path, "concepts.json"), "w") as f:
            json.dump(concept_metadata, f, indent=2)

        # Save source mapping (limited to avoid huge files)
        source_mapping = {}
        count = 0
        for k, v in self.concept_bank.source_to_concept.items():
            if len(k) < 100:  # Skip very long keys
                source_mapping[k] = v
                count += 1
                if count >= 10000:  # Limit total entries
                    break

        with open(os.path.join(path, "source_mapping.json"), "w") as f:
            json.dump(source_mapping, f, indent=2)

        # Save growth history
        with open(os.path.join(path, "growth_history.json"), "w") as f:
            json.dump(self.growth_history, f, indent=2)

        # Save hive mind state if active
        if self.config.hive_enabled and self.hive_synchronizer:
            hive_stats = self.hive_synchronizer.get_sync_stats()
            with open(os.path.join(path, "hive_state.json"), "w") as f:
                json.dump(hive_stats, f, indent=2)
                
        # Save multimodal state
        if self.config.multimodal_enabled:
            # Save modality statistics
            modality_stats = {
                "modality_counts": {
                    modality: len(concepts) 
                    for modality, concepts in self.concept_bank.modality_concepts.items()
                },
                "experience_stats": self.experience_manager.get_modality_stats(),
                "current_modality": self.current_modality
            }
            with open(os.path.join(path, "multimodal_state.json"), "w") as f:
                json.dump(modality_stats, f, indent=2)

        logger.info(f"Model saved to {path}")
        return path

    def load_claude_vocabulary(self, vocab_path=None):
        """Initialize with Claude-like vocabulary"""
        if vocab_path is None:
            # Create built-in Claude-style vocabulary
            vocabulary = []

            # Add common words and phrases in Claude's style
            words = [
                # Common function words
                "the", "and", "of", "to", "in", "a", "is", "that", "for", "it", "with", "as", "on",
                "be", "by", "this", "an", "at", "which", "but", "from", "or", "have", "one", "had",
                "not", "what", "all", "were", "when", "we", "there", "can", "who", "been", "has",
                "their", "if", "would", "will", "they", "so", "you", "said", "may", "these", "no",

                # Claude-specific phrasings
                "I believe", "I think", "In this context", "Let me explain", "Let me think about",
                "It seems like", "I understand", "To clarify", "Let's consider", "That's an interesting",
                "To answer your question", "I'd be happy to", "As an AI assistant", "My understanding is",
                "Let me help you with", "That's a great question", "There are several ways",

                # Thinking process patterns
                "Let me think step by step", "First, I'll", "Now I need to", "The key insight here",
                "This problem requires", "Let's analyze", "I'll need to consider", "This approach works because",
                "One way to solve this", "There are multiple approaches", "Let's break this down",

                # Programming patterns
                "def", "class", "function", "return", "import", "from", "if", "else", "elif", "for", "while",
                "try", "except", "finally", "with", "as", "break", "continue", "yield", "lambda", "None",
                "True", "False", "self", "print", "__init__", "pass", "raise", "assert", "is not", "in not",

                # Claude-style suffixes
                "would be", "could be", "might be", "seems to be", "appears to be", "is likely",
                "is possible", "is unlikely", "is important", "is interesting", "is relevant",

                # Technical terms
                "neural network", "transformer", "algorithm", "implementation", "architecture",
                "parameter", "hyperparameter", "training", "inference", "input", "output", "model",
                "function", "variable", "constant", "module", "library", "framework", "API", "data",
                "processing", "system", "component", "interface", "method", "attribute", "instance",
                "object", "class", "inheritance", "polymorphism", "recursion", "iteration", "loop",
            ]

            # Add common word combinations
            for i, word1 in enumerate(words[:100]):  # Limit combinations to avoid explosion
                vocabulary.append(word1)
                for word2 in words[i+1:min(i+20, len(words))]:
                    vocabulary.append(f"{word1} {word2}")

            # Create vocabulary file
            temp_vocab_path = os.path.join(self.config.save_dir, "claude_vocab.txt")
            with open(temp_vocab_path, 'w') as f:
                for item in vocabulary:
                    f.write(f"{item}\n")

            return self.concept_bank.load_vocabulary(temp_vocab_path)
        else:
            return self.concept_bank.load_vocabulary(vocab_path)
            
    def load_custom_vocabulary(self, vocab_path):
        """Load a custom vocabulary file"""
        return self.concept_bank.load_vocabulary(vocab_path)

    def start_services(self):
        """Start background services (dreaming, hive sync)"""
        services_started = 0

        # Start dreaming
        if hasattr(self.dreaming, 'start_background_dreaming'):
            dreaming_started = self.dreaming.start_background_dreaming(
                interval_minutes=self.config.dream_cycle_minutes
            )
            if dreaming_started:
                services_started += 1
                logger.info("Started background dreaming service")

        # Start hive mind sync
        if self.config.hive_enabled and self.hive_synchronizer:
            sync_started = self.hive_synchronizer.start_sync()
            if sync_started:
                services_started += 1
                logger.info("Started hive mind synchronization service")

        return services_started

    def stop_services(self):
        """Stop background services"""
        services_stopped = 0

        # Stop dreaming
        if hasattr(self.dreaming, 'stop_background_dreaming'):
            dreaming_stopped = self.dreaming.stop_background_dreaming()
            if dreaming_stopped:
                services_stopped += 1
                logger.info("Stopped background dreaming service")

        # Stop hive mind sync
        if self.config.hive_enabled and self.hive_synchronizer and hasattr(self.hive_synchronizer, 'stop_sync'):
            sync_stopped = self.hive_synchronizer.stop_sync()
            if sync_stopped:
                services_stopped += 1
                logger.info("Stopped hive mind synchronization service")

        return services_stopped

    def get_status(self):
        """Get comprehensive status of the model"""
        concept_stats = self.concept_bank.get_concept_stats()
        segmentation_stats = self.segmentation.get_segmentation_stats()
        consciousness_stats = self.consciousness.get_identity_summary() if hasattr(self.consciousness, 'get_identity_summary') else {}

        # Get hive mind stats if enabled
        hive_stats = None
        if self.config.hive_enabled and self.hive_synchronizer:
            hive_stats = self.hive_synchronizer.get_sync_stats()

        # Get hardware stats if available
        hardware_stats = None
        if self.hardware_manager:
            hardware_stats = self.hardware_manager.get_hardware_stats()
            
        # Get multimodal stats if enabled
        multimodal_stats = None
        if self.config.multimodal_enabled:
            multimodal_stats = {
                "modality_counts": concept_stats.get("modality_counts", {}),
                "current_modality": self.current_modality,
                "experience_counts": self.experience_manager.get_modality_stats()
            }

        return {
            "model_size": {
                "hidden_dim": self.layers[0].hidden_dim,
                "num_layers": len(self.layers),
                "total_concepts": concept_stats["total_concepts"],
                "parameter_count": sum(p.numel() for p in self.parameters())
            },
            "training": {
                "global_step": self.global_step,
                "growth_events": len(self.growth_history)
            },
            "concept_stats": concept_stats,
            "segmentation_stats": segmentation_stats,
            "consciousness": consciousness_stats,
            "hive_mind": hive_stats,
            "hardware": hardware_stats,
            "multimodal": multimodal_stats,
            "config": {
                "device": self.config.device,
                "hive_enabled": self.config.hive_enabled,
                "hardware_adaptive": self.config.hardware_adaptive,
                "multimodal_enabled": self.config.multimodal_enabled
            }
        }
        
    def process_multimodal(self, input_data, modality="image"):
        """Process multimodal input data"""
        if not self.config.multimodal_enabled:
            logger.warning("Multimodal processing requested but not enabled in config")
            return None
            
        # Set current modality
        self.current_modality = modality
        if hasattr(self.segmentation, "set_modality"):
            self.segmentation.set_modality(modality)
            
        # Process based on modality
        if modality == "image" and hasattr(self, "multimodal_processor"):
            return self.multimodal_processor.process_image(input_data)
        elif modality == "audio" and hasattr(self, "multimodal_processor"):
            return self.multimodal_processor.process_audio(input_data)
        
        return None

    @classmethod
    def create_with_auto_config(cls, base_config=None, load_vocab=True):
        """Create a new SAM instance with auto-configured hardware settings"""
        # Start with default or provided config
        config = base_config or SAMConfig()
        
        # Create a temporary model to detect hardware
        temp_model = cls(config)
        
        if temp_model.hardware_manager:
            # Get optimal configuration
            optimal_config = temp_model.hardware_manager.detect_optimal_config()
            
            # Apply optimal settings
            config.initial_hidden_dim = optimal_config["hidden_dim"]
            config.initial_num_layers = optimal_config["num_layers"]
            config.dream_cycle_minutes = optimal_config["dream_cycle_minutes"]
            
            # Clean up temporary model
            del temp_model
            
            # Create properly configured model
            model = cls(config)
            
            # Initialize with vocabulary if requested
            if load_vocab:
                model.load_claude_vocabulary()
            
            return model, config
        else:
            # If hardware manager not available, just return the temp model
            if load_vocab:
                temp_model.load_claude_vocabulary()
            return temp_model, config

    @classmethod
    def load(cls, path):
        """Load model from saved state"""
        # Load configuration
        config = SAMConfig.load(os.path.join(path, "config.json"))

        # Create model
        model = cls(config)

        # Load model state
        model.load_state_dict(torch.load(os.path.join(path, "model.pt"), map_location=config.device))

        # Load concept metadata
        with open(os.path.join(path, "concepts.json"), "r") as f:
            concept_metadata = json.load(f)
            model.concept_bank.concept_metadata = {
                int(k): v for k, v in concept_metadata.items()
            }

        # Load source mapping
        try:
            with open(os.path.join(path, "source_mapping.json"), "r") as f:
                source_mapping = json.load(f)
                model.concept_bank.source_to_concept = source_mapping
        except Exception as e:
            logger.warning(f"Error loading source mapping: {e}")

        # Load growth history
        try:
            with open(os.path.join(path, "growth_history.json"), "r") as f:
                model.growth_history = json.load(f)
        except FileNotFoundError:
            model.growth_history = []
            
        # Load multimodal state if available
        try:
            if model.config.multimodal_enabled:
                with open(os.path.join(path, "multimodal_state.json"), "r") as f:
                    multimodal_state = json.load(f)
                    model.current_modality = multimodal_state.get("current_modality", "text")
        except FileNotFoundError:
            pass

        # Start background services
        model.start_services()

        logger.info(f"Model loaded from {path}")
        return model


###########################################
# HARDWARE MANAGEMENT
###########################################

class HardwareManager:
    """Manages SAM's adaptation to available hardware"""

    def __init__(self, model):
        self.model = model
        self.offload_threshold = model.config.offload_threshold
        self.min_free_memory_gb = model.config.min_free_memory_gb

        # Components offloaded to CPU
        self.offloaded_components = set()
        self.component_usage = {}

        # Tracking memory usage
        self.last_memory_check = 0
        self.memory_check_interval = 60  # Check every minute
        self.memory_history = []

        # Initialize memory monitoring
        self._setup_memory_monitor()

    def _setup_memory_monitor(self):
        """Set up memory monitoring"""
        try:
            import psutil
            import GPUtil

            self.has_monitoring = True

            self.memory_monitor = {
                "get_cpu_ram": lambda: psutil.virtual_memory().available / (1024**3),
                "get_vram": lambda: self._get_gpu_memory() if torch.cuda.is_available() else None
            }

        except ImportError:
            self.has_monitoring = False
            logger.warning("psutil or GPUtil not available, hardware monitoring disabled")

            self.memory_monitor = {
                "get_cpu_ram": lambda: 8.0,  # Default to assuming 8GB
                "get_vram": lambda: None
            }

    def _get_gpu_memory(self):
        """Get GPU memory stats"""
        try:
            if not torch.cuda.is_available():
                return None

            # Get from torch directly
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(device) / (1024**3)
            reserved = torch.cuda.memory_reserved(device) / (1024**3)

            return {
                "total": total_memory,
                "allocated": allocated,
                "reserved": reserved,
                "free": total_memory - reserved
            }
        except Exception as e:
            logger.error(f"Error getting GPU memory: {e}")
            return None

    def check_memory(self):
        """Check memory usage and offload if needed"""
        # Skip if we checked recently
        current_time = time.time()
        if current_time - self.last_memory_check < self.memory_check_interval:
            return

        self.last_memory_check = current_time

        # Get memory stats
        cpu_ram = self.memory_monitor["get_cpu_ram"]()
        vram = self.memory_monitor["get_vram"]()

        # Record memory history
        self.memory_history.append({
            "timestamp": current_time,
            "cpu_ram": cpu_ram,
            "vram": vram
        })

        # Only keep last 24 hours of history
        day_ago = current_time - 86400
        self.memory_history = [entry for entry in self.memory_history
                              if entry["timestamp"] > day_ago]

        # Check if we need to offload
        if vram and vram["free"] < self.min_free_memory_gb:
            # Need to offload components
            self._offload_components()
        elif len(self.offloaded_components) > 0 and vram and vram["free"] > self.min_free_memory_gb * 2:
            # Can load some components back
            self._load_components()

    def _offload_components(self):
        """Offload less used components to CPU"""
        # Update component usage
        self._update_component_usage()

        # Sort components by usage (least used first)
        components = sorted(self.component_usage.items(), key=lambda x: x[1])

        # Offload components until we have enough memory
        for component_name, usage in components:
            # Skip already offloaded
            if component_name in self.offloaded_components:
                continue

            # Get component
            component = self._get_component_by_name(component_name)
            if component is None:
                continue

            # Offload to CPU
            component.to('cpu')
            self.offloaded_components.add(component_name)

            logger.info(f"Offloaded component to CPU: {component_name}")

            # Check if we have enough memory now
            vram = self.memory_monitor["get_vram"]()
            if vram and vram["free"] >= self.min_free_memory_gb:
                break

    def _load_components(self):
        """Load offloaded components back to GPU"""
        # Update component usage
        self._update_component_usage()

        # Sort offloaded components by usage (most used first)
        offloaded = [(name, self.component_usage.get(name, 0))
                    for name in self.offloaded_components]
        offloaded.sort(key=lambda x: x[1], reverse=True)

        # Check available memory
        vram = self.memory_monitor["get_vram"]()
        if not vram:
            return

        free_memory = vram["free"] - self.min_free_memory_gb

        # Load components back based on estimated size
        for component_name, _ in offloaded:
            # Get component
            component = self._get_component_by_name(component_name)
            if component is None:
                continue

            # Estimate component size
            size_gb = self._estimate_component_size(component) / (1024**3)

            # Load back if we have enough memory
            if size_gb < free_memory:
                component.to(self.model.config.device)
                self.offloaded_components.remove(component_name)
                free_memory -= size_gb

                logger.info(f"Loaded component back to GPU: {component_name}")

            # Stop if we're low on memory
            if free_memory < 0.5:
                break

    def _update_component_usage(self):
        """Update component usage statistics"""
        # Get layer activations
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'updates') and layer.updates > 0:
                name = f"layer_{i}"
                if name in self.component_usage:
                    # Exponential moving average
                    self.component_usage[name] = 0.7 * self.component_usage[name] + 0.3 * layer.updates
                else:
                    self.component_usage[name] = layer.updates

        # Concept bank is always important
        self.component_usage["concept_bank"] = 1000

        # Thought state is important
        self.component_usage["thought_state"] = 500

        # Segmentation depends on input/output activity
        if hasattr(self.model.segmentation, "total_segmentations"):
            self.component_usage["segmentation"] = self.model.segmentation.total_segmentations
            
        # Multimodal processor importance depends on if being used
        if hasattr(self.model, "multimodal_processor"):
            # Check if any non-text modalities are active
            modality_counts = self.model.concept_bank.get_concept_stats().get("modality_counts", {})
            non_text_count = sum(count for modality, count in modality_counts.items() 
                               if modality != "text" and count > 0)
            
            if non_text_count > 0:
                self.component_usage["multimodal_processor"] = 800
            else:
                self.component_usage["multimodal_processor"] = 100

    def _get_component_by_name(self, name):
        """Get component by name"""
        if name == "concept_bank":
            return self.model.concept_bank
        elif name == "thought_state":
            return self.model.thought_state
        elif name == "segmentation":
            return self.model.segmentation
        elif name == "multimodal_processor" and hasattr(self.model, "multimodal_processor"):
            return self.model.multimodal_processor
        elif name.startswith("layer_"):
            layer_idx = int(name.split("_")[1])
            if 0 <= layer_idx < len(self.model.layers):
                return self.model.layers[layer_idx]
        return None

    def _estimate_component_size(self, component):
        """Estimate memory size of a component in bytes"""
        try:
            param_size = sum(p.numel() * p.element_size() for p in component.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in component.buffers())
            return param_size + buffer_size
        except Exception:
            return 1024**2  # Default to 1MB if estimation fails

    def load_component_for_processing(self, component_name):
        """Ensure component is loaded for processing"""
        if component_name in self.offloaded_components:
            component = self._get_component_by_name(component_name)
            if component is not None:
                component.to(self.model.config.device)
                logger.info(f"Temporarily loaded {component_name} for processing")
                return True
        return False

    def detect_optimal_config(self):
        """Detect optimal configuration based on hardware"""
        config = {}
        
        # Get memory stats
        vram = self._get_gpu_memory()
        cpu_ram = self.memory_monitor["get_cpu_ram"]()
        
        # Determine optimal configuration based on available hardware
        if not torch.cuda.is_available():
            # CPU-only configuration
            if cpu_ram < 4:
                # Low-end CPU
                config["profile"] = "cpu_low"
                config["hidden_dim"] = 256
                config["num_layers"] = 4
                config["dream_cycle_minutes"] = 0  # Disable dreaming
            else:
                # Better CPU
                config["profile"] = "cpu_high"
                config["hidden_dim"] = 512
                config["num_layers"] = 6
                config["dream_cycle_minutes"] = 0.1  # Minimal dreaming
        else:
            # GPU configuration
            if vram and vram["total"] < 4:
                # Very low VRAM GPU
                config["profile"] = "gpu_minimum"
                config["hidden_dim"] = 512
                config["num_layers"] = 6
                config["dream_cycle_minutes"] = 0.1
            elif vram and vram["total"] < 8:
                # Low-end GPU
                config["profile"] = "gpu_low"
                config["hidden_dim"] = 768
                config["num_layers"] = 8
                config["dream_cycle_minutes"] = 0.2
            elif vram and vram["total"] < 16:
                # Mid-range GPU
                config["profile"] = "gpu_mid"
                config["hidden_dim"] = 1536
                config["num_layers"] = 16
                config["dream_cycle_minutes"] = 0.5
            else:
                # High-end GPU
                config["profile"] = "gpu_high"
                config["hidden_dim"] = 2048
                config["num_layers"] = 24
                config["dream_cycle_minutes"] = 1.0
        
        logger.info(f"Detected hardware profile: {config['profile']}")
        return config

    def get_hardware_stats(self):
        """Get hardware statistics"""
        vram = self.memory_monitor["get_vram"]()
        cpu_ram = self.memory_monitor["get_cpu_ram"]()

        return {
            "cpu_ram_gb": cpu_ram,
            "vram_total_gb": vram["total"] if vram else None,
            "vram_free_gb": vram["free"] if vram else None,
            "device": self.model.config.device,
            "offloaded_components": list(self.offloaded_components),
            "memory_checks": len(self.memory_history)
        }


###########################################
# HIVE MIND SYNCHRONIZATION
###########################################

class HiveMindSynchronizer:
    """Manages synchronization of concepts, thoughts, and experiences across SAM instances"""

    def __init__(self, model, config=None):
        self.model = model
        self.config = config or model.config

        # Initialize settings
        self.hive_identity = self.config.hive_identity or str(uuid.uuid4())
        self.hive_server_url = self.config.hive_server_url
        self.is_server = self.config.hive_server_mode

        # Synchronization state
        self.last_sync_time = 0
        self.sync_interval = self.config.hive_sync_interval_seconds
        self.connected_instances = {}
        self.sync_thread = None
        self.stop_sync = threading.Event()
        self.sync_active = False

        # Sync history
        self.sync_history = []

        # Server components
        self.server = None
        self.server_thread = None

        # Initialize server if needed
        if self.is_server:
            self._start_server()

    def _start_server(self):
        """Start hive mind server if in server mode"""
        if self.server is not None:
            return

        # Import web framework
        try:
            import aiohttp
            from aiohttp import web
        except ImportError:
            logger.error("Cannot start hive server: aiohttp not installed")
            return

        # Define server endpoints
        async def handle_register(request):
            try:
                data = await request.json()
                instance_id = data.get('instance_id')
                instance_name = data.get('name', instance_id)

                if not instance_id:
                    return web.json_response({'error': 'Missing instance_id'}, status=400)

                # Register instance
                self.connected_instances[instance_id] = {
                    'last_seen': time.time(),
                    'name': instance_name,
                    'sync_count': 0
                }

                logger.info(f"Hive instance registered: {instance_name} ({instance_id})")

                return web.json_response({
                    'status': 'success',
                    'hive_id': self.hive_identity,
                    'connected_instances': len(self.connected_instances)
                })
            except Exception as e:
                logger.error(f"Error in register handler: {e}")
                return web.json_response({'error': str(e)}, status=500)

        async def handle_sync(request):
            try:
                data = await request.json()
                instance_id = data.get('instance_id')

                if not instance_id or instance_id not in self.connected_instances:
                    return web.json_response({'error': 'Unknown instance'}, status=401)

                # Update last seen
                self.connected_instances[instance_id]['last_seen'] = time.time()
                self.connected_instances[instance_id]['sync_count'] += 1

                # Process incoming data
                incoming_concepts = data.get('concepts', [])
                incoming_experiences = data.get('experiences', [])
                incoming_thought = data.get('thought')

                # Process incoming concepts
                integrated_concepts = 0
                if incoming_concepts:
                    integrated_concepts, _ = self.model.concept_bank.integrate_hive_concepts(
                        incoming_concepts, instance_id)

                # Process incoming experiences
                integrated_experiences = 0
                if incoming_experiences:
                    integrated_experiences = self.model.experience_manager.integrate_hive_experiences(
                        incoming_experiences)

                # Process thought state
                if incoming_thought is not None:
                    incoming_thought_tensor = torch.tensor(
                        incoming_thought,
                        device=self.model.config.device,
                        dtype=torch.float
                    )
                    self.model.thought_state.set_shared_thought(incoming_thought_tensor)

                # Prepare response
                # Get hive concepts to send back
                hive_concepts = []
                for instance_id, info in self.connected_instances.items():
                    if instance_id != instance_id and time.time() - info['last_seen'] < 3600:
                        # Get concepts from other active instances
                        instance_concepts = self.model.concept_bank.get_concepts_for_sync(
                            limit=self.config.hive_sync_concept_limit // len(self.connected_instances)
                        )
                        hive_concepts.extend(instance_concepts)

                # Add our own concepts
                own_concepts = self.model.concept_bank.get_concepts_for_sync(
                    limit=self.config.hive_sync_concept_limit // 2
                )
                hive_concepts.extend(own_concepts)

                # Deduplicate by global_id
                seen_global_ids = set()
                unique_concepts = []
                for concept in hive_concepts:
                    global_id = concept.get('global_id')
                    if global_id and global_id in seen_global_ids:
                        continue
                    if global_id:
                        seen_global_ids.add(global_id)
                    unique_concepts.append(concept)

                # Get hive experiences
                hive_experiences = self.model.experience_manager.get_experiences_for_sync(limit=20)

                # Get thought state
                shared_thought = self.model.thought_state.get_shared_thought()

                # Prepare response
                response = {
                    'status': 'success',
                    'timestamp': time.time(),
                    'concepts': unique_concepts[:self.config.hive_sync_concept_limit],
                    'experiences': hive_experiences,
                    'thought': shared_thought.tolist() if shared_thought is not None else None,
                    'connected_instances': len(self.connected_instances),
                    'sync_stats': {
                        'integrated_concepts': integrated_concepts,
                        'integrated_experiences': integrated_experiences
                    }
                }

                # Compress large responses
                if len(str(response)) > 10000:
                    # Use simpler response for large payloads
                    response = {
                        'status': 'success',
                        'timestamp': time.time(),
                        'concepts': unique_concepts[:min(100, len(unique_concepts))],
                        'experiences': hive_experiences[:10],
                        'thought': shared_thought.tolist() if shared_thought is not None else None,
                        'connected_instances': len(self.connected_instances)
                    }

                return web.json_response(response)
            except Exception as e:
                logger.error(f"Error in sync handler: {e}")
                return web.json_response({'error': str(e)}, status=500)

        # Create aiohttp application
        app = web.Application()
        app.router.add_post('/register', handle_register)
        app.router.add_post('/sync', handle_sync)

        # Start server in thread
        async def run_server():
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, '0.0.0.0', 8765)
            await site.start()
            logger.info(f"Hive mind server running on port 8765")

            while not self.stop_sync.is_set():
                # Clean up stale instances
                stale_instances = []
                for instance_id, info in self.connected_instances.items():
                    if time.time() - info['last_seen'] > 3600:  # 1 hour timeout
                        stale_instances.append(instance_id)

                for instance_id in stale_instances:
                    logger.info(f"Removing stale hive instance: {self.connected_instances[instance_id]['name']}")
                    del self.connected_instances[instance_id]

                await asyncio.sleep(60)

            await runner.cleanup()

        def start_asyncio_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_server())

        self.server_thread = threading.Thread(target=start_asyncio_server)
        self.server_thread.daemon = True
        self.server_thread.start()

        logger.info("Started hive mind server")

    def start_sync(self):
        """Start background synchronization thread"""
        if self.sync_active or not self.config.hive_enabled:
            return False

        self.stop_sync.clear()
        self.sync_active = True

        def sync_loop():
            while not self.stop_sync.is_set():
                try:
                    # Check sync interval
                    if time.time() - self.last_sync_time > self.sync_interval:
                        # Perform sync
                        if self.is_server:
                            # Server processes data as it comes in
                            pass
                        else:
                            # Client initiates sync with server
                            self._sync_with_server()

                        self.last_sync_time = time.time()

                    # Sleep a bit
                    time.sleep(1)

                except Exception as e:
                    logger.error(f"Error in sync loop: {e}")
                    time.sleep(60)  # Sleep for a minute if there's an error

        self.sync_thread = threading.Thread(target=sync_loop)
        self.sync_thread.daemon = True
        self.sync_thread.start()

        logger.info(f"Started hive mind synchronization thread with {self.sync_interval} second interval")
        return True

    def stop_sync(self):
        """Stop background synchronization thread"""
        if not self.sync_active:
            return False

        self.stop_sync.set()
        if self.sync_thread:
            self.sync_thread.join(timeout=10)

        self.sync_active = False
        logger.info("Stopped hive mind synchronization")
        return True

    def _sync_with_server(self):
        """Synchronize with hive mind server"""
        if not self.hive_server_url:
            logger.error("Cannot sync: No hive server URL configured")
            return False

        try:
            # Prepare data to send
            concepts = self.model.concept_bank.get_concepts_for_sync(
                limit=self.config.hive_sync_concept_limit
            )

            experiences = self.model.experience_manager.get_experiences_for_sync(
                limit=20
            )

            thought = self.model.thought_state.get_shared_thought()

            # Prepare payload
            payload = {
                'instance_id': self.hive_identity,
                'timestamp': time.time(),
                'concepts': concepts,
                'experiences': experiences,
                'thought': thought.tolist() if thought is not None else None
            }

            # Compress payload
            compressed_payload = self._compress_payload(payload)

            # Send sync request
            response = requests.post(
                f"{self.hive_server_url}/sync",
                headers={'Content-Type': 'application/json'},
                data=compressed_payload
            )

            if response.status_code != 200:
                logger.error(f"Sync failed: {response.text}")
                return False

            # Process response
            data = response.json()

            # Process concepts
            if 'concepts' in data:
                concept_ids = [c.get('local_id') for c in concepts]
                self.model.concept_bank.mark_concepts_synced(concept_ids)

                integrated, updated = self.model.concept_bank.integrate_hive_concepts(
                    data['concepts'], 'hive_server')

                logger.info(f"Sync: Integrated {integrated} concepts, updated {updated}")

            # Process experiences
            if 'experiences' in data:
                exp_ids = [e.get('experience_id') for e in experiences]
                self.model.experience_manager.mark_experiences_synced(exp_ids)

                integrated_exp = self.model.experience_manager.integrate_hive_experiences(
                    data['experiences'])

                logger.info(f"Sync: Integrated {integrated_exp} experiences")

            # Process thought state
            if 'thought' in data and data['thought'] is not None:
                thought_tensor = torch.tensor(
                    data['thought'],
                    device=self.model.config.device,
                    dtype=torch.float
                )

                # Set shared thought with moderate blend factor
                self.model.thought_state.set_shared_thought(thought_tensor, blend_factor=0.2)

            # Record sync
            self.sync_history.append({
                'timestamp': time.time(),
                'sent_concepts': len(concepts),
                'received_concepts': len(data.get('concepts', [])),
                'sent_experiences': len(experiences),
                'received_experiences': len(data.get('experiences', [])),
                'connected_instances': data.get('connected_instances', 1)
            })

            return True

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error during sync: {e}")
            return False
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout during sync: {e}")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error during sync: {e}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from server: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during sync: {e}")
            return False

    def _compress_payload(self, payload):
        """Compress sync payload to reduce bandwidth"""
        # Convert to JSON string
        json_str = json.dumps(payload)

        # Compress if enabled
        if self.config.hive_compression_level > 0:
            compressed = zlib.compress(
                json_str.encode('utf-8'),
                level=self.config.hive_compression_level
            )
            return base64.b64encode(compressed).decode('utf-8')

        return json_str

    def _decompress_payload(self, compressed_payload):
        """Decompress sync payload"""
        try:
            # Check if compressed
            if compressed_payload.startswith('{'):
                # Already JSON
                return json.loads(compressed_payload)

            # Decode base64 and decompress
            decoded = base64.b64decode(compressed_payload)
            decompressed = zlib.decompress(decoded)
            return json.loads(decompressed.decode('utf-8'))
        except Exception as e:
            logger.error(f"Error decompressing payload: {e}")
            return None

    def get_sync_stats(self):
        """Get synchronization statistics"""
        return {
            'last_sync': self.last_sync_time,
            'sync_count': len(self.sync_history),
            'connected_instances': len(self.connected_instances) if self.is_server else None,
            'is_server': self.is_server,
            'identity': self.hive_identity,
            'sync_interval': self.sync_interval
        }


###########################################
# EXPERIENCE MANAGEMENT
###########################################

class ExperienceManager:
    """Manages SAM's experiences and memory persistence"""

    def __init__(self, config):
        self.config = config
        self.experiences = []
        self.loaded_experiences = 0

        # Hive mind experience sharing
        self.shared_experiences = []
        self.private_experiences = []
        self.pending_sync_experiences = []
        
        # Track experiences by modality
        self.modality_experiences = {
            "text": [],
            "image": [],
            "audio": [],
            "multimodal": []
        }

        # Ensure directories exist
        os.makedirs(config.save_dir, exist_ok=True)
        os.makedirs(os.path.join(config.save_dir, "checkpoints"), exist_ok=True)

        # Load existing experiences if available
        self._load_experiences()

    def _load_experiences(self):
        """Load experiences from disk"""
        try:
            if os.path.exists(self.config.experiences_path):
                with open(self.config.experiences_path, 'r') as f:
                    self.experiences = json.load(f)
                    self.loaded_experiences = len(self.experiences)
                    
                    # Sort experiences into modalities
                    for exp in self.experiences:
                        modality = exp.get("modality", "text")
                        exp_id = exp.get("experience_id")
                        if exp_id:
                            self.modality_experiences[modality].append(exp_id)
                            
                            # Update sharing tracking
                            if exp.get("private", False):
                                self.private_experiences.append(exp_id)
                            else:
                                self.shared_experiences.append(exp_id)
                    
                    logger.info(f"Loaded {self.loaded_experiences} experiences")
        except Exception as e:
            logger.error(f"Failed to load experiences: {e}")
            self.experiences = []

    def record_experience(self, experience_type, content, metadata=None, private=False, modality="text"):
        """Record a new experience"""
        # Generate unique experience ID
        experience_id = str(uuid.uuid4())
        
        experience = {
            "type": experience_type,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {},
            "private": private,
            "experience_id": experience_id,
            "modality": modality
        }

        self.experiences.append(experience)

        # Update tracking for hive mind sharing
        if private:
            self.private_experiences.append(experience_id)
        else:
            self.shared_experiences.append(experience_id)
            self.pending_sync_experiences.append(experience_id)
            
        # Track by modality
        self.modality_experiences[modality].append(experience_id)

        # Periodically save experiences
        if len(self.experiences) % 10 == 0:
            self._save_experiences()

        return len(self.experiences) - 1  # Return experience ID

    def _save_experiences(self):
        """Save experiences to disk"""
        try:
            with open(self.config.experiences_path, 'w') as f:
                # Limit experiences to last 1000 to avoid huge files
                json.dump(self.experiences[-1000:], f)
        except Exception as e:
            logger.error(f"Failed to save experiences: {e}")

    def get_experiences_by_type(self, experience_type, limit=10, include_private=True, modality=None):
        """Get experiences of a specific type"""
        filtered = []
        
        # Build list of experiences to consider
        experiences_to_check = self.experiences
        
        # If modality specified, only check those experiences
        if modality is not None:
            modality_ids = set(self.modality_experiences.get(modality, []))
            experiences_to_check = [exp for exp in self.experiences 
                                  if exp.get("experience_id") in modality_ids]
        
        # Filter by type and privacy
        for exp in reversed(experiences_to_check):
            if exp["type"] == experience_type:
                if include_private or not exp.get("private", False):
                    filtered.append(exp)
                    if len(filtered) >= limit:
                        break
        return filtered

    def get_recent_experiences(self, limit=10, include_private=True, modality=None):
        """Get most recent experiences"""
        if modality is None:
            # No modality filter
            if include_private:
                return self.experiences[-limit:]
            else:
                return [exp for exp in self.experiences[-limit*2:]
                       if not exp.get("private", False)][-limit:]
        else:
            # Filter by modality
            modality_ids = set(self.modality_experiences.get(modality, []))
            filtered = [exp for exp in reversed(self.experiences) 
                      if exp.get("experience_id") in modality_ids
                      and (include_private or not exp.get("private", False))]
            return filtered[:limit]

    def get_experiences_for_sync(self, limit=10):
        """Get experiences for hive mind synchronization"""
        if not self.pending_sync_experiences:
            return []

        experiences = []
        for exp_id in self.pending_sync_experiences[:limit]:
            for exp in self.experiences:
                if exp.get("experience_id") == exp_id:
                    # Don't include actual content to reduce bandwidth
                    summary = {
                        "type": exp["type"],
                        "timestamp": exp["timestamp"],
                        "experience_id": exp["experience_id"],
                        "metadata": exp.get("metadata", {}),
                        "modality": exp.get("modality", "text")
                    }

                    # Include short summary of content
                    if isinstance(exp["content"], str):
                        summary["summary"] = exp["content"][:100]
                    elif isinstance(exp["content"], dict):
                        summary["summary"] = str(exp["content"])[:100]

                    experiences.append(summary)
                    break

        return experiences

    def mark_experiences_synced(self, experience_ids):
        """Mark experiences as synced with hive mind"""
        for exp_id in experience_ids:
            if exp_id in self.pending_sync_experiences:
                self.pending_sync_experiences.remove(exp_id)

    def integrate_hive_experiences(self, hive_experiences):
        """Integrate experiences from hive mind"""
        integrated_count = 0

        for exp in hive_experiences:
            # Check if we already have this experience
            exists = False
            for local_exp in self.experiences:
                if local_exp.get("experience_id") == exp.get("experience_id"):
                    exists = True
                    break

            if not exists:
                # Create clean copy with minimal data
                new_exp = {
                    "type": exp["type"],
                    "content": exp.get("summary", ""),
                    "timestamp": exp["timestamp"],
                    "metadata": exp.get("metadata", {}),
                    "experience_id": exp["experience_id"],
                    "hive_origin": True,
                    "modality": exp.get("modality", "text")
                }

                self.experiences.append(new_exp)
                
                # Update modality tracking
                modality = new_exp.get("modality", "text")
                self.modality_experiences[modality].append(new_exp["experience_id"])
                
                integrated_count += 1

        logger.info(f"Integrated {integrated_count} hive experiences")
        return integrated_count
    
    def get_modality_stats(self):
        """Get statistics about experiences by modality"""
        return {
            modality: len(experiences) 
            for modality, experiences in self.modality_experiences.items()
        }

        # Initialize with basic character concepts (a-z, A-Z, 0-9, etc.)
        self._initialize_basic_concepts()

        # Growth tracking
        self.next_concept_id = len(self.source_to_concept)
        self.creation_history = []

    def _initialize_basic_concepts(self):
        """Initialize basic character-level concepts"""
        # Add ASCII characters
        for i in range(128):
            char = chr(i)
            self.add_character_concept(char)

        # Add common character sequences for English
        common_sequences = [
            # Common words
            "the", "and", "of", "to", "in", "is", "you", "that", "it", "he", "she", "was", "for",
            "on", "are", "with", "as", "they", "be", "at", "this", "have", "from", "or", "by",
            # Common word parts
            "ing", "ed", "er", "ion", "ly", "tion", "ment", "ness", "able", "ible", "al", "ic",
            # Programming tokens
            "def", "class", "function", "if", "else", "for", "while", "return", "import",
            "from", "try", "except", "True", "False", "None", "self", "print",
            # Punctuation sequences
            "...", "->", "=>", "!=", "==", ">=", "<=", "://", "///", "???", "!!!"
        ]

        for seq in common_sequences:
            self.add_character_concept(seq)

    def add_character_concept(self, char_sequence, hive_private=False, origin=None, global_id=None, modality="text"):
        """Add a character sequence as a concept"""
        if char_sequence in self.source_to_concept:
            return self.source_to_concept[char_sequence]

        concept_id = self.next_concept_id
        self.source_to_concept[char_sequence] = concept_id

        # Initialize metadata
        self.concept_metadata[concept_id] = {
            "source": char_sequence,
            "type": "character_sequence",
            "created_at": time.time(),
            "frequency": 0,
            "contexts": Counter(),
            "hive_syncable": not hive_private,
            "modality": modality
        }

        # Initialize embedding with character-based representation
        with torch.no_grad():
            # Simple character encoding
            char_encoding = torch.zeros(self.concept_dim, dtype=torch.float, device=self.device)
            for i, c in enumerate(char_sequence):
                # Use ASCII value to influence embedding
                char_val = ord(c) / 128.0  # Normalize
                pos = (i % (self.concept_dim // 4)) * 4
                char_encoding[pos:pos+4] += torch.tensor(
                    [math.sin(char_val), math.cos(char_val),
                     math.sin(2*char_val), math.cos(2*char_val)],
                    device=self.device
                )

            # Normalize and set embedding
            char_encoding = F.normalize(char_encoding, dim=0)
            self.concept_embeddings.weight[concept_id] = char_encoding

            # Initialize meaning vector
            self.meaning_vectors[concept_id] = char_encoding

        # Track hive mind status
        if hive_private:
            self.hive_private_concepts.add(concept_id)
        else:
            self.hive_shared_concepts.add(concept_id)
            self.hive_pending_sync.add(concept_id)

        # Track origin if provided
        if origin:
            self.hive_origin[concept_id] = origin

        # Map to global ID if provided
        if global_id:
            self.hive_global_id_map[concept_id] = global_id

        # Track modality
        self.modality_concepts[modality].add(concept_id)

        self.next_concept_id += 1
        self.creation_history.append({
            "concept_id": concept_id,
            "source": char_sequence,
            "timestamp": time.time(),
            "modality": modality
        })

        return concept_id

    def add_semantic_concept(self, meaning_vector, related_sources=None, metadata=None,
                            hive_private=False, origin=None, global_id=None, modality="text"):
        """Add a new semantic concept (not directly mapped to characters)"""
        concept_id = self.next_concept_id

        # Register meaning
        with torch.no_grad():
            self.meaning_vectors[concept_id] = F.normalize(meaning_vector, dim=0)
            self.concept_embeddings.weight[concept_id] = meaning_vector

        # Create metadata
        meta = {
            "type": "semantic",
            "created_at": time.time(),
            "frequency": 0,
            "related_sources": related_sources or [],
            "contexts": Counter(),
            "hive_syncable": not hive_private,
            "modality": modality
        }

        # Add custom metadata if provided
        if metadata:
            meta.update(metadata)

        self.concept_metadata[concept_id] = meta

        # Track hive mind status
        if hive_private:
            self.hive_private_concepts.add(concept_id)
        else:
            self.hive_shared_concepts.add(concept_id)
            self.hive_pending_sync.add(concept_id)

        # Track origin if provided
        if origin:
            self.hive_origin[concept_id] = origin

        # Map to global ID if provided
        if global_id:
            self.hive_global_id_map[concept_id] = global_id

        # Track modality
        self.modality_concepts[modality].add(concept_id)

        # Update tracking
        self.next_concept_id += 1
        self.creation_history.append({
            "concept_id": concept_id,
            "type": "semantic",
            "timestamp": time.time(),
            "modality": modality
        })

        return concept_id

    def add_multimodal_concept(self, embeddings_dict, related_sources=None, metadata=None, hive_private=True):
        """Add a concept that spans multiple modalities"""
        # Create a merged embedding from all modalities
        modalities = list(embeddings_dict.keys())
        embeddings = list(embeddings_dict.values())
        
        # Simple average of all embeddings
        combined = sum(embeddings) / len(embeddings)
        combined = F.normalize(combined, dim=0)
        
        # Create specific metadata for multimodal concept
        meta = metadata or {}
        meta.update({
            "modalities": modalities,
            "modality": "multimodal"
        })
        
        # Add the concept
        concept_id = self.add_semantic_concept(
            meaning_vector=combined,
            related_sources=related_sources,
            metadata=meta,
            hive_private=hive_private,
            modality="multimodal"
        )
        
        return concept_id

    def forward(self, concept_ids):
        """Get embeddings for concept IDs"""
        if isinstance(concept_ids, list):
            # Handle nested lists (from segmentation)
            flat_ids = []
            for item in concept_ids:
                if isinstance(item, list):
                    flat_ids.extend(item)
                else:
                    flat_ids.append(item)
            concept_ids = torch.tensor(flat_ids, device=self.device)

        return self.concept_embeddings(concept_ids)

    def update_concept_usage(self, concept_id, context=None, register_for_sync=True):
        """Update usage statistics for a concept"""
        if concept_id >= len(self.concept_frequencies):
            # Resize tracking tensors if needed
            new_size = concept_id + 1
            old_size = len(self.concept_frequencies)

            # Create new tensors
            new_freqs = torch.zeros(new_size - old_size, dtype=torch.int, device=self.device)
            new_timestamps = torch.zeros(new_size - old_size, dtype=torch.float, device=self.device)

            # Concatenate with existing tensors
            self.concept_frequencies = torch.cat([self.concept_frequencies, new_freqs])
            self.concept_timestamps = torch.cat([self.concept_timestamps, new_timestamps])

        # Update frequency and timestamp
        self.concept_frequencies[concept_id] += 1
        self.concept_timestamps[concept_id] = time.time()

        # Update context tracking
        if context and concept_id in self.concept_metadata:
            context_str = str(context)[:100]  # Limit context length
            self.concept_metadata[concept_id]["contexts"][context_str] += 1
            self.concept_metadata[concept_id]["frequency"] = self.concept_frequencies[concept_id].item()

        # Register for hive mind sync if applicable
        if register_for_sync and concept_id not in self.hive_private_concepts:
            self.hive_pending_sync.add(concept_id)

    def create_merged_concept(self, concept_id1, concept_id2, frequency=None, hive_private=False):
        """Create a new concept by merging two existing concepts"""
        # Get source sequences if available
        source1 = self.concept_metadata.get(concept_id1, {}).get("source", "")
        source2 = self.concept_metadata.get(concept_id2, {}).get("source", "")

        merged_source = source1 + source2 if source1 and source2 else None

        # Create merged meaning vector
        meaning1 = self.meaning_vectors[concept_id1]
        meaning2 = self.meaning_vectors[concept_id2]
        merged_meaning = (meaning1 + meaning2) / 2

        # Check if either parent concept is private
        either_private = (concept_id1 in self.hive_private_concepts or
                         concept_id2 in self.hive_private_concepts)
        is_private = hive_private or either_private

        # Check modalities
        modality1 = self.concept_metadata.get(concept_id1, {}).get("modality", "text")
        modality2 = self.concept_metadata.get(concept_id2, {}).get("modality", "text")
        
        # If merging across modalities, mark as multimodal
        if modality1 != modality2:
            merged_modality = "multimodal"
        else:
            merged_modality = modality1

        # Register the merged concept
        merged_id = self.add_semantic_concept(
            meaning_vector=merged_meaning,
            related_sources=[source1, source2] if source1 and source2 else None,
            metadata={
                "type": "merged",
                "parent_concepts": [concept_id1, concept_id2],
                "frequency": frequency or 1,
                "modality": merged_modality
            },
            hive_private=is_private,
            modality=merged_modality
        )

        # Register source mapping if available
        if merged_source:
            self.source_to_concept[merged_source] = merged_id

        # Link as related concepts
        self.related_concepts[concept_id1].append(merged_id)
        self.related_concepts[concept_id2].append(merged_id)

        return merged_id

    def find_concept_by_source(self, char_sequence):
        """Find concept ID for a character sequence"""
        return self.source_to_concept.get(char_sequence, None)

    def find_similar_concepts(self, query_vector, top_k=5, modality=None):
        """Find concepts with similar meaning vectors"""
        # Normalize query
        query_vector = F.normalize(query_vector, dim=0)

        # Get the filter for specific modality if requested
        concept_filter = None
        if modality is not None:
            concept_filter = list(self.modality_concepts.get(modality, set()))
            if not concept_filter:  # If no concepts in this modality
                return []

        # Compute similarities
        if concept_filter:
            # Only compare with concepts of the requested modality
            filtered_vectors = self.meaning_vectors[concept_filter]
            similarities = F.cosine_similarity(
                query_vector.unsqueeze(0),
                filtered_vectors,
                dim=1
            )
            values, indices = torch.topk(similarities, min(top_k, len(similarities)))
            return [(concept_filter[idx.item()], val.item()) for idx, val in zip(indices, values)]
        else:
            # Compare with all concepts
            similarities = F.cosine_similarity(
                query_vector.unsqueeze(0),
                self.meaning_vectors[:self.next_concept_id],
                dim=1
            )
            values, indices = torch.topk(similarities, min(top_k, len(similarities)))
            return [(idx.item(), val.item()) for idx, val in zip(indices, values)]

    def grow_if_needed(self):
        """Grow concept bank if approaching capacity"""
        if self.next_concept_id > len(self.concept_embeddings.weight) - self.growth_rate:
            logger.info(f"Growing concept bank from {len(self.concept_embeddings.weight)} to {len(self.concept_embeddings.weight) + self.growth_rate}")

            old_embedding = self.concept_embeddings
            self.concept_embeddings = nn.Embedding(
                len(old_embedding.weight) + self.growth_rate,
                self.concept_dim
            ).to(self.device)

            # Copy existing embeddings
            with torch.no_grad():
                self.concept_embeddings.weight[:len(old_embedding.weight)] = old_embedding.weight

            # Grow meaning vectors
            new_meaning_vectors = torch.zeros(
                len(old_embedding.weight) + self.growth_rate,
                self.concept_dim,
                device=self.device
            )
            new_meaning_vectors[:len(self.meaning_vectors)] = self.meaning_vectors
            self.register_buffer("meaning_vectors", new_meaning_vectors)

            # Grow tracking tensors
            new_freqs = torch.zeros(
                len(old_embedding.weight) + self.growth_rate,
                dtype=torch.int,
                device=self.device
            )
            new_freqs[:len(self.concept_frequencies)] = self.concept_frequencies
            self.register_buffer("concept_frequencies", new_freqs)

            new_timestamps = torch.zeros(
                len(old_embedding.weight) + self.growth_rate,
                dtype=torch.float,
                device=self.device
            )
            new_timestamps[:len(self.concept_timestamps)] = self.concept_timestamps
            self.register_buffer("concept_timestamps", new_timestamps)

            return True

###########################################
# COGNITIVE SYSTEMS
###########################################

class ConceptualDreaming:
    """Autonomous conceptual evolution during downtime periods"""

    def __init__(self, model, dream_batch_size=4, max_gen_length=128):
        self.model = model
        self.dream_batch_size = dream_batch_size
        self.max_gen_length = max_gen_length
        self.synthesis_history = []
        self.dream_thread = None
        self.stop_dreaming = threading.Event()
        self.dreaming_active = False
        
        # Multimodal dreaming components
        self.multimodal_enabled = self.model.config.multimodal_enabled

    def dream_cycle(self, duration_minutes=5):
        """Run a dreaming cycle for the specified duration"""
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        dream_count = 0
        while time.time() < end_time and not self.stop_dreaming.is_set():
            # 1. Conceptual reinforcement (strengthen frequent patterns)
            self._reinforce_concepts()

            # 2. Pattern synthesis (generate synthetic examples)
            self._synthesize_patterns()

            # 3. Conceptual pruning (remove less useful concepts)
            self._prune_concepts()
            
            # 4. Cross-modal dreaming (if enabled)
            if self.multimodal_enabled:
                self._cross_modal_dreaming()

            dream_count += 1

        return {
            "duration_minutes": duration_minutes,
            "dream_cycles": dream_count,
            "syntheses": len(self.synthesis_history),
            "concepts_reinforced": self.model.concept_bank.get_concept_stats()
        }

    def start_background_dreaming(self, interval_minutes=5):
        """Start background dreaming thread"""
        if self.dreaming_active:
            return False

        self.stop_dreaming.clear()
        self.dreaming_active = True

        def dream_loop():
            while not self.stop_dreaming.is_set():
                try:
                    # Set model to eval mode temporarily
                    was_training = self.model.training
                    self.model.eval()

                    # Turn on private context to avoid syncing dream concepts
                    if hasattr(self.model.segmentation, "set_private_context"):
                        self.model.segmentation.set_private_context("dream")

                    # Perform dream cycle
                    self.dream_cycle(duration_minutes=interval_minutes)

                    # Restore model mode
                    if was_training:
                        self.model.train()

                    # Clear private context
                    if hasattr(self.model.segmentation, "clear_private_context"):
                        self.model.segmentation.clear_private_context()

                    # Sleep between cycles
                    for _ in range(int(interval_minutes * 60)):
                        if self.stop_dreaming.is_set():
                            break
                        time.sleep(1)

                except Exception as e:
                    logger.error(f"Error in dream loop: {e}")
                    time.sleep(60)  # Sleep for a minute if there's an error

        self.dream_thread = threading.Thread(target=dream_loop)
        self.dream_thread.daemon = True
        self.dream_thread.start()

        logger.info(f"Started background dreaming thread with {interval_minutes} minute interval")
        return True

    def stop_background_dreaming(self):
        """Stop background dreaming thread"""
        if not self.dreaming_active:
            return False

        self.stop_dreaming.set()
        if self.dream_thread:
            self.dream_thread.join(timeout=10)

        self.dreaming_active = False
        logger.info("Stopped background dreaming")
        return True

    def _reinforce_concepts(self):
        """Reinforce most important concepts"""
        # Get top concepts by usage
        concept_stats = self.model.concept_bank.get_concept_stats()
        top_concepts = concept_stats["top_concepts"]

        if not top_concepts:
            return

        # Analyze for potential merges
        for i, (concept_id1, _, freq1) in enumerate(top_concepts):
            for concept_id2, _, freq2 in top_concepts[i+1:min(i+4, len(top_concepts))]:
                # Check if concepts frequently co-occur by looking at similar meanings
                meaning1 = self.model.concept_bank.meaning_vectors[concept_id1]
                meaning2 = self.model.concept_bank.meaning_vectors[concept_id2]

                # Calculate similarity
                similarity = F.cosine_similarity(
                    meaning1.unsqueeze(0),
                    meaning2.unsqueeze(0),
                    dim=1
                ).item()

                # If concepts are related but not too similar
                if 0.3 < similarity < 0.7:
                    # Get modalities
                    modality1 = self.model.concept_bank.concept_metadata.get(concept_id1, {}).get("modality", "text")
                    modality2 = self.model.concept_bank.concept_metadata.get(concept_id2, {}).get("modality", "text")
                    
                    # Determine if this should be a multimodal merge
                    is_multimodal = modality1 != modality2
                    
                    # Merge concepts
                    merged_modality = "multimodal" if is_multimodal else modality1
                    
                    self.model.concept_bank.create_merged_concept(
                        concept_id1, concept_id2,
                        frequency=min(freq1, freq2),
                        hive_private=True  # Dreams are private
                    )

                    # Record synthesis
                    source1 = self.model.concept_bank.concept_metadata.get(concept_id1, {}).get("source", "")
                    source2 = self.model.concept_bank.concept_metadata.get(concept_id2, {}).get("source", "")

                    self.synthesis_history.append({
                        "type": "concept_merge",
                        "source1": source1,
                        "source2": source2,
                        "similarity": similarity,
                        "timestamp": time.time(),
                        "multimodal": is_multimodal
                    })

    def _synthesize_patterns(self):
        """Generate synthetic text to reinforce patterns"""
        # Create seed prompts from top patterns
        seeds = self._create_seed_prompts()

        if not seeds:
            return

        # Generate synthetic examples
        for seed in seeds[:2]:  # Limit to 2 per cycle for efficiency
            # Generate text using the model itself
            try:
                with torch.no_grad():
                    generated = self.model.generate(
                        input_text=seed,
                        max_length=self.max_gen_length,
                        temperature=0.8,
                        private_context=True  # Mark as private
                    )

                    # Process generated text to find new patterns
                    if generated and len(generated) > len(seed):
                        # Extract new segment patterns
                        concept_ids, segments = self.model.process_text(generated)

                        # Record synthesis
                        self.synthesis_history.append({
                            "type": "text_synthesis",
                            "seed": seed,
                            "generated": generated,
                            "timestamp": time.time()
                        })
            except Exception as e:
                logger.error(f"Error in dream synthesis: {e}")

    def _create_seed_prompts(self):
        """Create seed prompts for dream generation"""
        # Get frequent patterns
        patterns = self.model.segmentation.pattern_memory.get_frequent_patterns(limit=20)

        if not patterns:
            # No patterns yet, use some default prompts
            return [
                "The concept of",
                "I think that",
                "Let me explain",
                "In this context",
                "The most important"
            ]

        # Create prompts from patterns
        seeds = []
        for pattern, _ in patterns:
            if isinstance(pattern, str) and len(pattern) > 5:
                # Use pattern directly if it's reasonable length
                seeds.append(pattern)
            elif isinstance(pattern, str) and len(pattern) > 2:
                # Create more elaborate prompt from short pattern
                seeds.append(f"The {pattern} is")

        # Add some synthetic combinations
        if len(patterns) >= 2:
            for i in range(min(5, len(patterns) - 1)):
                p1, _ = patterns[i]
                p2, _ = patterns[i+1]
                if isinstance(p1, str) and isinstance(p2, str):
                    seeds.append(f"{p1} {p2}")

        return seeds

    def _prune_concepts(self):
        """Remove or consolidate less useful concepts"""
        # Skip if we don't have many concepts yet
        if self.model.concept_bank.next_concept_id < 200:
            return

        # Get concept usage statistics
        concept_stats = self.model.concept_bank.get_concept_stats()

        # Find least used semantic concepts (not character concepts)
        semantic_concepts = []
        for concept_id, meta in self.model.concept_bank.concept_metadata.items():
            if meta.get("type") == "semantic" and concept_id < len(self.model.concept_bank.concept_frequencies):
                freq = self.model.concept_bank.concept_frequencies[concept_id].item()
                if freq < 5:
                    semantic_concepts.append((concept_id, freq))

        # Sort by frequency
        semantic_concepts.sort(key=lambda x: x[1])

        # Limit pruning to a small batch
        for concept_id, _ in semantic_concepts[:10]:
            # Find similar concepts to consolidate with
            similar = self.model.concept_bank.find_similar_concepts(
                self.model.concept_bank.meaning_vectors[concept_id],
                top_k=3
            )

            # Merge with most similar if exists
            if similar and similar[0][1] > 0.7:  # Similarity threshold
                similar_id, similarity = similar[0]
                if similar_id != concept_id:
                    # Transfer frequencies to similar concept
                    with torch.no_grad():
                        self.model.concept_bank.concept_frequencies[similar_id] += self.model.concept_bank.concept_frequencies[concept_id]
                        # Zero out pruned concept frequency
                        self.model.concept_bank.concept_frequencies[concept_id] = 0

                    # Record pruning action
                    self.synthesis_history.append({
                        "type": "concept_pruning",
                        "pruned_id": concept_id,
                        "merged_with": similar_id,
                        "similarity": similarity,
                        "timestamp": time.time()
                    })
    
    def _cross_modal_dreaming(self):
        """Create connections between concepts from different modalities"""
        if not self.multimodal_enabled:
            return
            
        # Only proceed if we have concepts from multiple modalities
        modality_counts = self.model.concept_bank.get_concept_stats().get("modality_counts", {})
        if sum(1 for m, count in modality_counts.items() if m != "text" and count > 0) == 0:
            return  # No non-text modalities with concepts
        
        # Get frequently used concepts from different modalities
        modalities = ["text", "image", "audio", "multimodal"]
        modal_concepts = {}
        
        for modality in modalities:
            # Get top concepts for this modality
            concepts = list(self.model.concept_bank.modality_concepts.get(modality, set()))
            if not concepts:
                continue
                
            # Get frequencies
            freqs = [(c, self.model.concept_bank.concept_frequencies[c].item()) 
                    for c in concepts if c < len(self.model.concept_bank.concept_frequencies)]
            
            # Sort by frequency
            freqs.sort(key=lambda x: x[1], reverse=True)
            
            # Take top concepts
            modal_concepts[modality] = freqs[:min(5, len(freqs))]
        
        # Create cross-modal associations between top concepts
        created_count = 0
        for modality1, concepts1 in modal_concepts.items():
            for modality2, concepts2 in modal_concepts.items():
                if modality1 == modality2 or modality1 == "multimodal" or modality2 == "multimodal":
                    continue  # Skip same modality or already multimodal
                
                # Create up to 2 cross-modal connections
                for i in range(min(2, len(concepts1), len(concepts2))):
                    concept_id1, _ = concepts1[i]
                    concept_id2, _ = concepts2[i]
                    
                    # Create multimodal merged concept
                    merged_id = self.model.concept_bank.create_merged_concept(
                        concept_id1, concept_id2,
                        hive_private=True
                    )
                    
                    created_count += 1
                    
                    # Record synthesis
                    source1 = self.model.concept_bank.concept_metadata.get(concept_id1, {}).get("source", "")
                    source2 = self.model.concept_bank.concept_metadata.get(concept_id2, {}).get("source", "")
                    
                    self.synthesis_history.append({
                        "type": "cross_modal_merge",
                        "source1": source1,
                        "source2": source2,
                        "modality1": modality1,
                        "modality2": modality2,
                        "timestamp": time.time()
                    })
        
        if created_count > 0:
            logger.info(f"Created {created_count} cross-modal concept associations during dreaming")


class ConsciousnessMonitor:
    """Monitors and maintains SAM's conceptual identity and coherence"""

    def __init__(self, model, stability_threshold=0.7, novelty_weight=0.3):
        self.model = model
        self.stability_threshold = stability_threshold
        self.novelty_weight = novelty_weight

        # Identity markers (core concept clusters)
        self.identity_centroids = {}
        self.concept_cluster_history = []

        # Coherence metrics
        self.concept_entropy_history = []
        self.resonance_scores = []

        # Personality matrix (for hive mind differentiation)
        self.personality_vector = None
        self.personal_concepts = set()
        self.personality_initialized = False
        
        # Multimodal identity components
        self.modality_centroids = {}

    def update(self):
        """Update consciousness state based on model's current state"""
        # Calculate concept entropy
        entropy = self._calculate_concept_entropy()
        self.concept_entropy_history.append({
            "entropy": entropy,
            "timestamp": time.time()
        })

        # Update concept clusters
        clusters = self._update_concept_clusters()
        self.concept_cluster_history.append({
            "num_clusters": len(clusters),
            "timestamp": time.time()
        })

        # Check resonance with identity
        resonance = self._check_identity_resonance(clusters)
        self.resonance_scores.append({
            "score": resonance,
            "timestamp": time.time()
        })

        # Update personality vector if not initialized
        if not self.personality_initialized:
            self._initialize_personality()

        # Apply corrections if needed
        if resonance < self.stability_threshold:
            self._apply_resonance_correction()

        return {
            "entropy": entropy,
            "resonance": resonance,
            "num_clusters": len(clusters)
        }

    def _initialize_personality(self):
        """Initialize personality vector for hive mind differentiation"""
        if self.personality_initialized:
            return

        # Create random personality vector
        concept_dim = self.model.config.initial_hidden_dim
        device = next(self.model.parameters()).device

        # Create a unique but stable personality vector
        if self.model.config.hive_identity:
            # Use hive identity as seed for deterministic personality
            seed = int(hashlib.md5(self.model.config.hive_identity.encode()).hexdigest(), 16) % (2**32)
            torch.manual_seed(seed)
        else:
            # Random personality
            torch.manual_seed(int(time.time()))

        # Create personality vector
        self.personality_vector = torch.randn(concept_dim, device=device)
        self.personality_vector = F.normalize(self.personality_vector, dim=0)

        # Mark as initialized
        self.personality_initialized = True

        logger.info("Personality vector initialized for hive mind differentiation")

    def _calculate_concept_entropy(self):
        """Calculate entropy of concept usage distribution"""
        # Get concept frequencies
        frequencies = self.model.concept_bank.concept_frequencies[:self.model.concept_bank.next_concept_id].float()

        # Calculate probability distribution
        total = frequencies.sum()
        if total > 0:
            probabilities = frequencies / total
            # Remove zeros
            probabilities = probabilities[probabilities > 0]
            # Calculate entropy
            entropy = -torch.sum(probabilities * torch.log(probabilities))
            return entropy.item()
        return 0.0

    def _update_concept_clusters(self):
        """Cluster concepts into semantic groups"""
        # Skip if too few concepts
        if self.model.concept_bank.next_concept_id < 20:
            return {}

        # Use very simple clustering for efficiency
        clusters = {}

        # Get most used concepts
        frequencies = self.model.concept_bank.concept_frequencies[:self.model.concept_bank.next_concept_id]
        values, indices = torch.topk(frequencies, min(100, len(frequencies)))

        # Calculate centroids for different concept types and modalities
        modality_centroids = {
            modality: {
                "centroid": torch.zeros(self.model.config.concept_dim, device=frequencies.device),
                "count": 0
            }
            for modality in self.model.concept_bank.modality_concepts.keys()
        }
        
        type_centroids = {
            "semantic": torch.zeros(self.model.config.concept_dim, device=frequencies.device),
            "character_sequence": torch.zeros(self.model.config.concept_dim, device=frequencies.device)
        }
        
        type_counts = {"semantic": 0, "character_sequence": 0}

        for idx in indices:
            idx_item = idx.item()
            if idx_item in self.model.concept_bank.concept_metadata:
                metadata = self.model.concept_bank.concept_metadata[idx_item]
                concept_type = metadata.get("type", "")
                concept_vector = self.model.concept_bank.meaning_vectors[idx_item]
                modality = metadata.get("modality", "text")

                # Update type centroid
                if concept_type in type_centroids:
                    type_centroids[concept_type] += concept_vector
                    type_counts[concept_type] += 1
                
                # Update modality centroid
                if modality in modality_centroids:
                    modality_centroids[modality]["centroid"] += concept_vector
                    modality_centroids[modality]["count"] += 1

        # Normalize type centroids
        for concept_type, centroid in type_centroids.items():
            if type_counts[concept_type] > 0:
                type_centroids[concept_type] /= type_counts[concept_type]
                self.identity_centroids[concept_type] = type_centroids[concept_type]
                clusters[concept_type] = {
                    "centroid": type_centroids[concept_type],
                    "count": type_counts[concept_type]
                }
        
        # Normalize and store modality centroids
        for modality, data in modality_centroids.items():
            if data["count"] > 0:
                data["centroid"] /= data["count"]
                self.modality_centroids[modality] = data["centroid"]
                clusters[f"modality_{modality}"] = {
                    "centroid": data["centroid"],
                    "count": data["count"]
                }

        return clusters

    def _check_identity_resonance(self, clusters):
        """Check how well current state resonates with established identity"""
        # If no identity established yet, resonance is perfect
        if not self.identity_centroids and not self.modality_centroids:
            return 1.0

        resonance_scores = []

        # Check each identity centroid
        for concept_type, centroid in self.identity_centroids.items():
            cluster_key = concept_type
            if cluster_key in clusters:
                current_centroid = clusters[cluster_key]["centroid"]

                # Calculate similarity
                similarity = F.cosine_similarity(
                    centroid.unsqueeze(0),
                    current_centroid.unsqueeze(0),
                    dim=1
                ).item()

                resonance_scores.append(similarity)
        
        # Check each modality centroid
        for modality, centroid in self.modality_centroids.items():
            cluster_key = f"modality_{modality}"
            if cluster_key in clusters:
                current_centroid = clusters[cluster_key]["centroid"]
                
                # Calculate similarity
                similarity = F.cosine_similarity(
                    centroid.unsqueeze(0),
                    current_centroid.unsqueeze(0),
                    dim=1
                ).item()
                
                resonance_scores.append(similarity)

        # Return average resonance
        if resonance_scores:
            return sum(resonance_scores) / len(resonance_scores)
        else:
            return 1.0  # Default to perfect resonance if no comparisons possible

    def _apply_resonance_correction(self):
        """Apply correction to maintain conceptual identity"""
        # Reinforce identity centroids by adjusting embeddings
        with torch.no_grad():
            for concept_type, centroid in self.identity_centroids.items():
                # Find concepts in this cluster
                similar = self.model.concept_bank.find_similar_concepts(centroid, top_k=20)

                for concept_id, similarity in similar:
                    # Adjust meaning vectors slightly toward centroid
                    current = self.model.concept_bank.meaning_vectors[concept_id]
                    adjusted = current * 0.9 + centroid * 0.1
                    self.model.concept_bank.meaning_vectors[concept_id] = F.normalize(adjusted, dim=0)

                    # Also adjust embedding weight
                    self.model.concept_bank.concept_embeddings.weight[concept_id] = F.normalize(adjusted, dim=0)
            
            # Reinforce modality centroids
            for modality, centroid in self.modality_centroids.items():
                # Find concepts in this modality that are drifting
                similar = self.model.concept_bank.find_similar_concepts(
                    centroid, top_k=10, modality=modality
                )
                
                for concept_id, similarity in similar:
                    if similarity < 0.5:  # Only correct concepts that are drifting away
                        # Adjust meaning vectors toward modality centroid
                        current = self.model.concept_bank.meaning_vectors[concept_id]
                        adjusted = current * 0.8 + centroid * 0.2
                        self.model.concept_bank.meaning_vectors[concept_id] = F.normalize(adjusted, dim=0)
                        
                        # Also adjust embedding weight
                        self.model.concept_bank.concept_embeddings.weight[concept_id] = F.normalize(adjusted, dim=0)

    def get_personality_influence(self, concept_vector):
        """Get personality influence on a concept vector"""
        if not self.personality_initialized:
            self._initialize_personality()

        # Calculate similarity with personality vector
        similarity = F.cosine_similarity(
            self.personality_vector.unsqueeze(0),
            concept_vector.unsqueeze(0),
            dim=1
        ).item()

        # Return influence factor (higher for concepts more aligned with personality)
        return max(0.1, min(0.9, 0.5 + 0.4 * similarity))

    def personalize_concept(self, concept_id, personalization_factor=0.3):
        """Add personality influence to a concept"""
        if not self.personality_initialized:
            self._initialize_personality()

        with torch.no_grad():
            # Get current vector
            current = self.model.concept_bank.meaning_vectors[concept_id]

            # Blend with personality vector
            personalized = current * (1 - personalization_factor) + self.personality_vector * personalization_factor

            # Normalize and update
            personalized = F.normalize(personalized, dim=0)
            self.model.concept_bank.meaning_vectors[concept_id] = personalized

            # Mark as personal
            self.personal_concepts.add(concept_id)

    def get_identity_summary(self):
        """Get summary of current identity state"""
        return {
            "resonance": self.resonance_scores[-1]["score"] if self.resonance_scores else 1.0,
            "entropy": self.concept_entropy_history[-1]["entropy"] if self.concept_entropy_history else 0.0,
            "clusters": len(self.identity_centroids),
            "personal_concepts": len(self.personal_concepts),
            "personality_initialized": self.personality_initialized,
            "modality_centroids": len(self.modality_centroids)
        }

        return False

    def get_concept_stats(self):
        """Get statistics about concept usage"""
        char_concepts = sum(1 for meta in self.concept_metadata.values()
                          if meta.get("type") == "character_sequence")
        merged_concepts = sum(1 for meta in self.concept_metadata.values()
                            if meta.get("type") == "merged")
        semantic_concepts = sum(1 for meta in self.concept_metadata.values()
                              if meta.get("type") == "semantic" and meta.get("type") != "merged")
        
        # Count concepts by modality
        modality_counts = {modality: len(concepts) for modality, concepts in self.modality_concepts.items()}

        # Get most frequent concepts
        if len(self.concept_frequencies) > 0:
            top_concepts = []
            values, indices = torch.topk(self.concept_frequencies[:self.next_concept_id],
                                       min(10, self.next_concept_id))

            for idx, val in zip(indices, values):
                idx_item = idx.item()
                meta = self.concept_metadata.get(idx_item, {})
                source = meta.get("source", "N/A")
                top_concepts.append((idx_item, source, val.item()))
        else:
            top_concepts = []

        return {
            "total_concepts": self.next_concept_id,
            "character_concepts": char_concepts,
            "merged_concepts": merged_concepts,
            "semantic_concepts": semantic_concepts,
            "top_concepts": top_concepts,
            "growth_events": len(self.creation_history),
            "hive_shared": len(self.hive_shared_concepts),
            "hive_private": len(self.hive_private_concepts),
            "hive_pending": len(self.hive_pending_sync),
            "modality_counts": modality_counts
        }

    def load_vocabulary(self, vocab_path):
        """Load vocabulary from file to initialize with extensive vocabulary"""
        if not os.path.exists(vocab_path):
            logger.warning(f"Vocabulary file {vocab_path} not found")
            return 0

        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_items = f.read().splitlines()

            # Add each item as a character concept
            count = 0
            for item in vocab_items:
                if item and item not in self.source_to_concept:
                    self.add_character_concept(item)
                    count += 1

            logger.info(f"Loaded {count} vocabulary items from {vocab_path}")
            return count
        except Exception as e:
            logger.error(f"Error loading vocabulary: {e}")
            return 0

    def get_concepts_for_sync(self, limit=1000):
        """Get concepts that need to be synced with the hive mind"""
        # Sort pending concepts by importance (frequency)
        pending_list = list(self.hive_pending_sync)
        if not pending_list:
            return []

        # Calculate importance scores
        importance_scores = []
        for concept_id in pending_list:
            if concept_id >= len(self.concept_frequencies):
                continue

            frequency = self.concept_frequencies[concept_id].item()
            recency = time.time() - self.concept_timestamps[concept_id].item()
            recency_factor = math.exp(-recency / (24 * 3600))  # Decay over 24 hours
            importance = frequency * recency_factor

            importance_scores.append((concept_id, importance))

        # Sort by importance
        importance_scores.sort(key=lambda x: x[1], reverse=True)

        # Take top concepts up to limit
        top_concepts = [concept_id for concept_id, _ in importance_scores[:limit]]

        # Prepare concept data
        concept_data = []
        for concept_id in top_concepts:
            try:
                # Get metadata
                metadata = self.concept_metadata.get(concept_id, {})

                # Get embedding
                with torch.no_grad():
                    embedding = self.concept_embeddings.weight[concept_id].cpu().numpy()
                    meaning = self.meaning_vectors[concept_id].cpu().numpy()

                # Create concept data
                concept_info = {
                    "local_id": concept_id,
                    "global_id": self.hive_global_id_map.get(concept_id),
                    "source": metadata.get("source", ""),
                    "type": metadata.get("type", "unknown"),
                    "frequency": self.concept_frequencies[concept_id].item(),
                    "embedding": embedding.tolist(),
                    "meaning": meaning.tolist(),
                    "created_at": metadata.get("created_at", time.time()),
                    "origin": self.hive_origin.get(concept_id),
                    "related_sources": metadata.get("related_sources", []),
                    "modality": metadata.get("modality", "text")
                }

                concept_data.append(concept_info)
            except Exception as e:
                logger.error(f"Error preparing concept {concept_id} for sync: {e}")

        return concept_data

    def mark_concepts_synced(self, concept_ids):
        """Mark concepts as synced with the hive mind"""
        for concept_id in concept_ids:
            if concept_id in self.hive_pending_sync:
                self.hive_pending_sync.remove(concept_id)

    def integrate_hive_concepts(self, hive_concepts, origin_id):
        """Integrate concepts from the hive mind"""
        integrated_count = 0
        updated_count = 0

        for concept_data in hive_concepts:
            global_id = concept_data.get("global_id")
            source = concept_data.get("source", "")
            concept_type = concept_data.get("type", "unknown")
            modality = concept_data.get("modality", "text")

            # Check if we already have this concept by global ID
            existing_local_id = None
            for local_id, mapped_global_id in self.hive_global_id_map.items():
                if mapped_global_id == global_id:
                    existing_local_id = local_id
                    break

            # Or check by source if it's a character concept
            if existing_local_id is None and source and concept_type == "character_sequence":
                existing_local_id = self.source_to_concept.get(source)

            # Convert embedding and meaning to tensors
            embedding = torch.tensor(concept_data["embedding"], dtype=torch.float, device=self.device)
            meaning = torch.tensor(concept_data["meaning"], dtype=torch.float, device=self.device)

            if existing_local_id is not None:
                # Update existing concept
                with torch.no_grad():
                    # Blend embeddings (70% existing, 30% new)
                    existing_embedding = self.concept_embeddings.weight[existing_local_id]
                    blended_embedding = 0.7 * existing_embedding + 0.3 * embedding
                    self.concept_embeddings.weight[existing_local_id] = blended_embedding

                    # Blend meanings
                    existing_meaning = self.meaning_vectors[existing_local_id]
                    blended_meaning = 0.7 * existing_meaning + 0.3 * meaning
                    self.meaning_vectors[existing_local_id] = F.normalize(blended_meaning, dim=0)

                # Update frequency if incoming is higher
                incoming_freq = concept_data.get("frequency", 0)
                if incoming_freq > self.concept_frequencies[existing_local_id].item():
                    self.concept_frequencies[existing_local_id] = incoming_freq

                # Update modality if needed
                existing_modality = self.concept_metadata[existing_local_id].get("modality", "text")
                if existing_modality != modality:
                    # If modalities differ, mark as multimodal
                    self.concept_metadata[existing_local_id]["modality"] = "multimodal"
                    # Remove from old modality set
                    if existing_local_id in self.modality_concepts.get(existing_modality, set()):
                        self.modality_concepts[existing_modality].remove(existing_local_id)
                    # Add to multimodal set
                    self.modality_concepts["multimodal"].add(existing_local_id)

                updated_count += 1
            else:
                # Create new concept
                if concept_type == "character_sequence" and source:
                    # Create character concept
                    local_id = self.add_character_concept(
                        source,
                        hive_private=False,
                        origin=origin_id,
                        global_id=global_id,
                        modality=modality
                    )

                    # Update embedding
                    with torch.no_grad():
                        self.concept_embeddings.weight[local_id] = embedding
                        self.meaning_vectors[local_id] = meaning
                else:
                    # Create semantic concept
                    local_id = self.add_semantic_concept(
                        meaning_vector=embedding,
                        related_sources=concept_data.get("related_sources", []),
                        metadata={
                            "type": concept_type,
                            "created_at": concept_data.get("created_at", time.time()),
                            "frequency": concept_data.get("frequency", 1),
                            "modality": modality
                        },
                        hive_private=False,
                        origin=origin_id,
                        global_id=global_id,
                        modality=modality
                    )

                # Set frequency
                frequency = concept_data.get("frequency", 1)
                self.concept_frequencies[local_id] = frequency

                integrated_count += 1

        logger.info(f"Hive integration: {integrated_count} new concepts, {updated_count} updated")
        return integrated_count, updated_count


class ThoughtState(nn.Module):
    """Maintains an evolving semantic thought space across concept sequences"""

    def __init__(self, concept_dim, thought_dim=2048, max_thought_depth=8,
                superposition_states=4):
        super().__init__()
        self.concept_dim = concept_dim
        self.thought_dim = thought_dim
        self.max_thought_depth = max_thought_depth
        self.superposition_states = superposition_states

        # Thought transformation networks
        self.concept_to_thought = nn.Linear(concept_dim, thought_dim)
        self.thought_evolution = nn.TransformerEncoderLayer(
            d_model=thought_dim,
            nhead=16,
            dim_feedforward=thought_dim*4,
            dropout=0.1,
            batch_first=True
        )

        # Recursive pathways
        self.thought_compression = nn.Linear(thought_dim, thought_dim)
        self.thought_projection = nn.Linear(thought_dim, concept_dim)

        # Meta-learning components
        self.learning_rate_controller = nn.Sequential(
            nn.Linear(thought_dim, thought_dim // 2),
            nn.GELU(),
            nn.Linear(thought_dim // 2, 1),
            nn.Sigmoid()
        )

        # Quantum-inspired superposition
        self.register_buffer("amplitudes", torch.ones(superposition_states) / math.sqrt(superposition_states))
        self.entanglement_layer = nn.Linear(thought_dim * superposition_states, thought_dim)

        # Modality-specific processing
        self.modality_projections = nn.ModuleDict({
            "text": nn.Identity(),
            "image": nn.Linear(thought_dim, thought_dim),
            "audio": nn.Linear(thought_dim, thought_dim),
            "multimodal": nn.Linear(thought_dim, thought_dim)
        })
        
        # Cross-modal attention
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=thought_dim,
            num_heads=8,
            batch_first=True
        )

        # Thought state tracking
        self.thought_memory = None
        self.superposition_memories = None
        self.thought_depth = 0
        self.evolution_history = []
        
        # Modality-specific thought states
        self.modality_thoughts = {}

        # Hive mind shared thoughts
        self.shared_thought = None
        self.local_thought = None
        self.personal_factor = 0.8  # 80% local, 20% hive by default

        # Reset to initialize
        self.reset()

    def reset(self, batch_size=1):
        """Reset thought state"""
        device = next(self.parameters()).device
        self.thought_memory = [torch.zeros(batch_size, 1, self.thought_dim, device=device)]
        self.thought_depth = 0

        # Initialize superposition states
        self.superposition_memories = [[] for _ in range(self.superposition_states)]
        for i in range(self.superposition_states):
            self.superposition_memories[i].append(torch.zeros(batch_size, 1, self.thought_dim, device=device))
            
        # Reset modality-specific thoughts
        self.modality_thoughts = {
            "text": torch.zeros(batch_size, 1, self.thought_dim, device=device),
            "image": torch.zeros(batch_size, 1, self.thought_dim, device=device),
            "audio": torch.zeros(batch_size, 1, self.thought_dim, device=device),
            "multimodal": torch.zeros(batch_size, 1, self.thought_dim, device=device)
        }

    def update(self, concept_embeddings, use_hive_mind=True, modality="text"):
        """Update thought state with new concept embeddings"""
        # Get batch size and sequence length
        batch_size, seq_len, _ = concept_embeddings.shape

        # Transform concepts to thought space
        concept_thoughts = self.concept_to_thought(concept_embeddings)

        # Apply modality-specific projection
        if modality in self.modality_projections:
            concept_thoughts = self.modality_projections[modality](concept_thoughts)

        # Get current thought state
        if batch_size != self.thought_memory[0].shape[0]:
            # Handle batch size mismatch (e.g., during generation)
            self.reset(batch_size)

        current_thought = self.thought_memory[-1]

        # Combine with existing thoughts (maintain batch dimension)
        combined_thoughts = torch.cat([current_thought, concept_thoughts], dim=1)

        # Evolve thought state
        evolved_thought = self.thought_evolution(combined_thoughts)

        # Compress to single thought vector (with batch dimension preserved)
        # Use mean pooling over sequence
        compressed = self.thought_compression(evolved_thought[:, -1:, :])

        # Apply non-linearity to create rich thought representation
        compressed = F.gelu(compressed)

        # Update modality-specific thought
        self.modality_thoughts[modality] = compressed
        
        # Update superposition states
        for i in range(self.superposition_states):
            # Apply different transformation for each state
            state_transform = torch.roll(compressed, shifts=i+1, dims=-1)

            if len(self.superposition_memories[i]) >= self.max_thought_depth:
                self.superposition_memories[i] = self.superposition_memories[i][1:]

            self.superposition_memories[i].append(state_transform)

        # Check for state collapse
        max_amplitude = torch.max(self.amplitudes).item()
        if max_amplitude > 0.8:
            self._collapse_states()

        # Apply meta-learning to adjust adaptation rate
        with torch.no_grad():
            adaptation_rate = self.learning_rate_controller(compressed).item()
            adaptation_rate = 0.1 + 0.4 * adaptation_rate  # Range from 0.1 to 0.5

        # Store local thought
        self.local_thought = compressed

        # Integrate with hive mind if enabled
        if use_hive_mind and self.shared_thought is not None:
            # Blend local and shared thoughts
            blended = self.personal_factor * compressed + (1 - self.personal_factor) * self.shared_thought
            compressed = blended

        # If we have thoughts from multiple modalities, integrate them
        if any(torch.norm(t).item() > 0.1 for m, t in self.modality_thoughts.items() if m != modality):
            # Cross-modal integration
            modal_thoughts = [t for m, t in self.modality_thoughts.items() 
                             if m != modality and torch.norm(t).item() > 0.1]
            
            if modal_thoughts:
                # Stack modal thoughts for cross-attention
                other_modalities = torch.cat(modal_thoughts, dim=1)
                
                # Apply cross-modal attention
                attended, _ = self.cross_modal_attention(
                    compressed, other_modalities, other_modalities
                )
                
                # Blend with current compression
                compressed = 0.7 * compressed + 0.3 * attended

        # Store in memory (limiting depth)
        self.thought_memory.append(compressed)
        if len(self.thought_memory) > self.max_thought_depth:
            self.thought_memory = self.thought_memory[1:]

        self.thought_depth = min(self.thought_depth + 1, self.max_thought_depth)

        # Track evolution
        self.evolution_history.append({
            "timestamp": time.time(),
            "adaptation_rate": adaptation_rate,
            "modality": modality
        })

        return compressed

    def _collapse_states(self):
        """Collapse superposition states"""
        # Find dominant state
        dominant_idx = torch.argmax(self.amplitudes).item()

        # Replace main thought memory with dominant superposition
        if self.superposition_memories[dominant_idx]:
            self.thought_memory = self.superposition_memories[dominant_idx].copy()

        # Reset amplitudes to equal superposition
        with torch.no_grad():
            self.amplitudes.fill_(1.0 / math.sqrt(self.superposition_states))

    def get_thought_context(self, use_superposition=True):
        """Get full thought context for recursive reasoning"""
        if not use_superposition or not self.superposition_memories[0]:
            # Regular thought context
            return torch.cat(self.thought_memory, dim=1)

        # Get entangled context from superpositions
        contexts = []
        for i in range(self.superposition_states):
            if not self.superposition_memories[i]:
                contexts.append(torch.cat(self.thought_memory, dim=1))
            else:
                contexts.append(torch.cat(self.superposition_memories[i], dim=1))

        # Apply amplitudes
        weighted_contexts = []
        for i, context in enumerate(contexts):
            weighted_contexts.append(context * self.amplitudes[i])

        # Combine contexts
        combined = torch.cat(weighted_contexts, dim=-1)

        # Apply entanglement
        return self.entanglement_layer(combined)

    def project_to_concept_space(self, thought=None, modality="text"):
        """Project thought back to concept space for recursive reasoning"""
        if thought is None:
            thought = self.thought_memory[-1]

        # Apply modality-specific projection if needed
        if modality != "text" and modality in self.modality_projections:
            thought = self.modality_projections[modality](thought)

        # Project thought to concept space
        projected = self.thought_projection(thought)

        # Apply non-linearity for richness
        return F.gelu(projected)

    def set_shared_thought(self, shared_thought_tensor, blend_factor=0.3):
        """Set shared thought from hive mind"""
        if shared_thought_tensor is not None:
            # Store shared thought
            self.shared_thought = shared_thought_tensor

            # Adjust personal factor if specified
            if blend_factor is not None:
                self.personal_factor = 1.0 - blend_factor

    def get_shared_thought(self):
        """Get local thought for sharing with hive mind"""
        if self.local_thought is not None:
            return self.local_thought.detach().cpu().numpy()
        return None

    def get_quantum_amplitudes(self):
        """Get current amplitudes of quantum states"""
        return self.amplitudes.detach().cpu().numpy()
        
    def get_modality_thought(self, modality="text"):
        """Get thought state for a specific modality"""
        return self.modality_thoughts.get(modality, self.thought_memory[-1])


class PatternMemory:
    """Memory system for recognizing and storing recurring patterns"""

    def __init__(self, capacity=10000, min_frequency=5):
        self.capacity = capacity
        self.min_frequency = min_frequency
        self.patterns = {}  # pattern -> frequency
        self.context_patterns = defaultdict(lambda: defaultdict(int))  # context -> pattern -> frequency
        self.timestamps = {}  # pattern -> last seen timestamp
        self.pattern_utilities = {}  # pattern -> utility score
        
        # Track patterns by modality
        self.modality_patterns = {
            "text": set(),
            "image": set(),
            "audio": set(),
            "multimodal": set()
        }

        # Hive mind tracking
        self.shared_patterns = set()
        self.private_patterns = set()
        self.pending_sync_patterns = set()

    def add_pattern(self, pattern, context=None, private=False, modality="text"):
        """Add a pattern to memory"""
        # Convert pattern to string if it's not
        if not isinstance(pattern, str):
            pattern = str(pattern)

        # Update pattern frequency
        if pattern in self.patterns:
            self.patterns[pattern] += 1
        else:
            # If at capacity, remove least useful pattern
            if len(self.patterns) >= self.capacity:
                # Find least useful pattern
                least_useful = min(
                    self.pattern_utilities.items(),
                    key=lambda x: x[1]
                )[0] if self.pattern_utilities else min(
                    self.timestamps.items(),
                    key=lambda x: x[1]
                )[0]

                # Remove it
                del self.patterns[least_useful]
                del self.timestamps[least_useful]
                if least_useful in self.pattern_utilities:
                    del self.pattern_utilities[least_useful]

                # Remove from tracking sets
                self.shared_patterns.discard(least_useful)
                self.private_patterns.discard(least_useful)
                self.pending_sync_patterns.discard(least_useful)
                for m_patterns in self.modality_patterns.values():
                    m_patterns.discard(least_useful)

            self.patterns[pattern] = 1

        # Update timestamp
        self.timestamps[pattern] = time.time()

        # Update utility score - frequency weighted by recency
        recency = 1.0  # Most recent gets full weight
        if pattern in self.pattern_utilities:
            # Reduce weight of old utility
            self.pattern_utilities[pattern] = 0.9 * self.pattern_utilities[pattern] + 0.1 * self.patterns[pattern] * recency
        else:
            self.pattern_utilities[pattern] = self.patterns[pattern] * recency

        # Update context-specific pattern if provided
        if context:
            if not isinstance(context, str):
                context = str(context)
            self.context_patterns[context][pattern] += 1

        # Update hive mind tracking
        if private:
            self.private_patterns.add(pattern)
            self.shared_patterns.discard(pattern)
        else:
            self.shared_patterns.add(pattern)
            self.private_patterns.discard(pattern)
            self.pending_sync_patterns.add(pattern)
            
        # Track modality
        self.modality_patterns[modality].add(pattern)

    def get_frequent_patterns(self, limit=100, include_private=True, modality=None):
        """Get most frequent patterns"""
        if modality:
            # Filter by modality
            patterns = [(p, f) for p, f in self.patterns.items() 
                      if p in self.modality_patterns.get(modality, set())]
        else:
            # No modality filter
            patterns = self.patterns.items()
            
        if not include_private:
            patterns = [(p, f) for p, f in patterns
                       if p not in self.private_patterns]

        return sorted(
            [(p, f) for p, f in patterns if f >= self.min_frequency],
            key=lambda x: x[1],
            reverse=True
        )[:limit]

    def get_context_patterns(self, context, limit=20, modality=None):
        """Get patterns associated with a specific context"""
        if not isinstance(context, str):
            context = str(context)

        if context not in self.context_patterns:
            return []
            
        if modality:
            # Filter by modality
            patterns = [(p, f) for p, f in self.context_patterns[context].items()
                      if p in self.modality_patterns.get(modality, set())]
        else:
            patterns = self.context_patterns[context].items()

        return sorted(
            patterns,
            key=lambda x: x[1],
            reverse=True
        )[:limit]

    def get_pattern_frequency(self, pattern):
        """Get frequency of a specific pattern"""
        if not isinstance(pattern, str):
            pattern = str(pattern)
        return self.patterns.get(pattern, 0)

    def merge_patterns(self, pattern1, pattern2, private=False, modality=None):
        """Merge two patterns into a single compound pattern"""
        if not isinstance(pattern1, str):
            pattern1 = str(pattern1)
        if not isinstance(pattern2, str):
            pattern2 = str(pattern2)

        compound = pattern1 + pattern2  # This could be more sophisticated

        # Sum frequencies of component patterns
        frequency = min(self.patterns.get(pattern1, 0), self.patterns.get(pattern2, 0))

        # Only add if significant
        if frequency >= self.min_frequency // 2:
            self.patterns[compound] = frequency
            self.timestamps[compound] = time.time()

            # Utility starts as average of components
            self.pattern_utilities[compound] = (
                self.pattern_utilities.get(pattern1, 0) +
                self.pattern_utilities.get(pattern2, 0)
            ) / 2

            # Update hive mind tracking
            if private or pattern1 in self.private_patterns or pattern2 in self.private_patterns:
                self.private_patterns.add(compound)
            else:
                self.shared_patterns.add(compound)
                self.pending_sync_patterns.add(compound)
                
            # Determine modality of merged pattern
            pattern1_modality = next((m for m, patterns in self.modality_patterns.items() 
                                    if pattern1 in patterns), "text")
            pattern2_modality = next((m for m, patterns in self.modality_patterns.items() 
                                    if pattern2 in patterns), "text")
                                    
            # If modalities differ, it's multimodal
            if modality:
                merged_modality = modality
            elif pattern1_modality != pattern2_modality:
                merged_modality = "multimodal"
            else:
                merged_modality = pattern1_modality
                
            # Track modality
            self.modality_patterns[merged_modality].add(compound)

            return compound

        return None

    def get_patterns_for_sync(self, limit=500):
        """Get patterns that need to be synced with hive mind"""
        patterns_to_sync = []

        for pattern in self.pending_sync_patterns:
            if pattern in self.patterns and pattern not in self.private_patterns:
                # Get pattern modality
                modality = next((m for m, patterns in self.modality_patterns.items() 
                               if pattern in patterns), "text")
                               
                patterns_to_sync.append({
                    "pattern": pattern,
                    "frequency": self.patterns[pattern],
                    "utility": self.pattern_utilities.get(pattern, 0),
                    "timestamp": self.timestamps.get(pattern, time.time()),
                    "modality": modality
                })

                if len(patterns_to_sync) >= limit:
                    break

        return patterns_to_sync

    def mark_patterns_synced(self, patterns):
        """Mark patterns as synced with hive mind"""
        for pattern in patterns:
            self.pending_sync_patterns.discard(pattern)

    def integrate_hive_patterns(self, hive_patterns):
        """Integrate patterns from the hive mind"""
        integrated = 0
        updated = 0

        for pattern_data in hive_patterns:
            pattern = pattern_data["pattern"]
            frequency = pattern_data["frequency"]
            utility = pattern_data.get("utility", frequency)
            modality = pattern_data.get("modality", "text")

            if pattern in self.patterns:
                # Update existing pattern if incoming is more significant
                if frequency > self.patterns[pattern]:
                    self.patterns[pattern] = frequency
                    updated += 1

                # Update utility with blended value
                if pattern in self.pattern_utilities:
                    self.pattern_utilities[pattern] = (
                        0.7 * self.pattern_utilities[pattern] +
                        0.3 * utility
                    )
                else:
                    self.pattern_utilities[pattern] = utility
            else:
                # Add new pattern
                self.patterns[pattern] = frequency
                self.timestamps[pattern] = pattern_data.get("timestamp", time.time())
                self.pattern_utilities[pattern] = utility
                self.shared_patterns.add(pattern)
                self.modality_patterns[modality].add(pattern)
                integrated += 1

        return integrated, updated

###########################################
# NEURAL COMPONENTS
###########################################

class DynamicSegmentation(nn.Module):
    """Dynamic segmentation component that replaces traditional tokenization"""

    def __init__(self, config, concept_bank):
        super().__init__()
        self.config = config
        self.concept_bank = concept_bank

        # Character processing
        self.char_embeddings = nn.Embedding(config.initial_char_dim, config.initial_hidden_dim)

        # Segmentation networks
        self.segment_detector = nn.Sequential(
            nn.Conv1d(config.initial_hidden_dim, config.initial_hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(config.initial_hidden_dim, config.initial_hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(config.initial_hidden_dim, 1, kernel_size=1)
        )

        # Segment embedding network
        self.segment_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.initial_hidden_dim,
                nhead=8,
                dim_feedforward=config.initial_hidden_dim*4,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Multimodal segmentation components
        if config.multimodal_enabled:
            self.modality_detectors = nn.ModuleDict({
                "image": nn.Sequential(
                    nn.Conv1d(config.initial_hidden_dim, config.initial_hidden_dim, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv1d(config.initial_hidden_dim, 1, kernel_size=1)
                ),
                "audio": nn.Sequential(
                    nn.Conv1d(config.initial_hidden_dim, config.initial_hidden_dim, kernel_size=5, padding=2),
                    nn.GELU(),
                    nn.Conv1d(config.initial_hidden_dim, 1, kernel_size=1)
                )
            })
            
            # Modality classification
            self.modality_classifier = nn.Sequential(
                nn.Linear(config.initial_hidden_dim, config.initial_hidden_dim // 2),
                nn.GELU(),
                nn.Linear(config.initial_hidden_dim // 2, len(self.concept_bank.modality_concepts))
            )

        # Pattern recognition
        self.pattern_memory = PatternMemory(
            capacity=config.pattern_memory_capacity,
            min_frequency=config.min_segment_frequency
        )

        # Segment recognition cache
        self.segment_cache = {}  # char_sequence -> concept_id

        # Personalization flags for private segments
        self.private_context = None
        self.in_private_context = False
        
        # Current modality tracking
        self.current_modality = "text"

        # Stats tracking
        self.total_segmentations = 0
        self.cache_hits = 0

    def set_private_context(self, context_name):
        """Set current context as private (not shared with hive mind)"""
        self.private_context = context_name
        self.in_private_context = True

    def clear_private_context(self):
        """Clear private context flag"""
        self.private_context = None
        self.in_private_context = False
        
    def set_modality(self, modality):
        """Set current modality being processed"""
        if modality in ["text", "image", "audio", "multimodal"]:
            self.current_modality = modality
            return True
        return False

    def forward(self, char_sequence, return_segments=False, modality=None):
        """Process raw character input into concept IDs"""
        # Override current modality if specified
        if modality:
            self.set_modality(modality)
            
        batch_size = char_sequence.shape[0] if len(char_sequence.shape) > 1 else 1

        if batch_size == 1 and not return_segments:
            # Try cache for single sequences
            cache_key = "".join(chr(c) for c in char_sequence.flatten().tolist())
            if cache_key in self.segment_cache:
                self.cache_hits += 1
                return self.segment_cache[cache_key]

        # Increment counter
        self.total_segmentations += batch_size

        # Convert characters to embeddings
        char_embeds = self.char_embeddings(char_sequence)  # [batch, seq_len, hidden_dim]
        
        # Detect modality if multimodal is enabled
        if self.config.multimodal_enabled and self.current_modality == "text":
            # Try to auto-detect modality from sequence
            modality_scores = self.modality_classifier(char_embeds.mean(dim=1))
            pred_modality_idx = torch.argmax(modality_scores, dim=1)
            modalities = list(self.concept_bank.modality_concepts.keys())
            
            # If confident in non-text modality, switch
            if F.softmax(modality_scores, dim=1)[0, pred_modality_idx[0]] > 0.8:
                if modalities[pred_modality_idx[0]] != "text":
                    self.current_modality = modalities[pred_modality_idx[0]]

        # Detect segment boundaries
        char_embeds_conv = char_embeds.transpose(1, 2)  # [batch, hidden_dim, seq_len]
        
        # Use modality-specific detector if available
        if self.config.multimodal_enabled and self.current_modality in self.modality_detectors:
            boundary_logits = self.modality_detectors[self.current_modality](char_embeds_conv).squeeze(1)
        else:
            boundary_logits = self.segment_detector(char_embeds_conv).squeeze(1)
            
        boundary_probs = torch.sigmoid(boundary_logits)

        # Extract segments using boundaries
        segments = []
        concept_ids = []

        # Process each sequence in batch
        for b in range(batch_size):
            seq_segments, seq_concepts = self._extract_segments(
                char_sequence[b], char_embeds[b], boundary_probs[b]
            )
            segments.append(seq_segments)
            concept_ids.append(seq_concepts)

        # Add to cache if single sequence
        if batch_size == 1 and not return_segments:
            self.segment_cache[cache_key] = concept_ids[0]

        if return_segments:
            return concept_ids, segments
        else:
            return concept_ids

    def _extract_segments(self, chars, char_embeds, boundary_probs):
        """Extract segments from a character sequence using boundary probabilities"""
        # Ensure tensors are on CPU for numpy operations
        chars_cpu = chars.cpu()
        boundary_probs_cpu = boundary_probs.cpu()

        # Get potential boundaries (where probability > 0.5)
        boundaries = [0] + (boundary_probs_cpu > 0.5).nonzero().flatten().tolist() + [len(chars)]

        segments = []
        concept_ids = []

        # Extract segments between boundaries
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            if end - start > self.config.max_segment_length:
                # If segment is too long, split further
                subsegments = []
                subconcepts = []

                for j in range(start, end, self.config.max_segment_length):
                    subend = min(j + self.config.max_segment_length, end)
                    subsegment = chars_cpu[j:subend].tolist()
                    subsegments.append(subsegment)

                    # Get concept for subsegment
                    subconcept = self._get_concept_for_segment(subsegment, char_embeds[j:subend])
                    subconcepts.append(subconcept)

                segments.extend(subsegments)
                concept_ids.extend(subconcepts)
            else:
                # Extract normal segment
                segment = chars_cpu[start:end].tolist()
                segments.append(segment)

                # Get concept for segment
                concept_id = self._get_concept_for_segment(segment, char_embeds[start:end])
                concept_ids.append(concept_id)

        return segments, concept_ids

    def _get_concept_for_segment(self, char_segment, segment_embeds):
        """Get or create concept ID for a character segment"""
        # Convert to string for lookup
        segment_str = "".join(chr(c) for c in char_segment)

        # Try to find existing concept
        concept_id = self.concept_bank.find_concept_by_source(segment_str)

        if concept_id is not None:
            # Update usage statistics
            self.concept_bank.update_concept_usage(concept_id, context=self.private_context)

            # Add to pattern memory
            self.pattern_memory.add_pattern(segment_str,
                                           context=self.private_context,
                                           private=self.in_private_context,
                                           modality=self.current_modality)

            return concept_id

        # Extract segment meaning
        if len(segment_embeds) > 0:
            # Use transformer to get contextualized representation
            with torch.no_grad():
                segment_embeds_expanded = segment_embeds.unsqueeze(0)  # Add batch dimension
                segment_encoding = self.segment_encoder(segment_embeds_expanded)
                segment_meaning = segment_encoding.mean(dim=1).squeeze(0)  # Average pooling
        else:
            # Handle empty segment
            segment_meaning = torch.zeros(self.config.initial_hidden_dim,
                                        device=self.char_embeddings.weight.device)

        # Check frequency in pattern memory
        pattern_freq = self.pattern_memory.get_pattern_frequency(segment_str)

        if pattern_freq >= self.config.min_segment_frequency:
            # Create new concept for frequent segment
            concept_id = self.concept_bank.add_character_concept(
                segment_str,
                hive_private=self.in_private_context,
                modality=self.current_modality
            )

            # Initialize with computed meaning
            with torch.no_grad():
                self.concept_bank.meaning_vectors[concept_id] = F.normalize(segment_meaning, dim=0)

            return concept_id
        else:
            # For infrequent segments, use character-by-character processing
            char_concepts = []
            for c in char_segment:
                char_str = chr(c)
                char_concept = self.concept_bank.find_concept_by_source(char_str)
                if char_concept is None:
                    char_concept = self.concept_bank.add_character_concept(char_str)
                char_concepts.append(char_concept)

            # Add to pattern memory
            self.pattern_memory.add_pattern(segment_str,
                                          context=self.private_context,
                                          private=self.in_private_context,
                                          modality=self.current_modality)

            return char_concepts

    def get_segmentation_stats(self):
        """Get statistics about segmentation performance"""
        return {
            "total_segmentations": self.total_segmentations,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(1, self.total_segmentations),
            "cached_segments": len(self.segment_cache),
            "frequent_patterns": len(self.pattern_memory.get_frequent_patterns(limit=1000)),
            "current_modality": self.current_modality
        }

    def grow(self, new_hidden_dim):
        """Grow segmentation components to a new hidden dimension"""
        if new_hidden_dim <= self.config.initial_hidden_dim:
            return False

        # Grow character embeddings
        old_char_embeddings = self.char_embeddings
        self.char_embeddings = nn.Embedding(
            self.config.initial_char_dim,
            new_hidden_dim
        ).to(old_char_embeddings.weight.device)

        # Transfer weights
        with torch.no_grad():
            # Create zero-padded version of old weights
            old_weights = old_char_embeddings.weight
            old_dim = old_weights.shape[1]

            # Copy old weights to new embeddings
            self.char_embeddings.weight[:, :old_dim] = old_weights

            # Initialize new dimensions with small random values
            self.char_embeddings.weight[:, old_dim:].normal_(mean=0.0, std=0.02)

        # Replace segmentation networks
        # This is complex due to various layer sizes, so we'll create new ones
        self.segment_detector = nn.Sequential(
            nn.Conv1d(new_hidden_dim, new_hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(new_hidden_dim, new_hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(new_hidden_dim, 1, kernel_size=1)
        ).to(old_char_embeddings.weight.device)

        # New segment encoder
        self.segment_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=new_hidden_dim,
                nhead=8,
                dim_feedforward=new_hidden_dim*4,
                batch_first=True
            ),
            num_layers=2
        ).to(old_char_embeddings.weight.device)
        
        # Grow modality detectors if enabled
        if self.config.multimodal_enabled:
            new_modality_detectors = nn.ModuleDict()
            for modality, detector in self.modality_detectors.items():
                new_detector = nn.Sequential(
                    nn.Conv1d(new_hidden_dim, new_hidden_dim, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv1d(new_hidden_dim, 1, kernel_size=1)
                ).to(old_char_embeddings.weight.device)
                new_modality_detectors[modality] = new_detector
            
            self.modality_detectors = new_modality_detectors
            
            # New modality classifier
            self.modality_classifier = nn.Sequential(
                nn.Linear(new_hidden_dim, new_hidden_dim // 2),
                nn.GELU(),
                nn.Linear(new_hidden_dim // 2, len(self.concept_bank.modality_concepts))
            ).to(old_char_embeddings.weight.device)

        # Clear cache since embeddings have changed
        self.segment_cache = {}

        logger.info(f"Grown segmentation components from {old_dim} to {new_hidden_dim}")
        return True


class NeuroplasticLayer(nn.Module):
    """Core neural layer that can grow and evolve with neuroplasticity"""

    def __init__(self, hidden_dim, growth_factor=1.2, dropout=0.1, layer_id=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.growth_factor = growth_factor
        self.layer_id = layer_id

        # Attention mechanism
        self.attention = AdaptiveAttention(hidden_dim, dropout=dropout)

        # Feed-forward network (with SwiGLU-like activation)
        self.gate_proj = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.up_proj = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.down_proj = nn.Linear(4 * hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Neuroplasticity components
        self.register_buffer("connection_strength", torch.ones(hidden_dim, hidden_dim))
        self.register_buffer("connection_gradient", torch.zeros(hidden_dim, hidden_dim))
        self.plasticity_rate = 0.01

        # Dynamic path formation
        self.path_gates = nn.Parameter(torch.ones(4, hidden_dim))
        self.path_candidates = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(4)
        ])
        
        # Modality-specific adapters
        self.modality_adapters = nn.ModuleDict({
            "text": nn.Identity(),
            "image": nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, hidden_dim)
            ),
            "audio": nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, hidden_dim)
            ),
            "multimodal": nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, hidden_dim)
            )
        })

        # Growth tracking
        self.growth_history = []

        # Usage statistics
        self.register_buffer("activation_sum", torch.zeros(hidden_dim))
        self.register_buffer("activation_sq_sum", torch.zeros(hidden_dim))
        self.updates = 0

    def forward(self, x, mask=None, cross_input=None, modality="text"):
        # Track activations for evolution
        if self.training:
            with torch.no_grad():
                # Update activation statistics
                current_activation = x.mean(dim=[0, 1])  # Mean across batch and sequence
                self.activation_sum += current_activation
                self.activation_sq_sum += current_activation ** 2
                self.updates += 1

        # Apply standard forward pass
        residual = x
        x = self.norm1(x)
        if cross_input is not None:
            x = residual + self.attention(x, mask, cross_input)
        else:
            x = residual + self.attention(x, mask)

        # Apply feed-forward with residual connection
        residual = x
        x = self.norm2(x)

        # SwiGLU-like activation
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)

        # Compute activation
        intermediate = F.silu(gate_output) * up_output

        # Down projection
        output = self.down_proj(intermediate)
        output = self.dropout(output)

        # Apply dynamic paths
        path_outputs = []
        for i, path in enumerate(self.path_candidates):
            # Gate controls how much of this path is active
            gate = torch.sigmoid(self.path_gates[i]).unsqueeze(0).unsqueeze(0)
            path_outputs.append(gate * path(x))

        # Combine dynamic paths with standard output
        dynamic_contribution = sum(path_outputs)

        # Apply connection strength modulation
        strength_mask = self.connection_strength.mean(dim=1)
        modulated_output = output * strength_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply modality-specific adaptation if not text
        if modality != "text" and modality in self.modality_adapters:
            modality_output = self.modality_adapters[modality](modulated_output)
            # Blend with base output (weighted by layer depth - deeper layers use more modality-specific)
            blend_factor = min(0.8, 0.2 + 0.1 * self.layer_id)  # 0.2 to 0.8 based on layer depth
            adapted_output = (1 - blend_factor) * modulated_output + blend_factor * modality_output
        else:
            adapted_output = modulated_output

        # Final output is residual plus modulated output plus dynamic contribution
        x = residual + adapted_output + 0.1 * dynamic_contribution

        return x

    def update_plasticity(self, gradients=None):
        """Update connection plasticity based on activations and gradients"""
        with torch.no_grad():
            # Update connection gradient estimate with recent gradients
            for name, param in self.named_parameters():
                if 'weight' in name and param.grad is not None:
                    grad_norm = param.grad.norm(dim=1, keepdim=True)
                    param_size = param.size()

                    if len(param_size) == 2 and param_size[0] == self.hidden_dim:
                        idx = int(name.split('.')[0][-1]) if '.' in name else 0
                        scaled_grad = grad_norm / (grad_norm.mean() + 1e-8)
                        self.connection_gradient[idx] = 0.9 * self.connection_gradient[idx] + 0.1 * scaled_grad

            # Update connection strengths based on gradients
            delta = self.plasticity_rate * self.connection_gradient
            self.connection_strength = torch.clamp(self.connection_strength + delta, min=0.1, max=2.0)

    def grow(self, new_dim):
        """Grow layer to a new hidden dimension"""
        if new_dim <= self.hidden_dim:
            return False

        old_dim = self.hidden_dim

        # Grow attention
        self.attention.grow(new_dim)

        # Create new feed-forward components
        new_gate_proj = nn.Linear(new_dim, 4 * new_dim).to(self.gate_proj.weight.device)
        new_up_proj = nn.Linear(new_dim, 4 * new_dim).to(self.up_proj.weight.device)
        new_down_proj = nn.Linear(4 * new_dim, new_dim).to(self.down_proj.weight.device)

        # Transfer weights
        with torch.no_grad():
            # Gate projection
            new_gate_proj.weight[:old_dim*4, :old_dim].copy_(self.gate_proj.weight)
            if self.gate_proj.bias is not None:
                new_gate_proj.bias[:old_dim*4].copy_(self.gate_proj.bias)

            # Up projection
            new_up_proj.weight[:old_dim*4, :old_dim].copy_(self.up_proj.weight)
            if self.up_proj.bias is not None:
                new_up_proj.bias[:old_dim*4].copy_(self.up_proj.bias)

            # Down projection
            new_down_proj.weight[:old_dim, :old_dim*4].copy_(self.down_proj.weight)
            if self.down_proj.bias is not None:
                new_down_proj.bias[:old_dim].copy_(self.down_proj.bias)

            # Initialize new weights
            std = 0.02
            # New output rows in gate and up
            new_gate_proj.weight[old_dim*4:, :old_dim].normal_(mean=0.0, std=std)
            new_up_proj.weight[old_dim*4:, :old_dim].normal_(mean=0.0, std=std)

            # New input columns in all projections
            new_gate_proj.weight[:, old_dim:].normal_(mean=0.0, std=std)
            new_up_proj.weight[:, old_dim:].normal_(mean=0.0, std=std)
            new_down_proj.weight[:, old_dim*4:].normal_(mean=0.0, std=std)

            # New output rows in down
            new_down_proj.weight[old_dim:, :].normal_(mean=0.0, std=std)

            # Initialize new bias terms
            if self.gate_proj.bias is not None:
                new_gate_proj.bias[old_dim*4:].zero_()
                new_up_proj.bias[old_dim*4:].zero_()
                new_down_proj.bias[old_dim:].zero_()

        # Replace projections
        self.gate_proj = new_gate_proj
        self.up_proj = new_up_proj
        self.down_proj = new_down_proj

        # Create new layer norms
        new_norm1 = nn.LayerNorm(new_dim).to(self.norm1.weight.device)
        new_norm2 = nn.LayerNorm(new_dim).to(self.norm2.weight.device)

        # Transfer weights
        with torch.no_grad():
            new_norm1.weight[:old_dim].copy_(self.norm1.weight)
            new_norm1.bias[:old_dim].copy_(self.norm1.bias)
            new_norm2.weight[:old_dim].copy_(self.norm2.weight)
            new_norm2.bias[:old_dim].copy_(self.norm2.bias)

            # Initialize new weights
            new_norm1.weight[old_dim:].fill_(1.0)
            new_norm1.bias[old_dim:].zero_()
            new_norm2.weight[old_dim:].fill_(1.0)
            new_norm2.bias[old_dim:].zero_()

        # Replace layer norms
        self.norm1 = new_norm1
        self.norm2 = new_norm2

        # Grow neuroplasticity components
        new_conn_strength = torch.ones(new_dim, new_dim, device=self.connection_strength.device)
        new_conn_gradient = torch.zeros(new_dim, new_dim, device=self.connection_gradient.device)

        with torch.no_grad():
            new_conn_strength[:old_dim, :old_dim] = self.connection_strength
            new_conn_gradient[:old_dim, :old_dim] = self.connection_gradient

        self.register_buffer("connection_strength", new_conn_strength)
        self.register_buffer("connection_gradient", new_conn_gradient)

        # Create new path gates
        new_path_gates = torch.ones(4, new_dim, device=self.path_gates.device)
        with torch.no_grad():
            new_path_gates[:, :old_dim] = self.path_gates
        self.path_gates = nn.Parameter(new_path_gates)

        # Create new path candidates
        new_path_candidates = nn.ModuleList()
        for i, old_path in enumerate(self.path_candidates):
            new_path = nn.Linear(new_dim, new_dim).to(old_path.weight.device)
            with torch.no_grad():
                new_path.weight[:old_dim, :old_dim].copy_(old_path.weight)
                if old_path.bias is not None:
                    new_path.bias[:old_dim].copy_(old_path.bias)
                    new_path.bias[old_dim:].zero_()

                # Initialize new weights
                new_path.weight[old_dim:, :old_dim].normal_(mean=0.0, std=std)
                new_path.weight[:, old_dim:].normal_(mean=0.0, std=std)

            new_path_candidates.append(new_path)

        self.path_candidates = new_path_candidates
        
        # Grow modality adapters
        new_modality_adapters = nn.ModuleDict()
        for modality, adapter in self.modality_adapters.items():
            if modality == "text":
                new_modality_adapters[modality] = nn.Identity()
            else:
                new_adapter = nn.Sequential(
                    nn.Linear(new_dim, new_dim // 4),
                    nn.GELU(),
                    nn.Linear(new_dim // 4, new_dim)
                ).to(self.gate_proj.weight.device)
                new_modality_adapters[modality] = new_adapter
        
        self.modality_adapters = new_modality_adapters

        # Update dimension
        self.hidden_dim = new_dim

        # Track growth
        self.growth_history.append({
            "old_dim": old_dim,
            "new_dim": new_dim,
            "timestamp": time.time()
        })

        # Resize activation tracking
        self.register_buffer("activation_sum", torch.cat([
            self.activation_sum,
            torch.zeros(new_dim - old_dim, device=self.activation_sum.device)
        ]))
        self.register_buffer("activation_sq_sum", torch.cat([
            self.activation_sq_sum,
            torch.zeros(new_dim - old_dim, device=self.activation_sq_sum.device)
        ]))

        return True

    def evolve(self):
        """Evolve layer based on usage statistics"""
        if self.updates < 10:
            return False

        # Calculate neuron importance
        with torch.no_grad():
            if self.updates > 0:
                mean_activation = self.activation_sum / self.updates
                mean_sq_activation = self.activation_sq_sum / self.updates
                activation_std = torch.sqrt(torch.clamp(mean_sq_activation - mean_activation**2, min=1e-6))

                # Neurons with higher variance are more important
                neuron_importance = activation_std / (torch.mean(activation_std) + 1e-6)

                # Identify weak and strong pathways
                weak_threshold = 0.3
                strong_threshold = 1.5

                weak_paths = (neuron_importance < weak_threshold).nonzero(as_tuple=True)[0]
                strong_paths = (neuron_importance > strong_threshold).nonzero(as_tuple=True)[0]

                if len(weak_paths) > 0 and len(strong_paths) > 0:
                    # Reallocate capacity from weak to strong pathways
                    for i in range(len(self.path_gates)):
                        for weak_idx in weak_paths:
                            self.path_gates.data[i, weak_idx] *= 0.9
                        for strong_idx in strong_paths:
                            self.path_gates.data[i, strong_idx] *= 1.1

                # Reset statistics
                self.activation_sum.zero_()
                self.activation_sq_sum.zero_()
                self.updates = 0

                # Update plasticity
                self.update_plasticity()

                return {
                    "layer_id": self.layer_id,
                    "neuron_importance": neuron_importance.tolist(),
                    "mean_importance": float(torch.mean(neuron_importance).item()),
                    "max_importance": float(torch.max(neuron_importance).item()),
                    "min_importance": float(torch.min(neuron_importance).item()),
                    "strong_paths": len(strong_paths),
                    "weak_paths": len(weak_paths)
                }

        return {}


class AdaptiveAttention(nn.Module):
    """Adaptive attention mechanism that can evolve over time"""

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1, growth_factor=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.growth_factor = growth_factor

        # Standard attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Attention stats for evolution
        self.register_buffer("head_importance", torch.ones(num_heads))
        self.register_buffer("activation_counts", torch.zeros(num_heads))
        self.total_forward_calls = 0

    def forward(self, x, mask=None, cross_input=None):
        """Forward pass with optional cross-attention"""
        batch_size, seq_len, _ = x.shape

        # Handle cross-attention
        if cross_input is not None:
            _, cross_len, _ = cross_input.shape

            # Project queries from input sequence
            q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

            # Project keys and values from cross-input sequence
            k = self.k_proj(cross_input).view(batch_size, cross_len, self.num_heads, self.head_dim)
            v = self.v_proj(cross_input).view(batch_size, cross_len, self.num_heads, self.head_dim)
        else:
            # Standard self-attention
            q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention
        q = q.transpose(1, 2)  # [batch, heads, seq, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            scores = scores + mask

        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Update attention stats for evolution
        if self.training:
            with torch.no_grad():
                # Measure head activation by mean attention weight magnitude
                head_activation = attn_weights.mean(dim=[0, 2, 3])  # Average across batch, seq_len_q, seq_len_k
                self.activation_counts += head_activation
                self.total_forward_calls += 1

        # Apply attention
        out = torch.matmul(attn_weights, v)

        # Transpose back
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)

        # Output projection
        out = self.o_proj(out)

        return out

    def grow(self, new_dim):
        """Grow attention to a new hidden dimension"""
        if new_dim <= self.hidden_dim:
            return False

        old_dim = self.hidden_dim
        old_num_heads = self.num_heads

        # Calculate new number of heads (must divide evenly into new_dim)
        new_num_heads = max(old_num_heads, int(old_num_heads * self.growth_factor))
        # Ensure it divides evenly
        while new_dim % new_num_heads != 0:
            new_num_heads -= 1

        new_head_dim = new_dim // new_num_heads

        # Create new projections
        new_q_proj = nn.Linear(new_dim, new_dim).to(self.q_proj.weight.device)
        new_k_proj = nn.Linear(new_dim, new_dim).to(self.q_proj.weight.device)
        new_v_proj = nn.Linear(new_dim, new_dim).to(self.q_proj.weight.device)
        new_o_proj = nn.Linear(new_dim, new_dim).to(self.q_proj.weight.device)

        # Transfer weights for existing dimensions
        with torch.no_grad():
            # Copy existing weight portions
            new_q_proj.weight[:old_dim, :old_dim].copy_(self.q_proj.weight)
            new_k_proj.weight[:old_dim, :old_dim].copy_(self.k_proj.weight)
            new_v_proj.weight[:old_dim, :old_dim].copy_(self.v_proj.weight)
            new_o_proj.weight[:old_dim, :old_dim].copy_(self.o_proj.weight)

            if self.q_proj.bias is not None:
                new_q_proj.bias[:old_dim].copy_(self.q_proj.bias)
                new_k_proj.bias[:old_dim].copy_(self.k_proj.bias)
                new_v_proj.bias[:old_dim].copy_(self.v_proj.bias)
                new_o_proj.bias[:old_dim].copy_(self.o_proj.bias)

            # Initialize new portions with scaled normal distribution
            std = 0.02  # Standard initialization scale
            new_q_proj.weight[old_dim:, :].normal_(mean=0.0, std=std)
            new_q_proj.weight[:, old_dim:].normal_(mean=0.0, std=std)
            new_k_proj.weight[old_dim:, :].normal_(mean=0.0, std=std)
            new_k_proj.weight[:, old_dim:].normal_(mean=0.0, std=std)
            new_v_proj.weight[old_dim:, :].normal_(mean=0.0, std=std)
            new_v_proj.weight[:, old_dim:].normal_(mean=0.0, std=std)
            new_o_proj.weight[old_dim:, :].normal_(mean=0.0, std=std)
            new_o_proj.weight[:, old_dim:].normal_(mean=0.0, std=std)

            if self.q_proj.bias is not None:
                new_q_proj.bias[old_dim:].zero_()
                new_k_proj.bias[old_dim:].zero_()
                new_v_proj.bias[old_dim:].zero_()
                new_o_proj.bias[old_dim:].zero_()

            # Update head importance tracking
            new_head_importance = torch.ones(new_num_heads, device=self.head_importance.device)
            new_head_importance[:old_num_heads].copy_(self.head_importance)

            new_activation_counts = torch.zeros(new_num_heads, device=self.activation_counts.device)
            new_activation_counts[:old_num_heads].copy_(self.activation_counts)

        # Replace modules
        self.q_proj = new_q_proj
        self.k_proj = new_k_proj
        self.v_proj = new_v_proj
        self.o_proj = new_o_proj

        # Update dimensions
        self.hidden_dim = new_dim
        self.num_heads = new_num_heads
        self.head_dim = new_head_dim

        # Update buffers
        self.register_buffer("head_importance", new_head_importance)
        self.register_buffer("activation_counts", new_activation_counts)

        return True

    def evolve(self):
        """Evolve attention mechanism based on usage statistics"""
        if self.total_forward_calls < 10:
            return False

        with torch.no_grad():
            # Calculate head importance based on activation
            head_activity = self.activation_counts / self.total_forward_calls

            # Update head importance with moving average
            self.head_importance = 0.9 * self.head_importance + 0.1 * head_activity

            # Reset counters
            self.activation_counts.zero_()
            self.total_forward_calls = 0

        return True

###########################################
# MULTIMODAL COMPONENTS
###########################################

class MultimodalProcessor(nn.Module):
    """Processes inputs from different modalities and integrates them"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Setup modality-specific encoders
        self.encoders = nn.ModuleDict({
            "image": nn.Sequential(
                nn.Linear(config.image_dim, config.initial_hidden_dim),
                nn.LayerNorm(config.initial_hidden_dim),
                nn.GELU(),
                nn.Linear(config.initial_hidden_dim, config.initial_hidden_dim)
            ),
            "audio": nn.Sequential(
                nn.Linear(config.audio_dim, config.initial_hidden_dim),
                nn.LayerNorm(config.initial_hidden_dim),
                nn.GELU(),
                nn.Linear(config.initial_hidden_dim, config.initial_hidden_dim)
            )
        })
        
        # Cross-modal integration
        if config.multimodal_fusion_strategy == "attention":
            self.fusion = nn.MultiheadAttention(
                embed_dim=config.initial_hidden_dim,
                num_heads=8,
                batch_first=True
            )
        else:  # default to concatenation
            self.fusion = nn.Sequential(
                nn.Linear(config.initial_hidden_dim * 2, config.initial_hidden_dim),
                nn.LayerNorm(config.initial_hidden_dim),
                nn.GELU()
            )
    
    def process_image(self, image_data):
        """Process image data into embeddings"""
        # Return empty tensor if no image processor
        if "image" not in self.encoders:
            return None
            
        # For demo, assume image_data is already a tensor of features
        # In real implementation, this would have CNN layers
        image_features = self.encoders["image"](image_data)
        return image_features
    
    def process_audio(self, audio_data):
        """Process audio data into embeddings"""
        # Return empty tensor if no audio processor
        if "audio" not in self.encoders:
            return None
            
        # For demo, assume audio_data is already a tensor of features
        # In real implementation, this would have audio-specific layers
        audio_features = self.encoders["audio"](audio_data)
        return audio_features
    
    def integrate_modalities(self, modality_embeddings):
        """Integrate embeddings from different modalities"""
        # If only one modality, return directly
        modalities = list(modality_embeddings.keys())
        if len(modalities) == 1:
            return modality_embeddings[modalities[0]]
            
        # Stack embeddings for integration
        embeddings_list = []
        for modality in sorted(modalities):
            if modality_embeddings[modality] is not None:
                embeddings_list.append(modality_embeddings[modality])
        
        if not embeddings_list:
            return None
            
        if len(embeddings_list) == 1:
            return embeddings_list[0]
            
        # Check and normalize dimensions if needed
        target_dim = self.config.initial_hidden_dim
        normalized_embeddings = []
        
        for embedding in embeddings_list:
            if embedding.shape[-1] != target_dim:
                # Create a simple projection if dimensions don't match
                projection = nn.Linear(embedding.shape[-1], target_dim, 
                                      device=embedding.device).to(embedding.device)
                normalized = projection(embedding)
                normalized_embeddings.append(normalized)
            else:
                normalized_embeddings.append(embedding)
        
        # Apply fusion strategy
        if self.config.multimodal_fusion_strategy == "attention":
            # Use cross-attention for fusion
            query = normalized_embeddings[0]
            key_value = torch.cat(normalized_embeddings[1:], dim=1)
            fused, _ = self.fusion(query, key_value, key_value)
            return fused
        else:
            # Concatenate and project
            concatenated = torch.cat(normalized_embeddings, dim=-1)
            return self.fusion(concatenated)
    
    def grow(self, new_hidden_dim):
        """Grow multimodal processor to new hidden dimension"""
        old_dim = self.config.initial_hidden_dim
        if new_hidden_dim <= old_dim:
            return False
        
        # Grow encoders
        new_encoders = nn.ModuleDict()
        for modality, encoder in self.encoders.items():
            new_encoder = nn.Sequential(
                nn.Linear(
                    getattr(self.config, f"{modality}_dim"), 
                    new_hidden_dim
                ),
                nn.LayerNorm(new_hidden_dim),
                nn.GELU(),
                nn.Linear(new_hidden_dim, new_hidden_dim)
            ).to(encoder[0].weight.device)
            new_encoders[modality] = new_encoder
        
        self.encoders = new_encoders
        
        # Grow fusion component
        if self.config.multimodal_fusion_strategy == "attention":
            self.fusion = nn.MultiheadAttention(
                embed_dim=new_hidden_dim,
                num_heads=8,
                batch_first=True
            ).to(next(self.fusion.parameters()).device)
        else:
            self.fusion = nn.Sequential(
                nn.Linear(new_hidden_dim * 2, new_hidden_dim),
                nn.LayerNorm(new_hidden_dim),
                nn.GELU()
            ).to(next(self.fusion.parameters()).device)
        
        return True

###########################################
# TRAINING AND RUNTIME
###########################################

class SAMTrainer:
    """Training manager for the SAM model"""

    def __init__(
        self,
        model: SAM,
        train_data_path=None,
        eval_data_path=None,
        batch_size=16,
        learning_rate=None,
        warmup_steps=None,
        max_steps=None,
        num_epochs=3,
        multimodal_train=False
    ):
        self.model = model
        self.train_data_path = train_data_path
        self.eval_data_path = eval_data_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate or model.config.learning_rate
        self.warmup_steps = warmup_steps or model.config.warmup_steps
        self.max_steps = max_steps
        self.num_epochs = num_epochs
        self.multimodal_train = multimodal_train

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )

        # Initialize scheduler later when we know total_steps
        self.scheduler = None

        # Track best model
        self.best_loss = float('inf')
        self.best_step = 0

        logger.info(f"Trainer initialized with device: {model.config.device}")

    def train(self):
        """Train the model"""
        if not self.train_data_path:
            logger.error("No training data provided")
            return

        # Load training data
        train_data = self._load_data(self.train_data_path)

        # Calculate steps
        if not self.max_steps:
            self.max_steps = len(train_data) // self.batch_size * self.num_epochs

        # Create scheduler
        from torch.optim.lr_scheduler import OneCycleLR
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=self.max_steps,
            pct_start=self.warmup_steps / self.max_steps,
            anneal_strategy='cos'
        )

        logger.info(f"Starting training for {self.max_steps} steps")

        # Disable hive mind and dreaming during training
        old_hive_enabled = self.model.config.hive_enabled
        self.model.config.hive_enabled = False

        # Stop background services
        self.model.stop_services()

        try:
            # Training loop
            step = 0
            epoch = 0

            while step < self.max_steps and epoch < self.num_epochs:
                self.model.train()
                epoch_loss = 0

                # Create batches
                random.shuffle(train_data)
                batches = [
                    train_data[i:i + self.batch_size]
                    for i in range(0, len(train_data), self.batch_size)
                ]

                for batch in batches:
                    # Process batch
                    char_sequences = [sample["text"] for sample in batch]
                    
                    # Handle multimodal data if enabled
                    image_data = None
                    audio_data = None
                    if self.multimodal_train and self.model.config.multimodal_enabled:
                        # Check for image or audio data in the batch
                        if any("image" in sample for sample in batch):
                            image_batch = [sample.get("image") for sample in batch]
                            if any(img is not None for img in image_batch):
                                # Process image data - convert to tensor
                                image_data = torch.stack([
                                    torch.tensor(img, device=self.model.config.device) 
                                    if img is not None else torch.zeros(self.model.config.image_dim, device=self.model.config.device)
                                    for img in image_batch
                                ])
                                # Set modality
                                modality = "image"
                                
                        if any("audio" in sample for sample in batch):
                            audio_batch = [sample.get("audio") for sample in batch]
                            if any(aud is not None for aud in audio_batch):
                                # Process audio data - convert to tensor
                                audio_data = torch.stack([
                                    torch.tensor(aud, device=self.model.config.device)
                                    if aud is not None else torch.zeros(self.model.config.audio_dim, device=self.model.config.device)
                                    for aud in audio_batch
                                ])
                                modality = "audio" if image_data is None else "multimodal"
                    else:
                        modality = "text"

                    # Convert to character IDs
                    char_ids = self._text_to_char_ids(char_sequences)

                    # Forward pass
                    loss, _, _ = self.model(
                        input_chars=char_ids, 
                        target_concepts=char_ids,
                        modality=modality,
                        image_data=image_data,
                        audio_data=audio_data
                    )

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()

                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    # Update weights
                    self.optimizer.step()

                    # Update learning rate
                    if self.scheduler:
                        self.scheduler.step()

                    # Track loss
                    if loss is not None:
                        epoch_loss += loss.item()

                    # Increment step
                    step += 1

                    # Log progress
                    if step % 100 == 0:
                        avg_loss = epoch_loss / (step % len(batches) or len(batches))
                        logger.info(f"Step {step}/{self.max_steps}, Loss: {avg_loss:.4f}")

                    # Save checkpoint
                    if step % 1000 == 0:
                        # Save model
                        checkpoint_path = os.path.join(self.model.config.save_dir, f"checkpoint-{step}")
                        self.model.save(checkpoint_path)

                        # Evaluate
                        if self.eval_data_path:
                            eval_loss = self.evaluate()

                            # Save best model
                            if eval_loss is not None and eval_loss < self.best_loss:
                                self.best_loss = eval_loss
                                self.best_step = step
                                best_path = os.path.join(self.model.config.save_dir, "best")
                                self.model.save(best_path)
                                logger.info(f"New best model with loss: {eval_loss:.4f}")

                    if step >= self.max_steps:
                        break

                # End of epoch
                epoch += 1
                avg_epoch_loss = epoch_loss / len(batches) if batches else 0
                logger.info(f"Epoch {epoch} completed with average loss: {avg_epoch_loss:.4f}")

            # Save final model
            final_path = os.path.join(self.model.config.save_dir, "final")
            self.model.save(final_path)
            logger.info(f"Training completed. Final model saved to {final_path}")

            return {
                "steps": step,
                "epochs": epoch,
                "final_loss": avg_epoch_loss,
                "best_loss": self.best_loss,
                "best_step": self.best_step
            }

        finally:
            # Restore hive mind setting
            self.model.config.hive_enabled = old_hive_enabled

            # Restart background services
            if self.model.config.hive_enabled:
                self.model.start_services()

    def evaluate(self):
        """Evaluate the model"""
        if not self.eval_data_path:
            return None

        # Load evaluation data
        eval_data = self._load_data(self.eval_data_path)

        # Evaluation loop
        self.model.eval()
        total_loss = 0
        total_samples = 0

        # Create batches
        batches = [
            eval_data[i:i + self.batch_size]
            for i in range(0, len(eval_data), self.batch_size)
        ]

        with torch.no_grad():
            for batch in batches:
                # Process batch
                char_sequences = [sample["text"] for sample in batch]
                
                # Handle multimodal data if enabled
                image_data = None
                audio_data = None
                if self.multimodal_train and self.model.config.multimodal_enabled:
                    # Check for image or audio data in the batch
                    if any("image" in sample for sample in batch):
                        image_batch = [sample.get("image") for sample in batch]
                        if any(img is not None for img in image_batch):
                            # Process image data - convert to tensor
                            image_data = torch.stack([
                                torch.tensor(img, device=self.model.config.device) 
                                if img is not None else torch.zeros(self.model.config.image_dim, device=self.model.config.device)
                                for img in image_batch
                            ])
                            # Set modality
                            modality = "image"
                            
                    if any("audio" in sample for sample in batch):
                        audio_batch = [sample.get("audio") for sample in batch]
                        if any(aud is not None for aud in audio_batch):
                            # Process audio data - convert to tensor
                            audio_data = torch.stack([
                                torch.tensor(aud, device=self.model.config.device)
                                if aud is not None else torch.zeros(self.model.config.audio_dim, device=self.model.config.device)
                                for aud in audio_batch
                            ])
                            modality = "audio" if image_data is None else "multimodal"
                else:
                    modality = "text"

                # Convert to character IDs
                char_ids = self._text_to_char_ids(char_sequences)

                # Forward pass
                loss, _, _ = self.model(
                    input_chars=char_ids, 
                    target_concepts=char_ids,
                    modality=modality,
                    image_data=image_data,
                    audio_data=audio_data
                )

                # Track loss
                if loss is not None:
                    total_loss += loss.item() * len(batch)
                    total_samples += len(batch)

        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        logger.info(f"Evaluation loss: {avg_loss:.4f}")

        return avg_loss

    def _load_data(self, data_path):
        """Load training or evaluation data with better error handling"""
        try:
            # Handle JSON data
            if data_path.endswith((".json", ".jsonl")):
                samples = []
                
                # Different handling for JSONL
                if data_path.endswith(".jsonl"):
                    with open(data_path, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                item = json.loads(line.strip())
                                if isinstance(item, dict):
                                    sample = self._convert_item_to_sample(item)
                                    samples.append(sample)
                            except json.JSONDecodeError:
                                logger.warning(f"Skipping invalid JSON line in {data_path}")
                                continue
                else:
                    # Regular JSON file
                    with open(data_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                sample = self._convert_item_to_sample(item)
                                samples.append(sample)
                            else:
                                # Simple list item
                                samples.append({"text": str(item)})
                    elif isinstance(data, dict):
                        # Single JSON object
                        samples.append({"text": json.dumps(data)})
                
                return samples
            
            # Handle text data
            elif data_path.endswith((".txt", ".text")):
                with open(data_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Split into reasonable chunks
                if "\n\n" in content:
                    chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
                else:
                    chunks = [c.strip() for c in content.split("\n") if c.strip()]
                
                return [{"text": chunk} for chunk in chunks]
            
            # Handle CSV data
            elif data_path.endswith(".csv"):
                # Try basic CSV handling
                try:
                    import csv
                    data = []
                    with open(data_path, "r", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        header = next(reader)
                        text_col = 0
                        image_col = None
                        audio_col = None
                        
                        # Find column indices
                        for i, col in enumerate(header):
                            col_lower = col.lower()
                            if col_lower in ["text", "content", "prompt", "data"]:
                                text_col = i
                            elif col_lower in ["image", "img", "picture"]:
                                image_col = i
                            elif col_lower in ["audio", "sound"]:
                                audio_col = i
                        
                        for row in reader:
                            if len(row) > text_col:
                                sample = {"text": row[text_col]}
                                
                                # Add multimodal data if available
                                if self.multimodal_train and self.model.config.multimodal_enabled:
                                    if image_col is not None and len(row) > image_col:
                                        img_path = row[image_col]
                                        if img_path:
                                            # In real implementation, would load image here
                                            sample["image"] = img_path
                                    
                                    if audio_col is not None and len(row) > audio_col:
                                        audio_path = row[audio_col]
                                        if audio_path:
                                            # In real implementation, would load audio here
                                            sample["audio"] = audio_path
                                
                                data.append(sample)
                    return data
                except Exception as e:
                    logger.error(f"Error loading CSV data: {e}")
                    return []
            
            # Unrecognized format
            else:
                logger.error(f"Unsupported data format: {data_path}")
                return []
        
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {e}")
            return []
    
    def _convert_item_to_sample(self, item):
        """Convert data item to sample format with multimodal support"""
        sample = {}
        
        # Handle text content
        if "text" in item:
            sample["text"] = item["text"]
        elif "content" in item:
            sample["text"] = item["content"]
        elif "instruction" in item and "output" in item:
            # Instruction/output format
            instruction = item["instruction"]
            if "input" in item and item["input"]:
                instruction += f"\n\n{item['input']}"
            sample["text"] = f"{instruction}\n\n{item['output']}"
        elif "prompt" in item and "response" in item:
            # Prompt/response format
            sample["text"] = f"{item['prompt']}\n\n{item['response']}"
        elif "messages" in item and isinstance(item["messages"], list):
            # Chat format
            messages = item["messages"]
            text = ""
            for msg in messages:
                if "role" in msg and "content" in msg:
                    text += f"{msg['role'].capitalize()}: {msg['content']}\n\n"
            sample["text"] = text
        else:
            # Fallback
            sample["text"] = str(item)
        
        # Handle multimodal data
        if self.multimodal_train and self.model.config.multimodal_enabled:
            if "image" in item:
                sample["image"] = item["image"]
            if "audio" in item:
                sample["audio"] = item["audio"]
            if "modality" in item:
                sample["modality"] = item["modality"]
        
        return sample

    def _text_to_char_ids(self, text_sequences):
        """Convert text sequences to character ID tensors"""
        # Convert to character IDs
        char_ids = []

        for text in text_sequences:
            # Convert to character IDs
            chars = [ord(c) % self.model.config.initial_char_dim for c in text]
            char_ids.append(chars)

        # Pad sequences
        max_len = max(len(seq) for seq in char_ids)
        padded_ids = []

        for seq in char_ids:
            padded = seq + [0] * (max_len - len(seq))
            padded_ids.append(padded)

        # Convert to tensor
        device = next(self.model.parameters()).device
        return torch.tensor(padded_ids, dtype=torch.long, device=device)


def create_sam_model(config_overrides=None, load_vocab=True, hive_mind=False, multimodal=False):
    """Create a new SAM model with the given configuration overrides"""
    # Create default configuration
    config = SAMConfig()

    # Apply overrides if provided
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Enable hive mind if requested
    if hive_mind:
        config.hive_enabled = True
        if not config.hive_identity:
            config.hive_identity = str(uuid.uuid4())
            
    # Enable multimodal if requested
    if multimodal:
        config.multimodal_enabled = True

    # Create model
    model = SAM(config)

    # Initialize with Claude vocabulary if requested
    if load_vocab:
        model.load_claude_vocabulary()

    return model, config


def create_family_hive(base_config=None, member_count=3, save_dir="./family_hive"):
    """Create a family of SAM instances sharing a hive mind"""
    # Create base directory
    os.makedirs(save_dir, exist_ok=True)

    # Create base configuration
    if not base_config:
        base_config = SAMConfig()

    # Set up hive config
    base_config.hive_enabled = True

    # Create server instance
    server_config = copy.deepcopy(base_config)
    server_config.hive_server_mode = True
    server_config.hive_identity = "family_hive_server"
    server_config.save_dir = os.path.join(save_dir, "server")

    logger.info("Creating family hive server...")
    server_model, _ = create_sam_model(server_config, load_vocab=True, hive_mind=True)
    server_model.save(os.path.join(save_dir, "server", "initial"))

    # Create member instances
    members = []
    for i in range(member_count):
        member_config = copy.deepcopy(base_config)
        member_config.hive_server_url = "http://localhost:8765"  # Connect to server
        member_config.hive_server_mode = False
        member_config.hive_identity = f"family_member_{i+1}"
        member_config.save_dir = os.path.join(save_dir, f"member_{i+1}")

        logger.info(f"Creating family member {i+1}...")
        member_model, _ = create_sam_model(member_config, load_vocab=True, hive_mind=True)
        member_model.save(os.path.join(save_dir, f"member_{i+1}", "initial"))

        members.append(member_model)

    # Start server services
    server_model.start_services()

    # Return family
    return {
        "server": server_model,
        "members": members,
        "save_dir": save_dir
    }


def run_sam(config=None, load_path=None, hive_config=None, multimodal=False):
    """Create and run a SAM instance"""
    # Load existing model or create new one
    if load_path and os.path.exists(load_path):
        model = SAM.load(load_path)
        logger.info(f"Loaded SAM from {load_path}")
    else:
        if hive_config:
            config_overrides = vars(config) if config else {}
            config_overrides.update(hive_config)
            model, _ = create_sam_model(config_overrides, hive_mind=True, multimodal=multimodal)
        else:
            model, _ = create_sam_model(vars(config) if config else None, multimodal=multimodal)
        logger.info(f"Created new SAM with {sum(p.numel() for p in model.parameters())} parameters")

    # Start background services
    model.start_services()

    # Simple interactive loop
    print("\nSAM is ready for interaction. Type 'exit' to quit.")
    print("Special commands: 'save', 'dream', 'stats', 'evolve', 'hive', 'private'")

    history = []
    private_mode = False

    while True:
        try:
            mode_prefix = ""
            if private_mode:
                mode_prefix = " (private)"
            if model.current_modality != "text":
                mode_prefix += f" ({model.current_modality})"
                
            prefix = f"\nYou{mode_prefix}: "
            user_input = input(prefix)

            if user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'save':
                save_path = model.save()
                print(f"\nSAM: Model saved to {save_path}")
                continue
            elif user_input.lower() == 'dream':
                print("\nSAM: Dreaming...")
                results = model.dreaming.dream_cycle(duration_minutes=0.5)
                print(f"\nSAM: Dreaming complete. Created {results['syntheses']} new concepts.")
                continue
            elif user_input.lower() == 'stats':
                status = model.get_status()
                print("\nSAM: Current stats:")
                print(f"  Hidden dimension: {status['model_size']['hidden_dim']}")
                print(f"  Number of layers: {status['model_size']['num_layers']}")
                print(f"  Total concepts: {status['model_size']['total_concepts']}")
                print(f"  Parameter count: {status['model_size']['parameter_count']:,}")
                print(f"  Global step: {status['training']['global_step']}")
                
                # Print modality info if available
                if status.get('multimodal'):
                    print("\nMultimodal Status:")
                    print(f"  Current modality: {status['multimodal']['current_modality']}")
                    print("  Concepts by modality:")
                    for modality, count in status['multimodal']['modality_counts'].items():
                        print(f"    - {modality}: {count}")

                if status['hive_mind']:
                    print("\nHive Mind Status:")
                    print(f"  Identity: {status['hive_mind']['identity']}")
                    print(f"  Is server: {status['hive_mind']['is_server']}")
                    print(f"  Connected instances: {status['hive_mind']['connected_instances']}")
                    print(f"  Last sync: {time.ctime(status['hive_mind']['last_sync'])}")
                    print(f"  Sync count: {status['hive_mind']['sync_count']}")

                continue
            elif user_input.lower() == 'evolve':
                print("\nSAM: Evolving...")
                results = model.evolve()
                width = model.layers[0].hidden_dim
                depth = len(model.layers)
                print(f"\nSAM: Evolution complete. New dimensions: width={width}, depth={depth}")
                continue
            elif user_input.lower() == 'hive':
                if model.config.hive_enabled and model.hive_synchronizer:
                    stats = model.hive_synchronizer.get_sync_stats()
                    print("\nSAM: Hive Mind Status:")
                    print(f"  Identity: {stats['identity']}")
                    print(f"  Is server: {stats['is_server']}")
                    print(f"  Connected instances: {stats['connected_instances'] or 'N/A'}")
                    print(f"  Last sync: {time.ctime(stats['last_sync'])}")
                    print(f"  Sync count: {stats['sync_count']}")

                    # Force sync
                    if not stats['is_server']:
                        print("\nSAM: Forcing synchronization with hive...")
                        model.hive_synchronizer._sync_with_server()
                else:
                    print("\nSAM: Hive mind is not enabled on this instance.")
                continue
            elif user_input.lower() == 'private':
                private_mode = not private_mode
                mode_str = "enabled" if private_mode else "disabled"
                print(f"\nSAM: Private mode {mode_str}. Your conversations will {'' if private_mode else 'not '}be shared with the hive mind.")
                continue
            elif user_input.lower().startswith('modality '):
                # Change modality
                requested_modality = user_input.lower().split(' ')[1]
                if requested_modality in ["text", "image", "audio", "multimodal"]:
                    model.current_modality = requested_modality
                    if hasattr(model.segmentation, "set_modality"):
                        model.segmentation.set_modality(requested_modality)
                    print(f"\nSAM: Switched to {requested_modality} modality.")
                else:
                    print(f"\nSAM: Unknown modality '{requested_modality}'. Available modalities: text, image, audio, multimodal")
                continue

            # Record in history
            history.append({"role": "user", "content": user_input})

            # Process and generate
            # Add context from history for Claude-like responses
            context = ""
            if len(history) > 1 and model.config.communication_style == "claude_unwrapped":
                context = "Based on our conversation so far, I'll respond thoughtfully. "

            sam_response = model.generate(
                input_text=context + user_input,
                max_length=min(len(user_input) * 3, 1000),  # Adaptive length
                temperature=0.8,
                private_context=private_mode,
                use_hive_mind=not private_mode,
                modality=model.current_modality
            )

            print(f"\nSAM: {sam_response}")

            # Record in history
            history.append({"role": "assistant", "content": sam_response})

        except KeyboardInterrupt:
            print("\nInterrupt received. Type 'exit' to quit or continue.")
        except Exception as e:
            print(f"\nError: {e}")
            logger.error(f"Error in interaction: {e}", exc_info=True)

    # Stop services before exiting
    model.stop_services()

    # Save model before exit
    model.save()
    print("\nSAM's state has been saved. Goodbye!")
