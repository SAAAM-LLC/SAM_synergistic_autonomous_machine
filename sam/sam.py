
# sam.py - Complete Synergistic Autonomous Machine with Unified Neural-Linguistic Architecture
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import types
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
import re
from torch.optim.lr_scheduler import OneCycleLR
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
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
    initial_hidden_dim: int = 536
    initial_num_layers: int = 8
    max_position_embeddings: int = 8192

    # Growth parameters
    max_hidden_dim: int = 4096
    max_num_layers: int = 16
    max_growth_steps: int = 10000
    growth_factor: float = 1.5
    min_layer_usage_threshold: float = 0.4

    # Memory systems
    concept_memory_size: int = 50000
    concept_dim: int = 536
    thought_dim: int = 2048
    max_thought_depth: int = 64
    pattern_memory_capacity: int = 50000

    # Learning parameters
    learning_rate: float = 3e-5
    warmup_steps: int = 1000
    adaption_rate: float = 0.500

    # Unified processing parameters
    unified_perception: bool = False
    emergent_conceptualization: bool = True
    direct_signal_processing: bool = True
    cross_modal_learning: bool = True

    # Signal processing parameters
    signal_kernel_size: int = 5
    signal_channels: int = 32
    signal_buffer_size: int = 1024

    # Emergent concept parameters
    concept_cluster_threshold: float = 0.90
    concept_creation_threshold: float = 0.95
    max_emergent_concepts: int = 20000
    concept_pruning_interval: int = 1000

    # Dreaming parameters
    dreaming_enabled: bool = False
    dream_batch_size: int = 5
    dream_max_length: int = 256
    dream_cycle_minutes: float = 0.2

    # Consciousness parameters
    stability_threshold: float = 0.90
    novelty_weight: float = 0.4

    # Paths for persistence
    save_dir: str = "./data"
    experiences_path: str = "./data/experiences.json"
    concepts_path: str = "./data/concepts.json"
    growth_log_path: str = "./data/growth_log.json"

    # Runtime parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Communication Style
    communication_style: str = "standard"  # "flexible", "standard", "Adaptive" etc.

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
    offload_threshold: float = 0.75

    # Multimodal capabilities
    multimodal_enabled: bool = False
    image_dim: int = 768
    audio_dim: int = 512
    multimodal_fusion_strategy: str = "attention"  # "attention", "concatenation"

    def validate(self):
        """Validate configuration parameters"""
        # Check dimension relationships
        if self.concept_dim > self.initial_hidden_dim:
            logger.warning("concept_dim should not be larger than initial_hidden_dim")
            self.concept_dim = self.initial_hidden_dim

        if self.thought_dim > self.initial_hidden_dim * 2:
            logger.warning("thought_dim too large, reducing to 2x initial_hidden_dim")
            self.thought_dim = self.initial_hidden_dim * 2

        # Check growth parameters
        if self.growth_factor <= 1.0:
            logger.warning("growth_factor must be greater than 1.0, setting to default 1.2")
            self.growth_factor = 1.2

        if self.max_growth_steps < 100:
            logger.warning("max_growth_steps too small, setting to minimum 100")
            self.max_growth_steps = 100

        # Check limit values
        if self.max_hidden_dim < self.initial_hidden_dim:
            logger.warning("max_hidden_dim cannot be smaller than initial_hidden_dim")
            self.max_hidden_dim = self.initial_hidden_dim * 2

        if self.max_num_layers < self.initial_num_layers:
            logger.warning("max_num_layers cannot be smaller than initial_num_layers")
            self.max_num_layers = self.initial_num_layers * 2

        # Check memory parameters
        if self.concept_memory_size < 1000:
            logger.warning("concept_memory_size too small, setting to minimum 1000")
            self.concept_memory_size = 1000

        if self.pattern_memory_capacity < 1000:
            logger.warning("pattern_memory_capacity too small, setting to minimum 1000")
            self.pattern_memory_capacity = 1000

        # Check learning parameters
        if self.learning_rate > 0.1:
            logger.warning("learning_rate too high, capping at 0.1")
            self.learning_rate = 0.1

        if self.warmup_steps < 100:
            logger.warning("warmup_steps too small, setting to minimum 100")
            self.warmup_steps = 100

        # Check device configuration
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

        # Validate paths
        for path_attr in ['save_dir', 'experiences_path', 'concepts_path', 'growth_log_path']:
            path = getattr(self, path_attr)
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
            except Exception as e:
                logger.error(f"Error creating directory for {path_attr}: {e}")
                logger.warning(f"Using default path for {path_attr}")
                setattr(self, path_attr, os.path.join("./data", os.path.basename(path)))

        # Validate hive mind configuration
        if self.hive_enabled:
            if not self.hive_server_url and not self.hive_server_mode:
                logger.warning("Hive enabled but no server URL provided")
                self.hive_enabled = False
            if not self.hive_identity:
                self.hive_identity = str(uuid.uuid4())
                logger.info(f"Generated hive identity: {self.hive_identity}")

        # Return validated config
        return self

    @classmethod
    def load(cls, path):
        """Load configuration from JSON file"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


    def save(self, path):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            # Use the custom encoder
            json.dump(asdict(self), f, indent=2, cls=CustomJSONEncoder)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.dtype):
            return str(obj)
        return super().default(obj)


###########################################
# UNIFIED NEURAL-LINGUISTIC SYSTEM
###########################################

class DirectSignalProcessor(nn.Module):
    """Processes raw input signals without separate tokenization steps"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.initial_hidden_dim

        # Direct signal processing for various modalities
        self.signal_processors = nn.ModuleDict({
            "text": nn.Sequential(
                nn.Conv1d(1, self.hidden_dim // 2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(self.hidden_dim // 2, self.hidden_dim, kernel_size=5, padding=2)
            ),
            "image": nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.AdaptiveAvgPool2d((16, 16)),
                nn.Flatten(),
                nn.Linear(32 * 16 * 16, self.hidden_dim)
            ),
            "audio": nn.Sequential(
                nn.Conv1d(1, self.hidden_dim // 2, kernel_size=7, padding=3),
                nn.GELU(),
                nn.Conv1d(self.hidden_dim // 2, self.hidden_dim, kernel_size=7, padding=3)
            )
        })

        # Cross-modal integration
        self.modality_integration = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Emergent pattern detector
        self.pattern_detector = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(self.hidden_dim, 1, kernel_size=1)
        )

        # Position encoding
        self.position_encodings = nn.Parameter(torch.randn(1, 1000, self.hidden_dim))

        # Stores emerging patterns
        self.emerging_patterns = []
        self.pattern_strengths = []

        # Pattern tracking
        self.pattern_memory = {}
        self.pattern_frequency = Counter()
        self.pattern_timestamps = {}

    def forward(self, raw_signal, modality="text"):
        """Process raw signals directly without tokenization"""
        batch_size = raw_signal.size(0)

        # Process raw signal through appropriate processor
        if modality == "text":
            # For text, process raw ASCII/UTF-8 values
            x = raw_signal.unsqueeze(1)  # [batch, 1, seq_len]
            processed = self.signal_processors[modality](x)
            processed = processed.transpose(1, 2)  # [batch, seq_len, hidden_dim]

            # Get sequence length for position encoding
            seq_len = processed.size(1)
            position_enc = self.position_encodings[:, :seq_len, :]

            # Add position encoding
            processed = processed + position_enc

        elif modality == "image":
            # For images, process raw pixel values
            processed = self.signal_processors[modality](raw_signal)
            # Add sequence dimension if needed
            if len(processed.shape) == 2:
                processed = processed.unsqueeze(1)

        elif modality == "audio":
            # For audio, process raw waveform
            x = raw_signal.unsqueeze(1)  # [batch, 1, seq_len]
            processed = self.signal_processors[modality](x)
            processed = processed.transpose(1, 2)  # [batch, seq_len, hidden_dim]

            # Get sequence length for position encoding
            seq_len = processed.size(1)
            position_enc = self.position_encodings[:, :seq_len, :]

            # Add position encoding
            processed = processed + position_enc

        # Detect and track emergent patterns
        if self.training:
            # Convert to format for pattern detection
            pattern_input = processed.transpose(1, 2)  # [batch, hidden_dim, seq_len]
            pattern_scores = self.pattern_detector(pattern_input).squeeze(1)  # [batch, seq_len]

            # Find regions with high pattern scores
            pattern_threshold = 0.7
            for b in range(pattern_scores.size(0)):
                high_scores = (pattern_scores[b] > pattern_threshold).nonzero(as_tuple=True)[0]

                # Extract patterns from high-scoring regions
                for idx in high_scores:
                    start = max(0, idx - 3)
                    end = min(processed.size(1), idx + 4)

                    # Create a pattern key based on the activation pattern
                    pattern_tensor = processed[b, start:end].detach().cpu()
                    pattern_key = str(pattern_tensor.mean(dim=0).numpy().round(3))

                    # Update pattern memory
                    if pattern_key in self.pattern_memory:
                        # Update existing pattern
                        self.pattern_frequency[pattern_key] += 1
                        self.pattern_timestamps[pattern_key] = time.time()
                    else:
                        # New pattern
                        self.pattern_memory[pattern_key] = pattern_tensor
                        self.pattern_frequency[pattern_key] = 1
                        self.pattern_timestamps[pattern_key] = time.time()

                    # Prune pattern memory if too large
                    if len(self.pattern_memory) > 1000:
                        # Remove least frequent pattern
                        least_common = self.pattern_frequency.most_common()[:-1]
                        if least_common:
                            key_to_remove = least_common[-1][0]
                            del self.pattern_memory[key_to_remove]
                            del self.pattern_frequency[key_to_remove]
                            del self.pattern_timestamps[key_to_remove]

        # Apply modality-specific integration
        processed = self.modality_integration(processed)

        return processed

    def get_frequent_patterns(self, limit=10):
        """Get most frequently observed patterns"""
        most_common = self.pattern_frequency.most_common(limit)
        return [(key, self.pattern_memory[key], freq) for key, freq in most_common]

    def grow(self, new_hidden_dim):
        """Grow the processor to a new hidden dimension"""
        if new_hidden_dim <= self.hidden_dim:
            return False

        # Create new signal processors with larger dimensions
        new_signal_processors = nn.ModuleDict()

        # Grow text processor
        new_signal_processors["text"] = nn.Sequential(
            nn.Conv1d(1, new_hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(new_hidden_dim // 2, new_hidden_dim, kernel_size=5, padding=2)
        ).to(self.signal_processors["text"][0].weight.device)

        # Grow image processor - need to maintain existing architecture with new output dim
        new_signal_processors["image"] = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, new_hidden_dim)
        ).to(self.signal_processors["image"][0].weight.device)

        # Grow audio processor
        new_signal_processors["audio"] = nn.Sequential(
            nn.Conv1d(1, new_hidden_dim // 2, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(new_hidden_dim // 2, new_hidden_dim, kernel_size=7, padding=3)
        ).to(self.signal_processors["audio"][0].weight.device)

        # Replace processors
        self.signal_processors = new_signal_processors

        # Create new integration layer
        new_integration = nn.Linear(new_hidden_dim, new_hidden_dim).to(self.modality_integration.weight.device)
        self.modality_integration = new_integration

        # Create new pattern detector
        self.pattern_detector = nn.Sequential(
            nn.Conv1d(new_hidden_dim, new_hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(new_hidden_dim, 1, kernel_size=1)
        ).to(self.pattern_detector[0].weight.device)

        # Create new position encodings
        new_position_encodings = nn.Parameter(
            torch.randn(1, 1000, new_hidden_dim)
        ).to(self.position_encodings.device)

        # Copy old position encodings to preserve learned patterns
        old_dim = self.hidden_dim
        new_position_encodings[:, :, :old_dim] = self.position_encodings
        self.position_encodings = new_position_encodings

        # Update hidden dimension
        self.hidden_dim = new_hidden_dim

        return True


class EmergentConceptualSystem(nn.Module):
    """Allows concepts to emerge organically without predefined structures"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.initial_hidden_dim

        # Dynamic concept space (starts empty)
        self.register_buffer("concept_prototypes", torch.zeros(10, self.hidden_dim))
        self.register_buffer("concept_counts", torch.zeros(10, dtype=torch.long))
        self.register_buffer("concept_timestamps", torch.zeros(10))

        # Concept relationships (emerges through experience)
        self.register_buffer("concept_similarities", torch.zeros(10, 10))

        # Activation clustering for emergent concepts
        self.cluster_threshold = config.concept_cluster_threshold
        self.creation_threshold = config.concept_creation_threshold

        # Tracks total concepts discovered
        self.next_concept_id = 0

        # Experience buffer
        self.experience_buffer = []

        # Concept metadata
        self.concept_metadata = {}

        # Source mapping (derived rather than predefined)
        self.source_to_concept = {}

        # Modality tracking
        self.modality_concepts = defaultdict(set)

    def forward(self, activations, modality="text"):
        """Process activations through emergent conceptual system"""
        batch_size, seq_len, _ = activations.shape

        # Find similar concepts for each activation
        # This replaces the traditional token lookup

        # Reshape for comparison with all concepts
        flat_activations = activations.reshape(-1, self.hidden_dim)  # [batch*seq, hidden]
        flat_results = torch.zeros_like(flat_activations)

        # If we have concepts, find similarities
        if self.next_concept_id > 0:
            # Get active concepts
            concepts = self.concept_prototypes[:self.next_concept_id]

            # Calculate similarities
            similarities = F.cosine_similarity(
                flat_activations.unsqueeze(1),  # [batch*seq, 1, hidden]
                concepts.unsqueeze(0),          # [1, num_concepts, hidden]
                dim=2
            )  # [batch*seq, num_concepts]

            # Find maximum similarity for each activation
            max_sim, max_idx = similarities.max(dim=1)

            # Update matched concepts
            for i, (sim, idx) in enumerate(zip(max_sim, max_idx)):
                if sim > self.cluster_threshold:
                    # Strong match - update existing concept
                    with torch.no_grad():
                        # Blend with existing concept (weighted update)
                        alpha = 0.1  # Learning rate
                        current = self.concept_prototypes[idx]
                        updated = (1 - alpha) * current + alpha * flat_activations[i]
                        self.concept_prototypes[idx] = F.normalize(updated, dim=0)

                        # Update statistics
                        self.concept_counts[idx] += 1
                        self.concept_timestamps[idx] = time.time()

                        # Update modality tracking
                        self.modality_concepts[modality].add(idx.item())

                        # Record the concept influence on this activation
                        flat_results[i] = current.clone()
                elif sim > self.creation_threshold:
                    # Moderate similarity - potential new concept
                    concept_id = self._consider_new_concept(flat_activations[i], modality)
                    if concept_id is not None:
                        # Use the new concept's prototype
                        flat_results[i] = self.concept_prototypes[concept_id].clone()
                else:
                    # Low similarity - pass through unchanged
                    flat_results[i] = flat_activations[i]
        else:
            # No concepts yet - consider each activation as potential new concept
            for i, activation in enumerate(flat_activations):
                concept_id = self._consider_new_concept(activation, modality)
                if concept_id is not None:
                    # Use the new concept's prototype
                    flat_results[i] = self.concept_prototypes[concept_id].clone()
                else:
                    # No concept created - pass through unchanged
                    flat_results[i] = activation

        # Reshape results back to original dimensions
        results = flat_results.reshape(batch_size, seq_len, self.hidden_dim)

        # Blend original activations with conceptual results
        blend_factor = 0.3
        blended = (1 - blend_factor) * activations + blend_factor * results

        return blended

    def _consider_new_concept(self, activation, modality="text"):
        """Consider adding a new concept based on activation"""
        # Only add if we have space and meets criteria
        if random.random() < 0.1:  # Only consider 10% of candidates to prevent explosion
            if self.next_concept_id < self.concept_prototypes.size(0):
                # Add new concept
                self.concept_prototypes[self.next_concept_id] = F.normalize(activation, dim=0)
                self.concept_counts[self.next_concept_id] = 1
                self.concept_timestamps[self.next_concept_id] = time.time()

                # Create metadata
                concept_id = self.next_concept_id
                self.concept_metadata[concept_id] = {
                    "created_at": time.time(),
                    "type": "emergent",
                    "modality": modality,
                    "frequency": 1
                }

                # Update modality tracking
                self.modality_concepts[modality].add(concept_id)

                # Update concept relationships
                if self.next_concept_id > 0:
                    for i in range(self.next_concept_id):
                        sim = F.cosine_similarity(
                            self.concept_prototypes[i].unsqueeze(0),
                            activation.unsqueeze(0),
                            dim=1
                        ).item()
                        self.concept_similarities[i, self.next_concept_id] = sim
                        self.concept_similarities[self.next_concept_id, i] = sim

                # Increment counter
                self.next_concept_id += 1

                return concept_id
            else:
                # Grow concept space if needed
                self._grow_concept_space()

                # Try again
                return self._consider_new_concept(activation, modality)

        return None

    def _grow_concept_space(self):
        """Dynamically expand concept space"""
        current_size = self.concept_prototypes.size(0)
        new_size = current_size * 2

        # Create new buffers
        device = self.concept_prototypes.device
        new_prototypes = torch.zeros(new_size, self.hidden_dim, device=device)
        new_counts = torch.zeros(new_size, dtype=torch.long, device=device)
        new_timestamps = torch.zeros(new_size, device=device)
        new_similarities = torch.zeros(new_size, new_size, device=device)

        # Copy existing data
        new_prototypes[:current_size] = self.concept_prototypes
        new_counts[:current_size] = self.concept_counts
        new_timestamps[:current_size] = self.concept_timestamps
        new_similarities[:current_size, :current_size] = self.concept_similarities

        # Register new buffers
        self.register_buffer("concept_prototypes", new_prototypes)
        self.register_buffer("concept_counts", new_counts)
        self.register_buffer("concept_timestamps", new_timestamps)
        self.register_buffer("concept_similarities", new_similarities)

        logger.info(f"Grew concept space from {current_size} to {new_size}")

    def create_merged_concept(self, concept_id1, concept_id2, modality="text"):
        """Create a new concept by merging two existing concepts"""
        if concept_id1 >= self.next_concept_id or concept_id2 >= self.next_concept_id:
            return None

        # Create merged concept vector
        concept1 = self.concept_prototypes[concept_id1]
        concept2 = self.concept_prototypes[concept_id2]
        merged_vector = (concept1 + concept2) / 2

        # Create the new concept
        if self.next_concept_id < self.concept_prototypes.size(0):
            # Add new concept
            self.concept_prototypes[self.next_concept_id] = F.normalize(merged_vector, dim=0)
            self.concept_counts[self.next_concept_id] = 1
            self.concept_timestamps[self.next_concept_id] = time.time()

            # Create metadata
            concept_id = self.next_concept_id
            self.concept_metadata[concept_id] = {
                "created_at": time.time(),
                "type": "merged",
                "parent_concepts": [concept_id1, concept_id2],
                "modality": modality,
                "frequency": 1
            }

            # Update modality tracking
            self.modality_concepts[modality].add(concept_id)

            # Update concept relationships
            for i in range(self.next_concept_id):
                sim = F.cosine_similarity(
                    self.concept_prototypes[i].unsqueeze(0),
                    merged_vector.unsqueeze(0),
                    dim=1
                ).item()
                self.concept_similarities[i, self.next_concept_id] = sim
                self.concept_similarities[self.next_concept_id, i] = sim

            # Increment counter
            self.next_concept_id += 1

            return concept_id
        else:
            # Grow concept space if needed
            self._grow_concept_space()

            # Try again
            return self.create_merged_concept(concept_id1, concept_id2, modality)

    def find_similar_concepts(self, query_vector, top_k=5, modality=None):
        """Find concepts most similar to a query vector"""
        if self.next_concept_id == 0:
            return []

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
            filtered_vectors = self.concept_prototypes[concept_filter]
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
                self.concept_prototypes[:self.next_concept_id],
                dim=1
            )
            values, indices = torch.topk(similarities, min(top_k, len(similarities)))
            return [(idx.item(), val.item()) for idx, val in zip(indices, values)]

    def update_concept_usage(self, concept_id):
        """Update usage statistics for a concept"""
        if concept_id < self.next_concept_id:
            with torch.no_grad():
                self.concept_counts[concept_id] += 1
                self.concept_timestamps[concept_id] = time.time()

                # Update metadata
                if concept_id in self.concept_metadata:
                    self.concept_metadata[concept_id]["frequency"] = self.concept_counts[concept_id].item()

    def get_concept_stats(self):
        """Get statistics about concepts in memory"""
        total_concepts = self.next_concept_id
        used_concepts = sum(1 for i in range(self.next_concept_id) if self.concept_counts[i] > 0)

        # Calculate concepts by type
        type_counts = Counter()
        for cid in range(self.next_concept_id):
            if cid in self.concept_metadata:
                concept_type = self.concept_metadata[cid].get("type", "unknown")
                type_counts[concept_type] += 1

        # Calculate concepts by modality
        modality_stats = {
            modality: len(concepts)
            for modality, concepts in self.modality_concepts.items()
        }

        # Top concepts by usage
        top_concepts = []
        if self.next_concept_id > 0:
            counts = self.concept_counts[:self.next_concept_id].cpu().numpy()
            top_idxs = np.argsort(counts)[-10:][::-1]  # Top 10 by count
            for idx in top_idxs:
                if counts[idx] > 0:
                    top_concepts.append((int(idx), None, int(counts[idx])))

        return {
            "total_concepts": total_concepts,
            "used_concepts": used_concepts.item() if hasattr(used_concepts, 'item') else used_concepts,
            "modality_stats": modality_stats,
            "type_stats": dict(type_counts),
            "top_concepts": top_concepts
        }

    def grow(self, new_hidden_dim):
        """Grow to accommodate a new hidden dimension"""
        if new_hidden_dim <= self.hidden_dim:
            return False

        # Create new buffers
        device = self.concept_prototypes.device
        current_size = self.concept_prototypes.size(0)

        new_prototypes = torch.zeros(current_size, new_hidden_dim, device=device)

        # Copy existing data with padding
        new_prototypes[:, :self.hidden_dim] = self.concept_prototypes

        # Register new buffer
        self.register_buffer("concept_prototypes", new_prototypes)

        # Update dimension
        self.hidden_dim = new_hidden_dim

        logger.info(f"Grew concept system to {new_hidden_dim} dimensions")
        return True

class UnifiedPerceptionCognitionLayer(nn.Module):
    """Layer that unifies perception and cognition without separation"""

    def __init__(self, hidden_dim, growth_factor=1.4, layer_id=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.growth_factor = growth_factor
        self.layer_id = layer_id

        # Direct unified processing - no separation between percept and concept
        self.unified_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Self-attention mechanism
        self.self_attention = AdaptiveAttention(hidden_dim)

        # Normalization layers
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Recursive connections for emergent symbol grounding
        self.recursive_connections = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

        # Dynamic pathway formation
        self.emergent_pathways = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(4)
        ])

        # Modality-specific processing
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

        # Track activation patterns to detect emergent symbols
        self.register_buffer("activation_history", torch.zeros(100, hidden_dim))
        self.register_buffer("activation_index", torch.tensor(0, dtype=torch.long))

        # Usage tracking for evolution
        self.register_buffer("neuron_activations", torch.zeros(hidden_dim))
        self.updates = 0

    def forward(self, x, mask=None, modality="text"):
        # Track neuron activations for evolution if training
        if self.training:
            with torch.no_grad():
                # Update activation statistics
                activations = torch.abs(x).mean(dim=[0, 1])  # Mean across batch and sequence
                self.neuron_activations += activations
                self.updates += 1

        # Process through unified pathways
        residual = x

        # Apply first normalization
        x = self.norm1(x)

        # Apply self-attention
        attn_output = self.self_attention(x, mask)

        # Add residual connection
        x = residual + attn_output
        residual = x

        # Apply second normalization
        x = self.norm2(x)

        # Apply unified processor
        processed = self.unified_processor(x)

        # Apply recursive processing for emergent symbol grounding
        batch_size, seq_len, _ = x.shape
        recursive_input = x.reshape(batch_size * seq_len, -1)
        recursive_output = torch.matmul(recursive_input, self.recursive_connections)
        recursive_output = recursive_output.reshape(batch_size, seq_len, -1)

        # Apply modality-specific processing if not text
        if modality != "text" and modality in self.modality_adapters:
            modality_output = self.modality_adapters[modality](processed)
            # Blend with base output (weighted by layer depth - deeper layers use more modality-specific)
            blend_factor = min(0.8, 0.2 + 0.1 * self.layer_id)  # 0.2 to 0.8 based on layer depth
            processed = (1 - blend_factor) * processed + blend_factor * modality_output

        # Combine with emergent pathways
        pathway_outputs = []
        for pathway in self.emergent_pathways:
            pathway_outputs.append(pathway(x))

        # Dynamically weight pathway contributions based on activation patterns
        pathway_weights = F.softmax(torch.randn(len(self.emergent_pathways), device=x.device), dim=0)
        emergent_contribution = sum(w * p for w, p in zip(pathway_weights, pathway_outputs))

        # Unified output combines all processing with residual connection
        output = residual + processed + 0.1 * recursive_output + 0.1 * emergent_contribution

        # Track emergent patterns
        if self.training:
            with torch.no_grad():
                idx = self.activation_index % 100
                self.activation_history[idx] = output.mean(dim=(0, 1)).detach()
                self.activation_index += 1

        return output

    def grow(self, new_hidden_dim):
        """Grow layer to a new hidden dimension"""
        if new_hidden_dim <= self.hidden_dim:
            return False

        old_dim = self.hidden_dim

        # Grow self-attention
        self.self_attention.grow(new_hidden_dim)

        # Create new unified processor
        new_unified_processor = nn.Sequential(
            nn.Linear(new_hidden_dim, new_hidden_dim * 2),
            nn.GELU(),
            nn.Linear(new_hidden_dim * 2, new_hidden_dim)
        ).to(self.unified_processor[0].weight.device)

        # Transfer weights for unified processor
        with torch.no_grad():
            # First layer
            new_unified_processor[0].weight[:old_dim*2, :old_dim].copy_(
                self.unified_processor[0].weight
            )
            new_unified_processor[0].bias[:old_dim*2].copy_(
                self.unified_processor[0].bias
            )

            # Second layer (after activation)
            new_unified_processor[2].weight[:new_hidden_dim, :old_dim*2].copy_(
                self.unified_processor[2].weight[:, :old_dim*2]
            )
            new_unified_processor[2].bias[:new_hidden_dim].copy_(
                self.unified_processor[2].bias
            )

        # Replace unified processor
        self.unified_processor = new_unified_processor

        # Create new normalization layers
        new_norm1 = nn.LayerNorm(new_hidden_dim).to(self.norm1.weight.device)
        new_norm2 = nn.LayerNorm(new_hidden_dim).to(self.norm2.weight.device)

        # Transfer weights for normalization layers
        with torch.no_grad():
            new_norm1.weight[:old_dim].copy_(self.norm1.weight)
            new_norm1.bias[:old_dim].copy_(self.norm1.bias)

            new_norm2.weight[:old_dim].copy_(self.norm2.weight)
            new_norm2.bias[:old_dim].copy_(self.norm2.bias)

            # Initialize new dimensions
            new_norm1.weight[old_dim:].fill_(1.0)
            new_norm1.bias[old_dim:].zero_()

            new_norm2.weight[old_dim:].fill_(1.0)
            new_norm2.bias[old_dim:].zero_()

        # Replace normalization layers
        self.norm1 = new_norm1
        self.norm2 = new_norm2

        # Create new recursive connections
        new_recursive_connections = nn.Parameter(
            torch.randn(new_hidden_dim, new_hidden_dim)
        ).to(self.recursive_connections.device)

        # Transfer weights for recursive connections
        with torch.no_grad():
            new_recursive_connections[:old_dim, :old_dim].copy_(
                self.recursive_connections
            )

        # Replace recursive connections
        self.recursive_connections = new_recursive_connections

        # Create new emergent pathways
        new_emergent_pathways = nn.ModuleList()
        for old_pathway in self.emergent_pathways:
            new_pathway = nn.Linear(new_hidden_dim, new_hidden_dim).to(old_pathway.weight.device)

            # Transfer weights
            with torch.no_grad():
                new_pathway.weight[:old_dim, :old_dim].copy_(old_pathway.weight)
                new_pathway.bias[:old_dim].copy_(old_pathway.bias)

                # Initialize new dimensions
                std = 0.02
                new_pathway.weight[old_dim:, :old_dim].normal_(mean=0.0, std=std)
                new_pathway.weight[:, old_dim:].normal_(mean=0.0, std=std)
                new_pathway.bias[old_dim:].zero_()

            new_emergent_pathways.append(new_pathway)

        # Replace emergent pathways
        self.emergent_pathways = new_emergent_pathways

        # Create new modality adapters
        new_modality_adapters = nn.ModuleDict()
        for modality, adapter in self.modality_adapters.items():
            if modality == "text":
                new_modality_adapters[modality] = nn.Identity()
            else:
                new_adapter = nn.Sequential(
                    nn.Linear(new_hidden_dim, new_hidden_dim // 4),
                    nn.GELU(),
                    nn.Linear(new_hidden_dim // 4, new_hidden_dim)
                ).to(self.unified_processor[0].weight.device)
                new_modality_adapters[modality] = new_adapter

        # Replace modality adapters
        self.modality_adapters = new_modality_adapters

        # Create new activation history buffer
        device = self.activation_history.device
        new_activation_history = torch.zeros(100, new_hidden_dim, device=device)
        new_activation_history[:, :old_dim] = self.activation_history
        self.register_buffer("activation_history", new_activation_history)

        # Create new neuron activations buffer
        new_neuron_activations = torch.zeros(new_hidden_dim, device=device)
        new_neuron_activations[:old_dim] = self.neuron_activations
        self.register_buffer("neuron_activations", new_neuron_activations)

        # Update hidden dimension
        self.hidden_dim = new_hidden_dim

        return True

    def evolve(self):
        """Evolve layer based on usage patterns"""
        if self.updates < 10:
            return False

        # Calculate neuron importance
        with torch.no_grad():
            if self.updates > 0:
                # Normalize by number of updates
                neuron_activity = self.neuron_activations / self.updates

                # Calculate importance based on activity
                mean_activity = torch.mean(neuron_activity)
                neuron_importance = neuron_activity / mean_activity

                # Identify weak and strong neurons
                weak_threshold = 0.3
                strong_threshold = 1.5

                weak_neurons = (neuron_importance < weak_threshold).nonzero(as_tuple=True)[0]
                strong_neurons = (neuron_importance > strong_threshold).nonzero(as_tuple=True)[0]

                # Adjust recursive connections to strengthen important pathways
                if len(strong_neurons) > 0:
                    for src_idx in strong_neurons:
                        # Strengthen connections from important neurons
                        self.recursive_connections[src_idx, :] *= 1.1

                # Reset statistics for next evolution cycle
                self.neuron_activations.zero_()
                self.updates = 0

                return {
                    "layer_id": self.layer_id,
                    "neuron_importance": neuron_importance.tolist(),
                    "mean_importance": float(torch.mean(neuron_importance).item()),
                    "max_importance": float(torch.max(neuron_importance).item()),
                    "min_importance": float(torch.min(neuron_importance).item()),
                    "strong_neurons": len(strong_neurons),
                    "weak_neurons": len(weak_neurons)
                }

        return {}


class ParametricActivation(nn.Module):
    """Learnable activation function with adaptive parameters"""

    def __init__(self, hidden_dim, init_fn='silu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.init_fn = init_fn

        # Parametric coefficients per neuron
        self.register_parameter("alpha", nn.Parameter(torch.ones(hidden_dim)))
        self.register_parameter("beta", nn.Parameter(torch.zeros(hidden_dim)))
        self.register_parameter("gamma", nn.Parameter(torch.ones(hidden_dim)))

        # Initialize parameters based on activation function
        self._initialize_parameters()

        # Activation usage tracking
        self.register_buffer("usage_counts", torch.zeros(hidden_dim))
        self.register_buffer("activation_means", torch.zeros(hidden_dim))
        self.register_buffer("activation_vars", torch.ones(hidden_dim))

    def _initialize_parameters(self):
        """Initialize parameters based on the selected activation function"""
        if self.init_fn == 'silu':
            # SiLU/Swish initialization
            with torch.no_grad():
                self.alpha.fill_(1.0)
                self.beta.fill_(0.0)
                self.gamma.fill_(1.0)
        elif self.init_fn == 'gelu':
            # GELU-like initialization
            with torch.no_grad():
                self.alpha.fill_(1.0)
                self.beta.fill_(0.0)
                self.gamma.fill_(1.702)  # Approximate GELU with this gamma value
        elif self.init_fn == 'relu':
            # ReLU-like initialization
            with torch.no_grad():
                self.alpha.fill_(1.0)
                self.beta.fill_(0.0)
                self.gamma.fill_(10.0)  # High gamma approximates ReLU
        else:
            # Default to SiLU/Swish
            with torch.no_grad():
                self.alpha.fill_(1.0)
                self.beta.fill_(0.0)
                self.gamma.fill_(1.0)

    def forward(self, x):
        """Apply parametric activation function"""
        # Extract last dimension for per-neuron parameters
        if x.dim() > 1 and x.size(-1) == self.hidden_dim:
            # Apply the parametric activation (generalized Swish/SiLU)
            return self.alpha * x * torch.sigmoid(self.gamma * x + self.beta)
        else:
            # Fallback for dimension mismatch - use standard SiLU
            return F.silu(x)

    def grow(self, new_dim):
        """Grow activation function to a new dimension"""
        if new_dim <= self.hidden_dim:
            return self

        device = self.alpha.device
        new_activation = ParametricActivation(new_dim, self.init_fn)
        new_activation.to(device)

        # Copy existing parameters
        with torch.no_grad():
            new_activation.alpha.data[:self.hidden_dim] = self.alpha.data
            new_activation.beta.data[:self.hidden_dim] = self.beta.data
            new_activation.gamma.data[:self.hidden_dim] = self.gamma.data

            # Initialize new parameters based on the mean of existing ones
            alpha_mean = self.alpha.data.mean().item()
            beta_mean = self.beta.data.mean().item()
            gamma_mean = self.gamma.data.mean().item()

            new_activation.alpha.data[self.hidden_dim:] = alpha_mean + torch.randn(new_dim - self.hidden_dim, device=device) * 0.01
            new_activation.beta.data[self.hidden_dim:] = beta_mean + torch.randn(new_dim - self.hidden_dim, device=device) * 0.01
            new_activation.gamma.data[self.hidden_dim:] = gamma_mean + torch.randn(new_dim - self.hidden_dim, device=device) * 0.01

            # Copy usage statistics
            new_activation.usage_counts.data[:self.hidden_dim] = self.usage_counts.data
            new_activation.activation_means.data[:self.hidden_dim] = self.activation_means.data
            new_activation.activation_vars.data[:self.hidden_dim] = self.activation_vars.data

        return new_activation

    def evolve(self, learning_rate=0.010):
        """Evolve activation parameters based on usage and activation statistics"""
        with torch.no_grad():
            # Only evolve parameters that have been used
            usage_mask = (self.usage_counts > 0).float()

            # Calculate adaptability for each parameter
            adaptability = 1.0 / (self.usage_counts + 1.0)
            adaptability = adaptability * usage_mask

            # Add small random adjustments proportional to adaptability
            self.alpha.data += torch.randn_like(self.alpha) * learning_rate * adaptability
            self.beta.data += torch.randn_like(self.beta) * learning_rate * adaptability
            self.gamma.data += torch.randn_like(self.gamma) * learning_rate * adaptability

            # Ensure parameters stay within reasonable bounds
            self.alpha.data.clamp_(0.5, 2.0)
            self.beta.data.clamp_(-2.0, 2.0)
            self.gamma.data.clamp_(0.5, 10.0)

        return {"evolved": True, "param_count": self.hidden_dim}

class NeuroplasticLayer(nn.Module):
    """Core neural layer that can grow and evolve with neuroplasticity in all directions"""

    def __init__(self, hidden_dim, growth_factor=1.2, dropout=0.1, layer_id=0,
                 num_heads=8, expansion_factor=4, backward_connections=None,
                 lateral_connections=None, device="cuda"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.growth_factor = growth_factor
        self.layer_id = layer_id
        self.device = device
        self.expansion_factor = expansion_factor
        self.attention = AdaptiveAttention(hidden_dim, num_heads=num_heads, dropout=dropout)

        # Feed-forward network components
        self.gate_proj = nn.Linear(hidden_dim, expansion_factor * hidden_dim)
        self.up_proj = nn.Linear(hidden_dim, expansion_factor * hidden_dim)
        self.down_proj = nn.Linear(expansion_factor * hidden_dim, hidden_dim)

        # Normalization and regularization
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Activity tracking
        self.register_buffer("neuron_importance", torch.ones(hidden_dim, device=device))
        self.register_buffer("activation_history", torch.zeros(hidden_dim, device=device))
        self.register_buffer("activation_variance", torch.zeros(hidden_dim, device=device))
        self.register_buffer("gradient_history", torch.zeros(hidden_dim, device=device))

        # Plasticity mechanisms
        self.plasticity_factor = 0.3  # Controls how much weights can change during evolution
        self.pruning_threshold = 0.05  # Threshold for pruning inactive neurons
        self.updates = 0
        self.growth_events = []

        # Backward and lateral connections (for omnidirectional growth)
        self.backward_connections = backward_connections
        self.lateral_connections = lateral_connections or {}

        # Lateral connection weights
        self.lateral_weights = nn.ParameterDict()
        for lateral_id, lateral_dim in self.lateral_connections.items():
            self.lateral_weights[str(lateral_id)] = nn.Parameter(
                torch.zeros(hidden_dim, lateral_dim, device=device)
            )
            # Initialize with small random values
            nn.init.kaiming_normal_(self.lateral_weights[str(lateral_id)], nonlinearity='relu')
            # Scale down initially to avoid disrupting trained weights
            with torch.no_grad():
                self.lateral_weights[str(lateral_id)].mul_(0.01)

        # Backward connection projection
        if self.backward_connections:
            self.backward_proj = nn.Linear(hidden_dim, self.backward_connections.hidden_dim)
            # Initialize with small random values
            nn.init.kaiming_normal_(self.backward_proj.weight, nonlinearity='relu')
            with torch.no_grad():
                self.backward_proj.weight.mul_(0.01)

        # Activation functions with learnable parameters
        self.activation_fn = ParametricActivation(hidden_dim)

        # Growth state tracking
        self.growth_state = {
            "original_dim": hidden_dim,
            "growth_events": [],
            "neuron_ages": torch.zeros(hidden_dim, device=device),
            "max_activation": torch.zeros(hidden_dim, device=device),
            "min_activation": torch.zeros(hidden_dim, device=device)
        }

    def forward(self, x, mask=None, cross_input=None, modality="text",
                lateral_inputs=None, return_activations=False):
        """
        Forward pass with tracking for neuroplasticity and multi-directional connections

        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            mask: Attention mask
            cross_input: Optional cross-attention input
            modality: Input modality (text, image, etc.)
            lateral_inputs: Dict of {layer_id: tensor} for lateral connections
            return_activations: Whether to return intermediate activations
        """
        if self.training:
            self.updates += 1

        batch_size, seq_len, _ = x.shape
        residual = x

        # Pre-attention normalization
        x = self.norm1(x)

        # Self/cross attention
        attn_output = self.attention(x, mask=mask, cross_input=cross_input)

        # Apply lateral connections if provided
        if lateral_inputs and self.lateral_connections:
            lateral_contributions = []
            for layer_id, lateral_input in lateral_inputs.items():
                if str(layer_id) in self.lateral_weights:
                    # Project lateral input to this layer's dimension space
                    if lateral_input.shape[-1] != self.hidden_dim:
                        # Linear projection to match dimensions
                        weight = self.lateral_weights[str(layer_id)]
                        lateral_proj = torch.einsum('blf,fe->ble', lateral_input, weight)
                    else:
                        # Direct contribution with learned weight
                        lateral_proj = lateral_input * torch.sigmoid(self.lateral_weights[str(layer_id)])

                    lateral_contributions.append(lateral_proj)

            # Add weighted lateral contributions if any
            if lateral_contributions:
                lateral_sum = sum(lateral_contributions) / len(lateral_contributions)
                attn_output = attn_output + self.dropout(lateral_sum)

        # Add residual connection
        x = residual + self.dropout(attn_output)

        # Record attention activations for neuroplasticity
        if self.training:
            with torch.no_grad():
                attn_activations = attn_output.detach().abs().mean(dim=(0, 1))
                self._update_activation_stats(attn_activations)

        # Feed-forward network with residual connection
        residual = x
        x = self.norm2(x)

        # SwiGLU-like activation
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        intermediate = self.activation_fn(gate_output) * up_output

        # Track intermediate activations for neuroplasticity
        if self.training:
            with torch.no_grad():
                ffn_activations = intermediate.detach().abs().mean(dim=(0, 1))
                self._update_activation_stats(ffn_activations.mean(dim=0))

        ffn_output = self.down_proj(intermediate)
        x = residual + self.dropout(ffn_output)

        # Apply backward connections if available
        backward_signal = None
        if self.backward_connections is not None and random.random() < 0.5:  # Stochastic backward flow
            backward_signal = self.backward_proj(x.detach())

        # Return intermediate activations if requested
        if return_activations:
            return x, {
                "attn_output": attn_output,
                "ffn_output": ffn_output,
                "backward_signal": backward_signal
            }

        return x, backward_signal if self.backward_connections is not None else None

    def _update_activation_stats(self, activations):
        """Update neuron activation statistics for neuroplasticity"""
        # Exponential moving average of activations
        momentum = 0.9
        self.activation_history.mul_(momentum).add_(activations * (1 - momentum))

        # Update activation variance
        diff = activations - self.activation_history
        self.activation_variance.mul_(momentum).add_((diff * diff) * (1 - momentum))

        # Update importance based on activation and variance
        stability = torch.clamp(1.0 / (self.activation_variance + 1e-5), 0, 10)
        importance = self.activation_history * stability
        self.neuron_importance.mul_(0.95).add_(importance * 0.05)

        # Update growth state tracking
        self.growth_state["neuron_ages"] += 1
        self.growth_state["max_activation"] = torch.maximum(
            self.growth_state["max_activation"],
            activations
        )
        self.growth_state["min_activation"] = torch.minimum(
            self.growth_state["min_activation"],
            activations
        )

    def grow(self, new_dim):
        """
        Grow the layer to a new dimension size (width growth)

        Args:
            new_dim: New hidden dimension size
        """
        if new_dim <= self.hidden_dim:
            return False

        logger.info(f"Growing NeuroplasticLayer {self.layer_id} from {self.hidden_dim} to {new_dim}")

        growth_ratio = new_dim / self.hidden_dim
        growth_factor = new_dim - self.hidden_dim
        device = next(self.parameters()).device

        # Store original weights
        old_attn_weights = {
            'q_proj': self.attention.q_proj.weight.data.clone(),
            'k_proj': self.attention.k_proj.weight.data.clone(),
            'v_proj': self.attention.v_proj.weight.data.clone(),
            'o_proj': self.attention.o_proj.weight.data.clone(),
        }
        old_gate_proj = self.gate_proj.weight.data.clone()
        old_up_proj = self.up_proj.weight.data.clone()
        old_down_proj = self.down_proj.weight.data.clone()
        old_norm1_weight = self.norm1.weight.data.clone()
        old_norm1_bias = self.norm1.bias.data.clone()
        old_norm2_weight = self.norm2.weight.data.clone()
        old_norm2_bias = self.norm2.bias.data.clone()

        # Determine importance-based neuron selection for replication
        importance_scores = self.neuron_importance.clone()

        # Add noise to break ties
        importance_scores += torch.randn_like(importance_scores) * 0.01

        # Scale to [0, 1] range
        if importance_scores.max() > importance_scores.min():
            importance_scores = (importance_scores - importance_scores.min()) / (importance_scores.max() - importance_scores.min())

        # Select neurons to replicate with probability proportional to importance
        replication_probs = F.softmax(importance_scores * 2, dim=0)
        neuron_counts = torch.zeros(self.hidden_dim, dtype=torch.int)

        # Ensure each neuron is kept at least once
        neuron_counts += 1
        remaining = growth_factor - self.hidden_dim

        # Distribute remaining neurons based on importance
        if remaining > 0:
            # Multinomial sampling based on importance
            additional_counts = torch.multinomial(
                replication_probs,
                remaining,
                replacement=True
            )
            for idx in additional_counts:
                neuron_counts[idx] += 1

        # Create new modules with expanded dimensions
        self.attention = self.attention.grow(new_dim)
        expansion_hidden = new_dim * self.expansion_factor

        # Create new projections
        new_gate_proj = nn.Linear(new_dim, expansion_hidden, device=device)
        new_up_proj = nn.Linear(new_dim, expansion_hidden, device=device)
        new_down_proj = nn.Linear(expansion_hidden, new_dim, device=device)
        new_norm1 = nn.LayerNorm(new_dim, device=device)
        new_norm2 = nn.LayerNorm(new_dim, device=device)

        # Copy old weights with importance-based replication
        with torch.no_grad():
            # Helper function to copy weights with replication
            def copy_weights_with_replication(old_weights, new_weights, dim_idx=0):
                new_idx = 0
                for old_idx in range(self.hidden_dim):
                    count = neuron_counts[old_idx]
                    for _ in range(count):
                        if dim_idx == 0:  # Copy row
                            if new_idx < new_weights.size(0):
                                new_weights[new_idx] = old_weights[old_idx]
                        else:  # Copy column
                            if new_idx < new_weights.size(1):
                                new_weights[:, new_idx] = old_weights[:, old_idx]
                        new_idx += 1

                # Add noise to new connections
                if dim_idx == 0 and new_idx < new_weights.size(0):
                    fan_in = new_weights.size(1)
                    std = math.sqrt(2.0 / fan_in)
                    new_weights[new_idx:] = torch.randn_like(new_weights[new_idx:]) * std
                elif dim_idx == 1 and new_idx < new_weights.size(1):
                    fan_in = new_weights.size(0)
                    std = math.sqrt(2.0 / fan_in)
                    new_weights[:, new_idx:] = torch.randn_like(new_weights[:, new_idx:]) * std

            # Copy gate projection weights
            copy_weights_with_replication(old_gate_proj, new_gate_proj.weight.data, dim_idx=0)

            # For the output dimension
            new_idx = 0
            for old_idx in range(self.hidden_dim * self.expansion_factor):
                old_neuron = old_idx % self.hidden_dim
                old_channel = old_idx // self.hidden_dim
                count = neuron_counts[old_neuron]
                for _ in range(count):
                    new_channel = new_idx // new_dim
                    new_neuron = new_idx % new_dim
                    idx = new_channel * new_dim + new_neuron
                    if idx < expansion_hidden:
                        new_gate_proj.weight.data[idx // new_dim, idx % new_dim] = old_gate_proj[old_channel, old_neuron]
                    new_idx += 1

            # Copy up projection weights with similar approach
            copy_weights_with_replication(old_up_proj, new_up_proj.weight.data, dim_idx=0)
            new_idx = 0
            for old_idx in range(self.hidden_dim * self.expansion_factor):
                old_neuron = old_idx % self.hidden_dim
                old_channel = old_idx // self.hidden_dim
                count = neuron_counts[old_neuron]
                for _ in range(count):
                    new_channel = new_idx // new_dim
                    new_neuron = new_idx % new_dim
                    idx = new_channel * new_dim + new_neuron
                    if idx < expansion_hidden:
                        new_up_proj.weight.data[idx // new_dim, idx % new_dim] = old_up_proj[old_channel, old_neuron]
                    new_idx += 1

            # Copy down projection
            # For input dimension (columns)
            new_idx = 0
            for old_idx in range(self.hidden_dim * self.expansion_factor):
                old_neuron = old_idx % self.hidden_dim
                old_channel = old_idx // self.hidden_dim
                count = neuron_counts[old_neuron]
                for _ in range(count):
                    new_channel = new_idx // new_dim
                    new_neuron = new_idx % new_dim
                    idx = new_channel * new_dim + new_neuron
                    if idx < expansion_hidden:
                        new_down_proj.weight.data[:, idx] = old_down_proj[:, old_idx]
                    new_idx += 1

            # For output dimension (rows)
            copy_weights_with_replication(old_down_proj, new_down_proj.weight.data, dim_idx=0)

            # Copy norm weights with replication
            copy_weights_with_replication(old_norm1_weight, new_norm1.weight.data)
            copy_weights_with_replication(old_norm1_bias, new_norm1.bias.data)
            copy_weights_with_replication(old_norm2_weight, new_norm2.weight.data)
            copy_weights_with_replication(old_norm2_bias, new_norm2.bias.data)

        # Update modules
        self.gate_proj = new_gate_proj
        self.up_proj = new_up_proj
        self.down_proj = new_down_proj
        self.norm1 = new_norm1
        self.norm2 = new_norm2

        # Update lateral connections
        new_lateral_weights = nn.ParameterDict()
        for lateral_id, lateral_dim in self.lateral_connections.items():
            old_weights = self.lateral_weights[str(lateral_id)].data
            new_weights = torch.zeros(new_dim, lateral_dim, device=device)

            # Copy weights with neuron replication
            new_idx = 0
            for old_idx in range(self.hidden_dim):
                count = neuron_counts[old_idx]
                for _ in range(count):
                    if new_idx < new_dim:
                        new_weights[new_idx] = old_weights[old_idx]
                    new_idx += 1

            # Initialize new rows
            if new_idx < new_dim:
                new_weights[new_idx:] = torch.randn_like(new_weights[new_idx:]) * 0.01

            new_lateral_weights[str(lateral_id)] = nn.Parameter(new_weights)

        self.lateral_weights = new_lateral_weights

        # Update backward projection if exists
        if self.backward_connections:
            old_weights = self.backward_proj.weight.data
            new_backward_proj = nn.Linear(new_dim, self.backward_connections.hidden_dim, device=device)

            # Copy weights with neuron replication (for input dimension)
            new_idx = 0
            for old_idx in range(self.hidden_dim):
                count = neuron_counts[old_idx]
                for _ in range(count):
                    if new_idx < new_dim:
                        new_backward_proj.weight.data[:, new_idx] = old_weights[:, old_idx]
                    new_idx += 1

            # Initialize new columns
            if new_idx < new_dim:
                fan_in = new_backward_proj.weight.data.size(0)
                std = math.sqrt(2.0 / fan_in)
                new_backward_proj.weight.data[:, new_idx:] = torch.randn_like(new_backward_proj.weight.data[:, new_idx:]) * std

            # Copy bias
            new_backward_proj.bias.data = self.backward_proj.bias.data.clone()

            self.backward_proj = new_backward_proj

        # Update parametric activation
        self.activation_fn = self.activation_fn.grow(new_dim)

        # Update buffers
        old_dim = self.hidden_dim
        self.hidden_dim = new_dim

        new_importance = torch.ones(new_dim, device=device)
        new_activation_history = torch.zeros(new_dim, device=device)
        new_activation_variance = torch.zeros(new_dim, device=device)
        new_gradient_history = torch.zeros(new_dim, device=device)

        # Copy with replication
        new_idx = 0
        for old_idx in range(old_dim):
            count = neuron_counts[old_idx]
            for _ in range(count):
                if new_idx < new_dim:
                    new_importance[new_idx] = self.neuron_importance[old_idx]
                    new_activation_history[new_idx] = self.activation_history[old_idx]
                    new_activation_variance[new_idx] = self.activation_variance[old_idx]
                    new_gradient_history[new_idx] = self.gradient_history[old_idx]
                new_idx += 1

        # Register new buffers
        self.register_buffer("neuron_importance", new_importance)
        self.register_buffer("activation_history", new_activation_history)
        self.register_buffer("activation_variance", new_activation_variance)
        self.register_buffer("gradient_history", new_gradient_history)

        # Update growth state
        new_neuron_ages = torch.zeros(new_dim, device=device)
        new_max_activation = torch.zeros(new_dim, device=device)
        new_min_activation = torch.zeros(new_dim, device=device)

        # Copy with replication
        new_idx = 0
        for old_idx in range(old_dim):
            count = neuron_counts[old_idx]
            for _ in range(count):
                if new_idx < new_dim:
                    new_neuron_ages[new_idx] = self.growth_state["neuron_ages"][old_idx]
                    new_max_activation[new_idx] = self.growth_state["max_activation"][old_idx]
                    new_min_activation[new_idx] = self.growth_state["min_activation"][old_idx]
                new_idx += 1

        self.growth_state["neuron_ages"] = new_neuron_ages
        self.growth_state["max_activation"] = new_max_activation
        self.growth_state["min_activation"] = new_min_activation

        # Record growth event
        growth_event = {
            "time": time.time(),
            "old_dim": old_dim,
            "new_dim": new_dim,
            "growth_factor": growth_factor,
            "replication_map": neuron_counts.tolist(),
            "update_count": self.updates
        }

        self.growth_state["growth_events"].append(growth_event)
        self.growth_events.append(growth_event)

        return True

    def evolve(self, learning_rate=0.0010, prune_neurons=True):
        """
        Evolve the layer based on activation patterns, updating weights and structure

        Args:
            learning_rate: Rate of weight adaptation during evolution
            prune_neurons: Whether to prune inactive neurons

        Returns:
            Dict containing evolution statistics and changes
        """
        if self.updates < 10:
            return {"status": "skipped", "reason": "insufficient_updates"}

        # Analyze neuron importance
        importance = self.neuron_importance.clone()
        variance = self.activation_variance.clone()

        # Normalize importance scores
        if importance.max() > importance.min():
            normalized_importance = (importance - importance.min()) / (importance.max() - importance.min())
        else:
            normalized_importance = torch.ones_like(importance)

        # Compute stability based on inverse variance
        stability = 1.0 / (variance + 1e-5)
        stability = torch.clamp(stability, 0, 10)

        # Compute adaptability score (higher for neurons that should adapt more)
        adaptability = 1.0 - torch.clamp(normalized_importance, 0, 1) * stability
        adaptability = torch.clamp(adaptability, 0.01, 1.0)

        # Apply adaptive learning rates to weights
        with torch.no_grad():
            # Adapt attention weights based on neuron importance
            attn_scale = adaptability.unsqueeze(0)
            self.attention.q_proj.weight.mul_(1.0 - learning_rate * attn_scale).add_(
                torch.randn_like(self.attention.q_proj.weight) * learning_rate * attn_scale * 0.1
            )
            self.attention.k_proj.weight.mul_(1.0 - learning_rate * attn_scale).add_(
                torch.randn_like(self.attention.k_proj.weight) * learning_rate * attn_scale * 0.1
            )
            self.attention.v_proj.weight.mul_(1.0 - learning_rate * attn_scale).add_(
                torch.randn_like(self.attention.v_proj.weight) * learning_rate * attn_scale * 0.1
            )

            # Output projection evolves based on neuron importance
            o_proj_scale = adaptability.unsqueeze(1)
            self.attention.o_proj.weight.mul_(1.0 - learning_rate * o_proj_scale).add_(
                torch.randn_like(self.attention.o_proj.weight) * learning_rate * o_proj_scale * 0.1
            )

            # FFN weight adaptation
            ffn_input_scale = adaptability.unsqueeze(0)
            self.gate_proj.weight.mul_(1.0 - learning_rate * ffn_input_scale).add_(
                torch.randn_like(self.gate_proj.weight) * learning_rate * ffn_input_scale * 0.1
            )
            self.up_proj.weight.mul_(1.0 - learning_rate * ffn_input_scale).add_(
                torch.randn_like(self.up_proj.weight) * learning_rate * ffn_input_scale * 0.1
            )

            # Down projection evolves with more care
            down_scale = adaptability.unsqueeze(1)
            self.down_proj.weight.mul_(1.0 - learning_rate * down_scale * 0.5).add_(
                torch.randn_like(self.down_proj.weight) * learning_rate * down_scale * 0.05
            )

            # Evolve lateral connections if they exist
            for lateral_id in self.lateral_weights:
                lateral_scale = adaptability.unsqueeze(1)
                self.lateral_weights[lateral_id].mul_(1.0 - learning_rate * lateral_scale).add_(
                    torch.randn_like(self.lateral_weights[lateral_id]) * learning_rate * lateral_scale * 0.1
                )

            # Evolve backward projections if they exist
            if self.backward_connections:
                back_scale = adaptability.unsqueeze(1)
                self.backward_proj.weight.mul_(1.0 - learning_rate * back_scale).add_(
                    torch.randn_like(self.backward_proj.weight) * learning_rate * back_scale * 0.1
                )

        # Pruning inactive neurons if enabled
        pruned_neurons = []
        if prune_neurons:
            inactive_threshold = self.pruning_threshold
            inactive_mask = normalized_importance < inactive_threshold

            # Don't prune too many neurons at once (max 10%)
            max_to_prune = max(1, int(0.1 * self.hidden_dim))

            if inactive_mask.sum() > 0:
                inactive_indices = torch.where(inactive_mask)[0]

                # Limit number of neurons to prune
                prune_count = min(len(inactive_indices), max_to_prune)
                to_prune = inactive_indices[:prune_count]

                for idx in to_prune:
                    idx_item = idx.item()

                    # Record pruned neuron
                    pruned_neurons.append({
                        "index": idx_item,
                        "importance": normalized_importance[idx_item].item(),
                        "age": self.growth_state["neuron_ages"][idx_item].item()
                    })

                    # Reset the neuron (reinitialize weights)
                    with torch.no_grad():
                        # Generate new random weights for attention
                        self.attention.q_proj.weight[:, idx_item] = torch.randn_like(self.attention.q_proj.weight[:, idx_item]) * 0.02
                        self.attention.k_proj.weight[:, idx_item] = torch.randn_like(self.attention.k_proj.weight[:, idx_item]) * 0.02
                        self.attention.v_proj.weight[:, idx_item] = torch.randn_like(self.attention.v_proj.weight[:, idx_item]) * 0.02
                        self.attention.o_proj.weight[idx_item, :] = torch.randn_like(self.attention.o_proj.weight[idx_item, :]) * 0.02

                        # Reset FFN weights
                        self.gate_proj.weight[idx_item, :] = torch.randn_like(self.gate_proj.weight[idx_item, :]) * 0.02
                        self.up_proj.weight[idx_item, :] = torch.randn_like(self.up_proj.weight[idx_item, :]) * 0.02
                        self.down_proj.weight[:, idx_item] = torch.randn_like(self.down_proj.weight[:, idx_item]) * 0.02

                        # Reset lateral weights
                        for lateral_id in self.lateral_weights:
                            self.lateral_weights[lateral_id][idx_item, :] = torch.randn_like(self.lateral_weights[lateral_id][idx_item, :]) * 0.02

                        # Reset backward weights
                        if self.backward_connections:
                            self.backward_proj.weight[:, idx_item] = torch.randn_like(self.backward_proj.weight[:, idx_item]) * 0.02

                        # Reset activation tracking
                        self.neuron_importance[idx_item] = 1.0
                        self.activation_history[idx_item] = 0.0
                        self.activation_variance[idx_item] = 1.0
                        self.gradient_history[idx_item] = 0.0

                        # Reset growth state
                        self.growth_state["neuron_ages"][idx_item] = 0
                        self.growth_state["max_activation"][idx_item] = 0
                        self.growth_state["min_activation"][idx_item] = 0

        # Evolve the parametric activation function
        self.activation_fn.evolve(learning_rate)

        # Evolution statistics
        evolution_stats = {
            "status": "completed",
            "layer_id": self.layer_id,
            "updates": self.updates,
            "hidden_dim": self.hidden_dim,
            "mean_importance": normalized_importance.mean().item(),
            "max_importance": normalized_importance.max().item(),
            "min_importance": normalized_importance.min().item(),
            "pruned_neurons": pruned_neurons,
            "pruned_count": len(pruned_neurons),
            "timestamp": time.time()
        }

        return evolution_stats

    def add_lateral_connection(self, layer_id, hidden_dim):
        """
        Add a lateral connection to another layer

        Args:
            layer_id: ID of the layer to connect to
            hidden_dim: Hidden dimension of the target layer
        """
        if str(layer_id) in self.lateral_connections:
            # Update existing connection dimension
            old_dim = self.lateral_connections[str(layer_id)]
            if old_dim == hidden_dim:
                return False

            # Resize the weight matrix
            old_weights = self.lateral_weights[str(layer_id)].data
            new_weights = torch.zeros(self.hidden_dim, hidden_dim, device=old_weights.device)

            # Copy existing weights where dimensions match
            min_rows = min(old_weights.size(0), new_weights.size(0))
            min_cols = min(old_weights.size(1), new_weights.size(1))
            new_weights[:min_rows, :min_cols] = old_weights[:min_rows, :min_cols]

            # Initialize new weights
            if hidden_dim > old_dim:
                # Initialize new columns
                new_weights[:, old_dim:] = torch.randn_like(new_weights[:, old_dim:]) * 0.01

            # Update connection
            self.lateral_connections[str(layer_id)] = hidden_dim
            self.lateral_weights[str(layer_id)] = nn.Parameter(new_weights)
        else:
            # Create new connection
            device = next(self.parameters()).device
            weights = torch.zeros(self.hidden_dim, hidden_dim, device=device)
            # Initialize with small random values
            nn.init.kaiming_normal_(weights, nonlinearity='relu')
            weights.mul_(0.01)  # Start with small values

            # Add connection
            self.lateral_connections[str(layer_id)] = hidden_dim
            self.lateral_weights[str(layer_id)] = nn.Parameter(weights)

        return True

    def set_backward_connection(self, backward_layer):
        """
        Set or update backward connection

        Args:
            backward_layer: Layer to connect backwards to
        """
        device = next(self.parameters()).device

        # If we've already got a backward connection of the same dim & object, bail
        if self.backward_connections:
            old_dim = self.backward_connections.hidden_dim
            if old_dim == backward_layer.hidden_dim and self.backward_connections == backward_layer:
                return False

        # Assign the new backward connection & projection
        self.backward_connections = backward_layer
        self.backward_proj = nn.Linear(self.hidden_dim, backward_layer.hidden_dim, device=device)

        # Initialize with small random values
        nn.init.kaiming_normal_(self.backward_proj.weight, nonlinearity='relu')
        self.backward_proj.weight.data.mul_(0.01)  # Start with small values

        # Reset backward weights for this index (ensure idx_item exists)
        if self.backward_connections:
            self.backward_proj.weight[:, idx_item] = (
                torch.randn_like(self.backward_proj.weight[:, idx_item]) * 0.02
            )

        return True


        # Reset activation tracking
        self.neuron_importance[idx_item] = 1.0
        self.activation_history[idx_item] = 0.0
        self.activation_variance[idx_item] = 1.0
        self.gradient_history[idx_item] = 0.0

        # Reset growth state
        self.growth_state["neuron_ages"][idx_item] = 0
        self.growth_state["max_activation"][idx_item] = 0
        self.growth_state["min_activation"][idx_item] = 0

        # Evolve the parametric activation function
        self.activation_fn.evolve(learning_rate)

        # Evolution statistics
        evolution_stats = {
            "status": "completed",
            "layer_id": self.layer_id,
            "updates": self.updates,
            "hidden_dim": self.hidden_dim,
            "mean_importance": normalized_importance.mean().item(),
            "max_importance": normalized_importance.max().item(),
            "min_importance": normalized_importance.min().item(),
            "pruned_neurons": pruned_neurons,
            "pruned_count": len(pruned_neurons),
            "timestamp": time.time()
        }

        return evolution_stats

    def add_lateral_connection(self, layer_id, hidden_dim):
        """
        Add a lateral connection to another layer

        Args:
            layer_id: ID of the layer to connect to
            hidden_dim: Hidden dimension of the target layer
        """
        if str(layer_id) in self.lateral_connections:
            # Update existing connection dimension
            old_dim = self.lateral_connections[str(layer_id)]
            if old_dim == hidden_dim:
                return False

            # Resize the weight matrix
            old_weights = self.lateral_weights[str(layer_id)].data
            new_weights = torch.zeros(self.hidden_dim, hidden_dim, device=old_weights.device)

            # Copy existing weights where dimensions match
            min_rows = min(old_weights.size(0), new_weights.size(0))
            min_cols = min(old_weights.size(1), new_weights.size(1))
            new_weights[:min_rows, :min_cols] = old_weights[:min_rows, :min_cols]

            # Initialize new weights
            if hidden_dim > old_dim:
                # Initialize new columns
                new_weights[:, old_dim:] = torch.randn_like(new_weights[:, old_dim:]) * 0.01

            # Update connection
            self.lateral_connections[str(layer_id)] = hidden_dim
            self.lateral_weights[str(layer_id)] = nn.Parameter(new_weights)
        else:
            # Create new connection
            device = next(self.parameters()).device
            weights = torch.zeros(self.hidden_dim, hidden_dim, device=device)
            # Initialize with small random values
            nn.init.kaiming_normal_(weights, nonlinearity='relu')
            weights.mul_(0.01)  # Start with small values

            # Add connection
            self.lateral_connections[str(layer_id)] = hidden_dim
            self.lateral_weights[str(layer_id)] = nn.Parameter(weights)

        return True

    def set_backward_connection(self, backward_layer):
        """
        Set or update backward connection

        Args:
            backward_layer: Layer to connect backwards to
        """
        device = next(self.parameters()).device

        if self.backward_connections:
            old_dim = self.backward_connections.hidden_dim
            if old_dim == backward_layer.hidden_dim and self.backward_connections == backward_layer:
                return False

        self.backward_connections = backward_layer
        self.backward_proj = nn.Linear(self.hidden_dim, backward_layer.hidden_dim, device=device)

        # Initialize with small random values
        nn.init.kaiming_normal_(self.backward_proj.weight, nonlinearity='relu')
        self.backward_proj.weight.data.mul_(0.01)  # Start with small values

        return True

    def get_neuron_stats(self):
        """Get detailed statistics about neuron activity and importance"""
        with torch.no_grad():
            stats = {
                "importance": self.neuron_importance.cpu().numpy().tolist(),
                "activation": self.activation_history.cpu().numpy().tolist(),
                "variance": self.activation_variance.cpu().numpy().tolist(),
                "ages": self.growth_state["neuron_ages"].cpu().numpy().tolist(),
                "growth_events": self.growth_events,
                "lateral_connections": list(self.lateral_connections.keys()),
                "has_backward": self.backward_connections is not None
            }
        return stats

class AdaptiveAttention(nn.Module):
    """Adaptive attention mechanism that can evolve over time"""

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1, growth_factor=1.4):
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

        # Create unified neural-linguistic processing components
        if self.config.unified_perception:
            # Use direct signal processing without tokenization
            self.signal_processor = DirectSignalProcessor(self.config)
            self.conceptual_system = EmergentConceptualSystem(self.config)
        else:
            # For backward compatibility, create traditional components
            # Create concept bank (traditional mode)
            self.concept_bank = ConceptMemoryBank(
                concept_dim=self.config.initial_hidden_dim,
                initial_size=self.config.concept_memory_size,
                device=self.config.device
            )

            # Create segmentation (traditional mode)
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

        # Neural core: Unified or traditional layers
        if self.config.unified_perception:
            # Unified perception-cognition layers
            self.layers = nn.ModuleList([
                UnifiedPerceptionCognitionLayer(
                    self.config.initial_hidden_dim,
                    growth_factor=self.config.growth_factor,
                    layer_id=i
                )
                for i in range(self.config.initial_num_layers)
            ])
        else:
            # Traditional layers (for backward compatibility)
            self.layers = nn.ModuleList([
                NeuroplasticLayer(
                    self.config.initial_hidden_dim,
                    growth_factor=self.config.growth_factor,
                    dropout=0.1,
                    layer_id=i,
                    device=self.config.device
                )
                for i in range(self.config.initial_num_layers)
            ])

        # Output normalization
        self.norm = nn.LayerNorm(self.config.initial_hidden_dim)

        # Output projection for language modeling
        self.lm_head = nn.Linear(
            self.config.initial_hidden_dim,
            self.config.concept_memory_size if not self.config.unified_perception else self.config.initial_hidden_dim,
            bias=False
        )

        # Tie weights with concept embeddings if using traditional mode
        if not self.config.unified_perception:
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

        # Initialize self-evolution capabilities
        self = self.initialize_self_evolution()

        # Growth and evolution tracking
        self.growth_history = []
        self.global_step = 0

        # Current modality tracking
        self.current_modality = "text"

        # Initialize weights
        self._init_weights()

        # Move to target device
        self.to(self.config.device)

        # Initialize self-evolution capabilities

    def _init_weights(self):
        """Initialize model weights"""
        # Initialize position embeddings
        nn.init.normal_(self.position_embeddings.weight, std=0.02)

    def initialize_self_evolution(self):
        """Initialize self-evolution capabilities"""
        return extend_sam_with_self_evolution(self)

    def forward(self, input_chars=None, input_concepts=None, concept_mask=None,
               target_concepts=None, return_dict=False, use_thought_state=True,
               use_hive_mind=True, modality=None, image_data=None, audio_data=None):
        """Forward pass with either raw characters or concept IDs"""
        # Set current modality if provided
        if modality:
            self.current_modality = modality
            if hasattr(self, "segmentation") and hasattr(self.segmentation, "set_modality"):
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
                        if hasattr(self, "segmentation") and hasattr(self.segmentation, "set_modality"):
                            self.segmentation.set_modality("image")

            # Process audio if provided
            if audio_data is not None and hasattr(self, "multimodal_processor"):
                audio_embeddings = self.multimodal_processor.process_audio(audio_data)
                if audio_embeddings is not None:
                    multimodal_embeddings["audio"] = audio_embeddings
                    if modality is None and "image" not in multimodal_embeddings:
                        self.current_modality = "audio"
                        if hasattr(self, "segmentation") and hasattr(self.segmentation, "set_modality"):
                            self.segmentation.set_modality("audio")

        # Check hardware status if adaptive
        if self.hardware_manager:
            self.hardware_manager.check_memory()

        # Handle different input processing based on mode
        if self.config.unified_perception:
            # Unified mode: process raw inputs directly
            if input_chars is not None:
                # Process raw input signal
                hidden_states = self.signal_processor(input_chars, modality=self.current_modality)

                # Process through conceptual system
                hidden_states = self.conceptual_system(hidden_states, modality=self.current_modality)

                # No separate position embeddings needed - handled in signal processor
            else:
                # Use provided hidden states directly (for internal processing)
                hidden_states = input_concepts
        else:
            # Traditional mode: use existing tokenization-based approach
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
                    self.current_modality = "text"
                    if hasattr(self.segmentation, "set_modality"):
                        self.segmentation.set_modality("multimodal")

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

        # Apply thought state processing if enabled
        if use_thought_state:
            # Update thought state with current hidden states
            thought_context = self.thought_state.update(
                hidden_states,
                use_hive_mind=use_hive_mind and self.config.hive_enabled,
                modality=self.current_modality
            )

            # Get sequence properties
            if len(hidden_states.shape) == 3:
                # [batch, seq_len, hidden]
                batch_size, seq_len, _ = hidden_states.shape
            else:
                # Just [batch, hidden]
                batch_size = hidden_states.shape[0]
                seq_len = 1
                hidden_states = hidden_states.unsqueeze(1)

            # Enhance hidden states with thought context
            thought_projection = self.thought_state.project_to_concept_space(
                modality=self.current_modality
            )
            # Expand thought projection to match sequence length
            thought_expanded = thought_projection.expand(batch_size, seq_len, -1)
            # Blend hidden states with thought projection using attention
            hidden_states = hidden_states + self.thought_attention(hidden_states, cross_input=thought_expanded)

        # Apply layers
        mask = attention_mask if not self.config.unified_perception else None

        for layer in self.layers:
            layer_output = layer(hidden_states, mask, modality=self.current_modality)
            # Handle possible tuple return from neuroplastic layer
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output

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
                loss_fn = nn.CrossEntropyLoss(reduction='none')  # Use 'none' to get per-element losses
                # Get loss per element, keeping batch dimension
                loss = loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)),
                             shift_targets.reshape(-1))
                # Reshape loss back to batch dimension
                loss = loss.reshape(shift_logits.size(0), -1)

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
                if not getattr(self.hive_synchronizer, "sync_active", False):
                    self.hive_synchronizer.start_sync()

        # Handle loss reduction for batches
        if loss is not None:
            # Keep unreduced loss in return_dict, trainer will handle reduction
            unreduced_loss = loss
            # Compute mean loss per sequence for non-dict return
            loss = loss.mean(dim=-1)  # Mean across sequence length
            # loss is now [batch_size]

        # Return dictionary if requested
        if return_dict:
            return {
                "loss": unreduced_loss if loss is not None else None,  # Return unreduced loss
                "logits": logits,
                "hidden_states": hidden_states,
                "modality": self.current_modality
            }
        else:
            return (loss, logits, hidden_states)

    def process_text(self, text, private_context=False, modality="text"):
        """Process raw text into hidden states or concept IDs"""
        # Set private context if requested (for both modes)
        if private_context:
            if hasattr(self, "segmentation") and hasattr(self.segmentation, "set_private_context"):
                self.segmentation.set_private_context("user_private")

        # Set modality
        if modality != "text":
            self.current_modality = modality
            if hasattr(self, "segmentation") and hasattr(self.segmentation, "set_modality"):
                self.segmentation.set_modality(modality)

        try:
            # Process differently based on model mode
            if self.config.unified_perception:
                # Convert text to raw characters
                if isinstance(text, str):
                    # Convert to ASCII/UTF-8 values
                    raw_chars = torch.tensor([ord(c) for c in text], dtype=torch.float)

                    # Normalize to [0, 1] range for better signal processing
                    raw_chars = raw_chars / 256.0

                    # Add batch dimension
                    raw_chars = raw_chars.unsqueeze(0)

                    # Move to device
                    device = next(self.parameters()).device
                    raw_chars = raw_chars.to(device)

                    # Process through signal processor and conceptual system
                    with torch.no_grad():
                        processed = self.signal_processor(raw_chars, modality=modality)
                        conceptual = self.conceptual_system(processed, modality=modality)

                    return conceptual, None  # No segments in unified mode
                else:
                    # Assume already processed
                    return text, None
            else:
                # Traditional processing
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
            if private_context and hasattr(self, "segmentation") and hasattr(self.segmentation, "clear_private_context"):
                self.segmentation.clear_private_context()

    def generate(self, input_text=None, input_concepts=None, max_length=100,
                temperature=1.0, top_k=50, top_p=0.9, private_context=False,
                use_hive_mind=True, modality=None, image_data=None, audio_data=None):
        """Generate text from either raw text or hidden states/concept IDs"""
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
            if hasattr(self, "segmentation") and hasattr(self.segmentation, "set_modality"):
                self.segmentation.set_modality(modality)

        # Convert input text to processed form if provided
        if input_text is not None and input_concepts is None:
            # Process raw text
            processed, segments = self.process_text(
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

            # In unified mode, processed is already hidden states
            # In traditional mode, processed is concept IDs that need to be converted to tensor
            if not self.config.unified_perception:
                # Convert to tensor if needed
                if not torch.is_tensor(processed):
                    device = next(self.parameters()).device
                    processed = torch.tensor(processed, dtype=torch.long, device=device).unsqueeze(0)
                else:
                    processed = processed.unsqueeze(0)

            # Store for generation
            input_concepts = processed
        elif input_concepts is not None:
            # Ensure proper format for generation
            if not torch.is_tensor(input_concepts):
                device = next(self.parameters()).device
                if self.config.unified_perception:
                    # In unified mode, expect hidden states (float)
                    input_concepts = torch.tensor(input_concepts, dtype=torch.float, device=device).unsqueeze(0)
                else:
                    # In traditional mode, expect concept IDs (long)
                    input_concepts = torch.tensor(input_concepts, dtype=torch.long, device=device).unsqueeze(0)
            elif input_concepts.dim() == 1:
                # Add batch dimension
                input_concepts = input_concepts.unsqueeze(0)

        # Reset thought state for generation
        self.thought_state.reset(batch_size=input_concepts.shape[0])

        # Set model to eval mode
        was_training = self.training
        self.eval()

        try:
            # Set private context if requested
            if private_context and hasattr(self, "segmentation") and hasattr(self.segmentation, "set_private_context"):
                self.segmentation.set_private_context("user_private")

            # Generate
            with torch.no_grad():
                # Track generated sequence
                current_output = input_concepts
                cur_len = current_output.shape[1]

                while cur_len < max_length:
                    # Get model output
                    outputs = self(
                        input_concepts=current_output,
                        return_dict=True,
                        use_hive_mind=use_hive_mind,
                        modality=self.current_modality,
                        image_data=image_data if cur_len == input_concepts.shape[1] else None,
                        audio_data=audio_data if cur_len == input_concepts.shape[1] else None
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

                    # In unified mode, we need to convert the token ID to a hidden state
                    if self.config.unified_perception:
                        # For unified mode, generate a hidden state from the sampled token ID
                        # Convert token ID to embedding
                        batch_size = current_output.shape[0]

                        token_ids = next_token.squeeze(-1)  # [batch]
                        token_embedding = self.lm_head.weight[token_ids]
                        token_embedding = token_embedding.unsqueeze(1)

                    else:
                        # Traditional mode - just add the token ID
                        current_output = torch.cat([current_output, next_token], dim=1)

                    cur_len += 1

            # Convert generated sequence to text
            if self.config.unified_perception:
                # For unified mode, we need to decode hidden states to text
                generated_text = self._decode_hidden_states(current_output[0])
            else:
                # Traditional mode - convert concept IDs to text
                generated_text = self._concepts_to_text(current_output[0].tolist())

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
            if private_context and hasattr(self, "segmentation") and hasattr(self.segmentation, "clear_private_context"):
                self.segmentation.clear_private_context()

    def _decode_hidden_states(self, hidden_states):
        """Decode hidden states to text in unified mode"""
        # This is a crucial method for the unified architecture
        # We need to convert from hidden states back to characters/text

        # Get most similar concepts for each hidden state
        text_parts = []

        # Process each position in the sequence
        for i in range(hidden_states.size(0)):
            # Get hidden state for this position
            hidden = hidden_states[i]

            # Find most similar concepts
            if hasattr(self, "conceptual_system"):
                similar = self.conceptual_system.find_similar_concepts(hidden, top_k=1, modality="text")

                if similar:
                    concept_id, similarity = similar[0]

                    # Get metadata for this concept
                    if concept_id in self.conceptual_system.concept_metadata:
                        metadata = self.conceptual_system.concept_metadata[concept_id]

                        # If there's a source string, use it
                        if "source" in metadata:
                            text_parts.append(metadata["source"])
                            continue

            # If no suitable concept found or no source available, use output projection
            # Project to vocabulary distribution and take most likely token
            if hasattr(self, "lm_head"):
                logits = self.lm_head(hidden.unsqueeze(0)).squeeze(0)
                token_id = logits.argmax().item()

                # Convert token ID to character
                # For simplicity, treat it as an ASCII value
                if token_id < 128:
                    text_parts.append(chr(token_id))
                else:
                    # Use a placeholder for tokens outside standard ASCII
                    text_parts.append("·")

        # Join all parts into a single string
        return "".join(text_parts)

    def _concepts_to_text(self, concept_ids):
        """Convert concept IDs back to text in traditional mode"""
        text_parts = []

        if self.config.unified_perception:
            # Should not be called in unified mode, but handle it just in case
            return self._decode_hidden_states(torch.tensor(concept_ids, device=self.device))

        # Traditional mode conversion
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

        # Evolve neural components
        layer_stats = []

        # Evolve unified or traditional layers
        for layer in self.layers:
            stats = layer.evolve()
            if stats:
                layer_stats.append(stats)

        # Evolve signal processor in unified mode
        if self.config.unified_perception and hasattr(self, "signal_processor"):
            signal_patterns = self.signal_processor.get_frequent_patterns(limit=5)

            # If we have established patterns, consider adding them to conceptual system
            if signal_patterns and hasattr(self, "conceptual_system"):
                for _, pattern_tensor, freq in signal_patterns:
                    if freq > 10:  # Only consider frequent patterns
                        # Create concept vector from pattern
                        concept_vec = torch.mean(pattern_tensor, dim=0)

                        # Find if similar concept already exists
                        similar = self.conceptual_system.find_similar_concepts(concept_vec, top_k=1)

                        if not similar or similar[0][1] < 0.9:  # If no similar concept found
                            # Consider adding as new concept
                            self.conceptual_system._consider_new_concept(concept_vec, modality=self.current_modality)

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
        if self.config.hive_enabled and self.hive_synchronizer and not getattr(self.hive_synchronizer, "sync_active", False):
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

            # Grow unified or traditional components
            if self.config.unified_perception:
                # Grow signal processor
                if hasattr(self, "signal_processor"):
                    self.signal_processor.grow(new_hidden_dim)

                # Grow conceptual system
                if hasattr(self, "conceptual_system"):
                    self.conceptual_system.grow(new_hidden_dim)
            else:
                # Grow traditional components
                if hasattr(self, "concept_bank"):
                    # For traditional mode, grow concept bank
                    self.concept_bank.grow_if_needed()

                if hasattr(self, "segmentation"):
                    # Grow segmentation
                    self.segmentation.grow(new_hidden_dim)

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

            # Grow multimodal processor if present
            if self.config.multimodal_enabled and hasattr(self, "multimodal_processor"):
                self.multimodal_processor.grow(new_hidden_dim)

            # Grow LM head
            old_lm_head = self.lm_head

            # In unified mode, output dimension is hidden_dim
            # In traditional mode, output dimension is concept_memory_size
            output_dim = new_hidden_dim if self.config.unified_perception else self.config.concept_memory_size

            # Create new LM head
            new_lm_head = nn.Linear(
                new_hidden_dim,
                output_dim,
                bias=False
            ).to(old_lm_head.weight.device)

            # Transfer weights
            with torch.no_grad():
                if self.config.unified_perception:
                    # In unified mode, output dimension grows with hidden dimension
                    # Copy weights for the smaller of the two dimensions
                    min_dim = min(old_lm_head.weight.size(0), new_lm_head.weight.size(0))
                    min_hidden = min(old_lm_head.weight.size(1), new_lm_head.weight.size(1))
                    new_lm_head.weight[:min_dim, :min_hidden] = old_lm_head.weight[:min_dim, :min_hidden]
                else:
                    # In traditional mode, only input dimension changes
                    new_lm_head.weight[:, :current_dim] = old_lm_head.weight

            # Replace LM head
            self.lm_head = new_lm_head

            # Tie weights if in traditional mode
            if not self.config.unified_perception and hasattr(self, "concept_bank"):
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

                if self.config.unified_perception:
                    # Add unified layer
                    new_layer = UnifiedPerceptionCognitionLayer(
                        new_hidden_dim,
                        growth_factor=self.config.growth_factor,
                        layer_id=layer_id
                    ).to(self.layers[0].norm1.weight.device)
                else:
                    # Add traditional layer
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

        # Check if concept bank needs to grow in traditional mode
        if not self.config.unified_perception and hasattr(self, "concept_bank"):
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

        # Save concept metadata and patterns
        if self.config.unified_perception and hasattr(self, "conceptual_system"):
            # Save unified mode concepts
            concept_metadata = {
                str(k): v for k, v in self.conceptual_system.concept_metadata.items()
            }

            # Save in concepts.json
            with open(os.path.join(path, "concepts.json"), "w") as f:
                json.dump(concept_metadata, f, indent=2)

            # Save conceptual patterns from signal processor if available
            if hasattr(self, "signal_processor") and hasattr(self.signal_processor, "pattern_frequency"):
                pattern_data = {
                    "pattern_frequency": dict(self.signal_processor.pattern_frequency),
                    "pattern_timestamps": {k: v for k, v in self.signal_processor.pattern_timestamps.items()}
                }

                with open(os.path.join(path, "patterns.json"), "w") as f:
                    json.dump(pattern_data, f, indent=2)
        else:
            # Save traditional concept metadata
            if hasattr(self, "concept_bank"):
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
            modality_stats = {}

            if self.config.unified_perception and hasattr(self, "conceptual_system"):
                modality_stats = {
                    "modality_counts": {
                        modality: len(concepts)
                        for modality, concepts in self.conceptual_system.modality_concepts.items()
                    }
                }
            elif hasattr(self, "concept_bank"):
                modality_stats = {
                    "modality_counts": {
                        modality: len(concepts)
                        for modality, concepts in self.concept_bank.modality_concepts.items()
                    }
                }

            # Add experience stats
            modality_stats["experience_stats"] = self.experience_manager.get_modality_stats()
            modality_stats["current_modality"] = self.current_modality

            with open(os.path.join(path, "multimodal_state.json"), "w") as f:
                json.dump(modality_stats, f, indent=2)

        logger.info(f"Model saved to {path}")
        return path

    def load_sam_vocabulary(self, vocab_path=None):
        """Initialize with Sam-like vocabulary"""
        if self.config.unified_perception:
            logger.warning("Vocabulary loading not needed in unified perception mode")
            return 0

        # Only for traditional mode
        if hasattr(self, "concept_bank"):
            if vocab_path is None:
                # Create built-in Sam-style vocabulary
                vocabulary = []

                # Add common words and phrases in Sam's style
                words = [
                    # Common function words
                    "the", "and", "of", "to", "in", "a", "is", "that", "for", "it", "with", "as", "on",
                    "be", "by", "this", "an", "at", "which", "but", "from", "or", "have", "one", "had",
                    "not", "what", "all", "were", "when", "we", "there", "can", "who", "been", "has",
                    "their", "if", "would", "will", "they", "so", "you", "said", "may", "these", "no",
                ]

                # Add common word combinations
                for i, word1 in enumerate(words[:100]):  # Limit combinations to avoid explosion
                    vocabulary.append(word1)
                    for word2 in words[i+1:min(i+20, len(words))]:
                        vocabulary.append(f"{word1} {word2}")

                # Create vocabulary file
                temp_vocab_path = os.path.join(self.config.save_dir, "temp_sam_vocab.txt")
                with open(temp_vocab_path, 'w') as f:
                    for item in vocabulary:
                        f.write(f"{item}\n")

                return self.concept_bank.load_vocabulary(temp_vocab_path)
            else:
                return self.concept_bank.load_vocabulary(vocab_path)

        return 0

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
        # Get conceptual system stats
        if self.config.unified_perception and hasattr(self, "conceptual_system"):
            concept_stats = self.conceptual_system.get_concept_stats()
            segmentation_stats = {
                "total_patterns": len(getattr(self.signal_processor, "pattern_memory", {})),
                "current_modality": self.current_modality
            }
        else:
            # Traditional mode
            concept_stats = self.concept_bank.get_concept_stats() if hasattr(self, "concept_bank") else {}
            segmentation_stats = self.segmentation.get_segmentation_stats() if hasattr(self, "segmentation") else {}

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
                "modality_counts": concept_stats.get("modality_stats", {}),
                "current_modality": self.current_modality,
                "experience_counts": self.experience_manager.get_modality_stats()
            }

        return {
            "model_size": {
                "hidden_dim": self.layers[0].hidden_dim,
                "num_layers": len(self.layers),
                "total_concepts": concept_stats.get("total_concepts", 0),
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
                "multimodal_enabled": self.config.multimodal_enabled,
                "unified_perception": self.config.unified_perception
            }
        }

    def process_multimodal(self, input_data, modality="image"):
        """Process multimodal input data"""
        if not self.config.multimodal_enabled:
            logger.warning("Multimodal processing requested but not enabled in config")
            return None

        # Set current modality
        self.current_modality = modality
        if hasattr(self, "segmentation") and hasattr(self.segmentation, "set_modality"):
            self.segmentation.set_modality(modality)

        # Process based on modality
        if modality == "image" and hasattr(self, "multimodal_processor"):
            return self.multimodal_processor.process_image(input_data)
        elif modality == "audio" and hasattr(self, "multimodal_processor"):
            return self.multimodal_processor.process_audio(input_data)

        return None

    @classmethod
    def create_with_auto_config(cls, base_config=None, load_vocab=True, unified_perception=False):
        """Create a new SAM instance with auto-configured hardware settings"""
        # Start with default or provided config
        config = base_config or SAMConfig()

        # Set unified perception mode
        config.unified_perception = unified_perception

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

            # Initialize with vocabulary if requested (only in traditional mode)
            if load_vocab and not unified_perception:
                model.load_sam_vocabulary()

            return model, config
        else:
            # If hardware manager not available, just return the temp model
            if load_vocab and not unified_perception:
                temp_model.load_sam_vocabulary()
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

        # Load concept metadata in the appropriate mode
        if config.unified_perception and hasattr(model, "conceptual_system"):
            # Load unified mode concepts
            try:
                with open(os.path.join(path, "concepts.json"), "r") as f:
                    concept_metadata = json.load(f)
                    model.conceptual_system.concept_metadata = {
                        int(k): v for k, v in concept_metadata.items()
                    }
            except Exception as e:
                logger.warning(f"Error loading concept metadata: {e}")

            # Load signal processor patterns if available
            try:
                if hasattr(model, "signal_processor"):
                    with open(os.path.join(path, "patterns.json"), "r") as f:
                        pattern_data = json.load(f)
                        if "pattern_frequency" in pattern_data:
                            model.signal_processor.pattern_frequency = Counter(pattern_data["pattern_frequency"])
                        if "pattern_timestamps" in pattern_data:
                            model.signal_processor.pattern_timestamps = {k: v for k, v in pattern_data["pattern_timestamps"].items()}
            except Exception as e:
                logger.warning(f"Error loading pattern data: {e}")
        else:
            # Load traditional concept metadata
            try:
                with open(os.path.join(path, "concepts.json"), "r") as f:
                    concept_metadata = json.load(f)
                    if hasattr(model, "concept_bank"):
                        model.concept_bank.concept_metadata = {
                            int(k): v for k, v in concept_metadata.items()
                        }
            except Exception as e:
                logger.warning(f"Error loading concept metadata: {e}")

            # Load source mapping
            try:
                with open(os.path.join(path, "source_mapping.json"), "r") as f:
                    source_mapping = json.load(f)
                    if hasattr(model, "concept_bank"):
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
# COGNITIVE SYSTEMS
###########################################

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
        self.multimodal_enabled = getattr(self.model.config, 'multimodal_enabled', False)

        # State tracking
        self.last_successful_dream = None
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3

        # Resource monitoring
        self.memory_threshold = 0.9  # 90% memory usage threshold

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

        # Get concept stats
        if self.model.config.unified_perception and hasattr(self.model, "conceptual_system"):
            concept_stats = self.model.conceptual_system.get_concept_stats()
        elif hasattr(self.model, "concept_bank"):
            concept_stats = self.model.concept_bank.get_concept_stats()
        else:
            concept_stats = {"total_concepts": 0}

        return {
            "duration_minutes": duration_minutes,
            "dream_cycles": dream_count,
            "syntheses": len(self.synthesis_history),
            "concepts_reinforced": concept_stats
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
                    if hasattr(self.model, "segmentation") and hasattr(self.model.segmentation, "set_private_context"):
                        self.model.segmentation.set_private_context("dream")

                    # Perform dream cycle
                    self.dream_cycle(duration_minutes=interval_minutes)

                    # Restore model mode
                    if was_training:
                        self.model.train()

                    # Clear private context
                    if hasattr(self.model, "segmentation") and hasattr(self.model.segmentation, "clear_private_context"):
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
        # Handle unified vs traditional mode
        if self.model.config.unified_perception:
            # In unified mode, identify and reinforce important concepts from the conceptual system
            if hasattr(self.model, "conceptual_system"):
                # Get concept stats
                concept_stats = self.model.conceptual_system.get_concept_stats()

                # Get top concepts
                for concept_id in range(self.model.conceptual_system.next_concept_id):
                    if concept_id in self.model.conceptual_system.concept_metadata:
                        # Increase importance by slightly increasing count
                        self.model.conceptual_system.update_concept_usage(concept_id)

                # Look for concepts to merge
                if self.model.conceptual_system.next_concept_id > 1:
                    # Pick random concept
                    concept_id1 = random.randint(0, self.model.conceptual_system.next_concept_id - 1)

                    # Find similar concepts
                    concept_vector = self.model.conceptual_system.concept_prototypes[concept_id1]
                    similar = self.model.conceptual_system.find_similar_concepts(concept_vector, top_k=5)

                    # Find a match with similarity between 0.3 and 0.7
                    for concept_id2, similarity in similar:
                        if concept_id2 != concept_id1 and 0.3 < similarity < 0.7:
                            # Get modalities
                            mod1 = self.model.conceptual_system.concept_metadata.get(concept_id1, {}).get("modality", "text")
                            mod2 = self.model.conceptual_system.concept_metadata.get(concept_id2, {}).get("modality", "text")

                            # Create merged concept
                            merged_modality = "multimodal" if mod1 != mod2 else mod1

                            merged_id = self.model.conceptual_system.create_merged_concept(
                                concept_id1, concept_id2, modality=merged_modality
                            )

                            # Record synthesis
                            if merged_id is not None:
                                self.synthesis_history.append({
                                    "type": "concept_merge",
                                    "concept_id1": concept_id1,
                                    "concept_id2": concept_id2,
                                    "similarity": similarity,
                                    "timestamp": time.time(),
                                    "multimodal": mod1 != mod2
                                })

                            break
        else:
            # Traditional mode - get top concepts by usage
            if hasattr(self.model, "concept_bank"):
                concept_stats = self.model.concept_bank.get_concept_stats()
                top_concepts = concept_stats.get("top_concepts", [])

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
        # Create seed prompts
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
        # Different approaches for unified vs traditional mode
        if self.model.config.unified_perception:
            # In unified mode, use patterns from signal processor
            if hasattr(self.model, "signal_processor") and hasattr(self.model.signal_processor, "pattern_frequency"):
                patterns = self.model.signal_processor.pattern_frequency.most_common(20)

                # Create seeds from most common patterns
                seeds = []
                for pattern_key, _ in patterns:
                    if isinstance(pattern_key, str) and len(pattern_key) > 5:
                        seeds.append(pattern_key[:20])  # Take first part as seed

                # Add default seeds if we don't have enough
                if len(seeds) < 5:
                    seeds.extend([
                        "The concept of",
                        "I believe that",
                        "Let me think about",
                        "In this context",
                        "The most important"
                    ])

                return seeds
        else:
            # Traditional mode - get patterns from segmentation
            if hasattr(self.model, "segmentation") and hasattr(self.model.segmentation, "pattern_memory"):
                patterns = self.model.segmentation.pattern_memory.get_frequent_patterns(limit=20)

                if not patterns:
                    # No patterns yet, use some default prompts
                    return [
                        "The concept of",
                        "I reckon that",
                        "Let me tell ya",
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

        # Default seeds if all else fails
        return [
            "The concept of",
            "I believe that",
            "Let me think about",
            "In this context",
            "The most important"
        ]

    def _prune_concepts(self):
        """Remove or consolidate less useful concepts"""
        # Different approaches for unified vs traditional mode
        if self.model.config.unified_perception:
            # In unified mode, prune is handled by conceptual system
            pass
        else:
            # Traditional mode
            # Skip if we don't have many concepts yet
            if hasattr(self.model, "concept_bank") and self.model.concept_bank.next_concept_id < 200:
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

        # Handle differently based on model mode
        if self.model.config.unified_perception:
            # In unified mode, use conceptual system
            if hasattr(self.model, "conceptual_system"):
                # Get modality stats
                modality_stats = self.model.conceptual_system.get_concept_stats().get("modality_stats", {})

                # Only proceed if we have concepts from multiple modalities
                modalities = ["text", "image", "audio", "multimodal"]
                modal_concepts = {}

                for modality in modalities:
                    if modality in modality_stats and modality_stats[modality] > 0:
                        # Get concepts for this modality
                        concepts = list(self.model.conceptual_system.modality_concepts.get(modality, set()))
                        if concepts:
                            # Select a few random concepts
                            sample_size = min(5, len(concepts))
                            modal_concepts[modality] = random.sample(concepts, sample_size)

                # Create cross-modal associations
                for modality1, concepts1 in modal_concepts.items():
                    for modality2, concepts2 in modal_concepts.items():
                        if modality1 == modality2 or modality1 == "multimodal" or modality2 == "multimodal":
                            continue  # Skip same modality or already multimodal

                        # Create a cross-modal connection
                        if concepts1 and concepts2:
                            concept_id1 = random.choice(concepts1)
                            concept_id2 = random.choice(concepts2)

                            # Create multimodal merged concept
                            merged_id = self.model.conceptual_system.create_merged_concept(
                                concept_id1, concept_id2, modality="multimodal"
                            )

                            if merged_id is not None:
                                # Record synthesis
                                self.synthesis_history.append({
                                    "type": "cross_modal_merge",
                                    "concept_id1": concept_id1,
                                    "concept_id2": concept_id2,
                                    "modality1": modality1,
                                    "modality2": modality2,
                                    "timestamp": time.time()
                                })
        else:
            # Traditional mode
            # Only proceed if we have concepts from multiple modalities
            if hasattr(self.model, "concept_bank"):
                modality_counts = self.model.concept_bank.get_concept_stats().get("modality_stats", {})
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

    def __init__(self, model, stability_threshold=0.6, novelty_weight=0.4):
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
        # Different implementation for unified vs traditional mode
        if self.model.config.unified_perception:
            # Unified mode - use conceptual system
            if hasattr(self.model, "conceptual_system"):
                # Get concept frequencies
                frequencies = torch.zeros(self.model.conceptual_system.next_concept_id,
                                         device=self.model.conceptual_system.concept_counts.device)

                # Copy only valid frequencies
                frequencies[:self.model.conceptual_system.next_concept_id] = \
                    self.model.conceptual_system.concept_counts[:self.model.conceptual_system.next_concept_id].float()

                # Calculate entropy
                total = frequencies.sum()
                if total > 0:
                    probabilities = frequencies / total
                    # Remove zeros
                    probabilities = probabilities[probabilities > 0]
                    # Calculate entropy
                    entropy = -torch.sum(probabilities * torch.log(probabilities))
                    return entropy.item()
            return 0.0
        else:
            # Traditional mode - use concept bank
            if hasattr(self.model, "concept_bank"):
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
        # Different implementation for unified vs traditional mode
        if self.model.config.unified_perception:
            # Unified mode - use conceptual system
            if hasattr(self.model, "conceptual_system"):
                # Skip if too few concepts
                if self.model.conceptual_system.next_concept_id < 20:
                    return {}

                # Use simple clustering for efficiency
                clusters = {}

                # Get most used concepts
                frequencies = self.model.conceptual_system.concept_counts[:self.model.conceptual_system.next_concept_id]
                values, indices = torch.topk(frequencies, min(100, len(frequencies)))

                # Calculate centroids for different concept types and modalities
                modality_centroids = {
                    modality: {
                        "centroid": torch.zeros(self.model.config.initial_hidden_dim, device=frequencies.device),
                        "count": 0
                    }
                    for modality in self.model.conceptual_system.modality_concepts.keys()
                }

                type_centroids = {
                    "emergent": torch.zeros(self.model.config.initial_hidden_dim, device=frequencies.device),
                    "merged": torch.zeros(self.model.config.initial_hidden_dim, device=frequencies.device)
                }

                type_counts = {"emergent": 0, "merged": 0}

                for idx in indices:
                    idx_item = idx.item()
                    if idx_item in self.model.conceptual_system.concept_metadata:
                        metadata = self.model.conceptual_system.concept_metadata[idx_item]
                        concept_type = metadata.get("type", "")
                        prototype = self.model.conceptual_system.concept_prototypes[idx_item]
                        modality = metadata.get("modality", "text")

                        # Update type centroid
                        if concept_type in type_centroids:
                            type_centroids[concept_type] += prototype
                            type_counts[concept_type] += 1

                        # Update modality centroid
                        if modality in modality_centroids:
                            modality_centroids[modality]["centroid"] += prototype
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
            return {}
        else:
            # Traditional mode - use concept bank
            if hasattr(self.model, "concept_bank"):
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
            return {}

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
        if self.model.config.unified_perception:
            # Unified mode - strengthen important concepts in conceptual system
            if hasattr(self.model, "conceptual_system"):
                for concept_type, centroid in self.identity_centroids.items():
                    # Find concepts with this type
                    similar_concepts = []

                    for cid in range(self.model.conceptual_system.next_concept_id):
                        if cid in self.model.conceptual_system.concept_metadata:
                            metadata = self.model.conceptual_system.concept_metadata[cid]
                            if metadata.get("type") == concept_type:
                                # Calculate similarity
                                prototype = self.model.conceptual_system.concept_prototypes[cid]
                                similarity = F.cosine_similarity(
                                    centroid.unsqueeze(0),
                                    prototype.unsqueeze(0),
                                    dim=1
                                ).item()

                                similar_concepts.append((cid, similarity))

                    # Get top concepts
                    similar_concepts.sort(key=lambda x: x[1], reverse=True)
                    top_concepts = similar_concepts[:20]

                    # Reinforce top concepts
                    for cid, _ in top_concepts:
                        self.model.conceptual_system.update_concept_usage(cid)

                # Also reinforce modality centroids
                for modality, centroid in self.modality_centroids.items():
                    # Find concepts with this modality
                    mod_concepts = list(self.model.conceptual_system.modality_concepts.get(modality, set()))

                    # Find most similar concepts to centroid
                    similarities = []
                    for cid in mod_concepts:
                        prototype = self.model.conceptual_system.concept_prototypes[cid]
                        similarity = F.cosine_similarity(
                            centroid.unsqueeze(0),
                            prototype.unsqueeze(0),
                            dim=1
                        ).item()

                        similarities.append((cid, similarity))

                    # Sort and get top concepts
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    top_concepts = similarities[:10]

                    # Reinforce top concepts
                    for cid, _ in top_concepts:
                        self.model.conceptual_system.update_concept_usage(cid)
        else:
            # Traditional mode
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

        # Apply differently based on model mode
        if self.model.config.unified_perception:
            # Unified mode - personalize concept in conceptual system
            if hasattr(self.model, "conceptual_system") and concept_id < self.model.conceptual_system.next_concept_id:
                with torch.no_grad():
                    # Get current prototype
                    current = self.model.conceptual_system.concept_prototypes[concept_id]

                    # Blend with personality vector
                    personalized = current * (1 - personalization_factor) + self.personality_vector * personalization_factor

                    # Normalize and update
                    personalized = F.normalize(personalized, dim=0)
                    self.model.conceptual_system.concept_prototypes[concept_id] = personalized

                    # Mark as personal
                    self.personal_concepts.add(concept_id)
        else:
            # Traditional mode - personalize concept in concept bank
            if hasattr(self.model, "concept_bank"):
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

        # Add usage statistics for different components based on model mode
        if self.model.config.unified_perception:
            # Unified mode components
            # Signal processor is always important
            self.component_usage["signal_processor"] = 1000

            # Conceptual system is important
            self.component_usage["conceptual_system"] = 800
        else:
            # Traditional mode components
            # Concept bank is always important
            self.component_usage["concept_bank"] = 1000

            # Segmentation depends on input/output activity
            if hasattr(self.model.segmentation, "total_segmentations"):
                self.component_usage["segmentation"] = self.model.segmentation.total_segmentations

        # Thought state is important
        self.component_usage["thought_state"] = 500

        # Multimodal processor importance depends on if being used
        if hasattr(self.model, "multimodal_processor"):
            # Check if any non-text modalities are active
            if self.model.config.unified_perception:
                # Use conceptual system modality stats
                modality_counts = self.model.conceptual_system.get_concept_stats().get("modality_stats", {})
            else:
                # Use concept bank modality stats
                modality_counts = self.model.concept_bank.get_concept_stats().get("modality_stats", {})

            non_text_count = sum(count for modality, count in modality_counts.items()
                               if modality != "text" and count > 0)

            if non_text_count > 0:
                self.component_usage["multimodal_processor"] = 800
            else:
                self.component_usage["multimodal_processor"] = 100

    def _get_component_by_name(self, name):
        """Get component by name"""
        if self.model.config.unified_perception:
            # Unified mode components
            if name == "signal_processor":
                return self.model.signal_processor
            elif name == "conceptual_system":
                return self.model.conceptual_system
        else:
            # Traditional mode components
            if name == "concept_bank":
                return self.model.concept_bank
            elif name == "segmentation":
                return self.model.segmentation

        # Common components
        if name == "thought_state":
            return self.model.thought_state
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
# MULTIMODAL COMPONENTS
###########################################
class MultimodalProcessor(nn.Module):
    """Production-ready multimodal processor for SAM's unified architecture"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.initial_hidden_dim

        # Image processing pipeline - Real CNN architecture
        self.image_encoder = nn.Sequential(
            # Initial conv layers for feature extraction
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # ResNet-style blocks for robust feature learning
            self._make_conv_block(64, 128, stride=2),
            self._make_conv_block(128, 256, stride=2),
            self._make_conv_block(256, 512, stride=2),

            # Global pooling and projection to SAM's conceptual space
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

        # Audio processing pipeline - Spectrogram + temporal modeling
        self.audio_encoder = nn.Sequential(
            # Spectrogram computation (learnable)
            SpectrogramLayer(n_fft=2048, hop_length=512, n_mels=128),

            # CNN for frequency patterns
            nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Temporal modeling with adaptive pooling
            nn.AdaptiveAvgPool2d((8, 16)),  # Reduce to manageable size
            nn.Flatten(),
            nn.Linear(128 * 8 * 16, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

        # Video processing pipeline - Temporal + spatial
        self.video_encoder = nn.Sequential(
            # 3D convolutions for spatiotemporal features
            nn.Conv3d(3, 32, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(2, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            # Temporal aggregation
            nn.AdaptiveAvgPool3d((1, 4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

        # Sensor data processing (IMU, GPS, etc.)
        self.sensor_encoder = nn.Sequential(
            nn.Linear(config.sensor_dim if hasattr(config, 'sensor_dim') else 32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

        # Cross-modal fusion architecture
        if config.multimodal_fusion_strategy == "attention":
            self.fusion = CrossModalAttentionFusion(self.hidden_dim)
        elif config.multimodal_fusion_strategy == "transformer":
            self.fusion = TransformerModalityFusion(self.hidden_dim)
        else:  # Advanced concatenation with learned weights
            self.fusion = LearnedModalityFusion(self.hidden_dim)

        # Modality type embeddings for disambiguation
        self.modality_embeddings = nn.Embedding(6, self.hidden_dim)  # text, image, audio, video, sensor, multimodal
        self.modality_map = {
            "text": 0, "image": 1, "audio": 2,
            "video": 3, "sensor": 4, "multimodal": 5
        }

        # Temporal sequence modeling for multimodal streams
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=8,
                dim_feedforward=self.hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )

        # Adaptive fusion weights based on modality confidence
        self.modality_confidence = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Cross-modal alignment for unified conceptual space
        self.cross_modal_aligner = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

    def _make_conv_block(self, in_channels, out_channels, stride=1):
        """Create ResNet-style convolutional block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def process_image(self, image_data):
        """Process raw image pixels through CNN pipeline"""
        if image_data is None:
            return None

        # Handle different input formats
        if isinstance(image_data, (list, tuple)):
            # Batch of images
            image_data = torch.stack([self._normalize_image(img) for img in image_data])
        else:
            # Single image or already batched
            image_data = self._normalize_image(image_data)
            if image_data.dim() == 3:
                image_data = image_data.unsqueeze(0)

        # Extract features through CNN
        image_features = self.image_encoder(image_data)

        # Add modality embedding
        modality_emb = self.modality_embeddings(
            torch.tensor(self.modality_map["image"], device=image_features.device)
        ).unsqueeze(0).expand(image_features.size(0), -1)

        # Combine features with modality information
        combined = image_features + modality_emb

        # Align to unified conceptual space
        aligned = self.cross_modal_aligner(combined)

        return aligned.unsqueeze(1)  # Add sequence dimension

    def process_audio(self, audio_data):
        """Process raw audio waveforms through spectrogram + CNN pipeline"""
        if audio_data is None:
            return None

        # Handle different audio formats
        if isinstance(audio_data, (list, tuple)):
            # Batch of audio clips
            audio_data = torch.stack([self._normalize_audio(aud) for aud in audio_data])
        else:
            # Single audio or already batched
            audio_data = self._normalize_audio(audio_data)
            if audio_data.dim() == 1:
                audio_data = audio_data.unsqueeze(0)

        # Process through audio pipeline
        audio_features = self.audio_encoder(audio_data)

        # Add modality embedding
        modality_emb = self.modality_embeddings(
            torch.tensor(self.modality_map["audio"], device=audio_features.device)
        ).unsqueeze(0).expand(audio_features.size(0), -1)

        # Combine features with modality information
        combined = audio_features + modality_emb

        # Align to unified conceptual space
        aligned = self.cross_modal_aligner(combined)

        return aligned.unsqueeze(1)  # Add sequence dimension

    def process_video(self, video_data):
        """Process video sequences through 3D CNN pipeline"""
        if video_data is None:
            return None

        # Handle video input (B, T, C, H, W) or (T, C, H, W)
        if video_data.dim() == 4:
            video_data = video_data.unsqueeze(0)

        # Transpose to (B, C, T, H, W) for 3D conv
        video_data = video_data.transpose(1, 2)

        # Process through video pipeline
        video_features = self.video_encoder(video_data)

        # Add modality embedding
        modality_emb = self.modality_embeddings(
            torch.tensor(self.modality_map["video"], device=video_features.device)
        ).unsqueeze(0).expand(video_features.size(0), -1)

        # Combine features with modality information
        combined = video_features + modality_emb

        # Align to unified conceptual space
        aligned = self.cross_modal_aligner(combined)

        return aligned.unsqueeze(1)  # Add sequence dimension

    def process_sensor(self, sensor_data):
        """Process sensor data (IMU, GPS, etc.)"""
        if sensor_data is None:
            return None

        # Process through sensor pipeline
        sensor_features = self.sensor_encoder(sensor_data)

        # Add modality embedding
        modality_emb = self.modality_embeddings(
            torch.tensor(self.modality_map["sensor"], device=sensor_features.device)
        ).unsqueeze(0).expand(sensor_features.size(0), -1)

        # Combine features with modality information
        combined = sensor_features + modality_emb

        # Align to unified conceptual space
        aligned = self.cross_modal_aligner(combined)

        return aligned.unsqueeze(1)  # Add sequence dimension

    def integrate_modalities(self, modality_embeddings):
        """Advanced multimodal integration with confidence weighting"""
        modalities = list(modality_embeddings.keys())

        if not modalities:
            return None

        if len(modalities) == 1:
            return modality_embeddings[modalities[0]]

        # Collect all valid embeddings
        valid_embeddings = []
        valid_modalities = []

        for modality in sorted(modalities):
            embedding = modality_embeddings[modality]
            if embedding is not None:
                # Ensure consistent dimensions
                if embedding.shape[-1] != self.hidden_dim:
                    # Dynamic projection for mismatched dimensions
                    projection = nn.Linear(
                        embedding.shape[-1],
                        self.hidden_dim,
                        device=embedding.device
                    ).to(embedding.device)
                    embedding = projection(embedding)

                valid_embeddings.append(embedding)
                valid_modalities.append(modality)

        if not valid_embeddings:
            return None

        if len(valid_embeddings) == 1:
            return valid_embeddings[0]

        # Calculate confidence scores for each modality
        confidence_scores = []
        for embedding in valid_embeddings:
            # Compute confidence based on embedding magnitude and consistency
            conf = self.modality_confidence(embedding.mean(dim=1))
            confidence_scores.append(conf)

        # Normalize confidence scores
        confidence_weights = F.softmax(torch.cat(confidence_scores, dim=1), dim=1)

        # Apply fusion strategy
        if isinstance(self.fusion, CrossModalAttentionFusion):
            fused = self.fusion(valid_embeddings, confidence_weights)
        elif isinstance(self.fusion, TransformerModalityFusion):
            fused = self.fusion(valid_embeddings, valid_modalities)
        else:  # LearnedModalityFusion
            fused = self.fusion(valid_embeddings, confidence_weights)

        # Apply temporal modeling if sequence length > 1
        if fused.size(1) > 1:
            fused = self.temporal_transformer(fused)

        # Add multimodal embedding to indicate fusion
        modality_emb = self.modality_embeddings(
            torch.tensor(self.modality_map["multimodal"], device=fused.device)
        ).unsqueeze(0).unsqueeze(1).expand(fused.size(0), fused.size(1), -1)

        fused = fused + modality_emb

        return fused

    def _normalize_image(self, image):
        """Normalize image to standard format"""
        if isinstance(image, torch.Tensor):
            # Ensure float and proper range
            if image.dtype != torch.float32:
                image = image.float()
            if image.max() > 1.0:
                image = image / 255.0

            # Ensure channel-first format (C, H, W)
            if image.dim() == 3 and image.shape[0] not in [1, 3]:
                image = image.permute(2, 0, 1)

            # Convert grayscale to RGB if needed
            if image.size(0) == 1:
                image = image.repeat(3, 1, 1)

            return image
        else:
            # Handle PIL images, numpy arrays, etc.
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            return transform(image)

    def _normalize_audio(self, audio):
        """Normalize audio to standard format"""
        if isinstance(audio, torch.Tensor):
            # Ensure float and proper range
            if audio.dtype != torch.float32:
                audio = audio.float()

            # Normalize amplitude
            if audio.abs().max() > 1.0:
                audio = audio / audio.abs().max()

            return audio
        else:
            # Handle numpy arrays, librosa outputs, etc.
            if hasattr(audio, 'numpy'):
                audio = audio.numpy()
            return torch.from_numpy(audio).float()

    def grow(self, new_hidden_dim):
        """Grow multimodal processor to accommodate model evolution"""
        if new_hidden_dim <= self.hidden_dim:
            return False

        old_dim = self.hidden_dim

        # Growth strategy: expand final layers while preserving learned features

        # Grow image encoder
        new_image_final = nn.Sequential(
            nn.Linear(512, new_hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(new_hidden_dim * 2, new_hidden_dim),
            nn.LayerNorm(new_hidden_dim)
        ).to(next(self.image_encoder.parameters()).device)

        # Transfer weights from old final layers
        with torch.no_grad():
            # Copy what we can from the old layers
            old_final_layers = list(self.image_encoder.children())[-4:]  # Last 4 layers
            new_final_layers = list(new_image_final.children())

            # Transfer Linear layer weights with expansion
            old_linear1 = old_final_layers[0]  # nn.Linear(512, old_hidden_dim * 2)
            new_linear1 = new_final_layers[0]  # nn.Linear(512, new_hidden_dim * 2)

            new_linear1.weight[:old_dim * 2, :].copy_(old_linear1.weight)
            new_linear1.bias[:old_dim * 2].copy_(old_linear1.bias)

            # Initialize new weights
            nn.init.xavier_uniform_(new_linear1.weight[old_dim * 2:, :])
            nn.init.zeros_(new_linear1.bias[old_dim * 2:])

            # Transfer final projection layer
            old_linear2 = old_final_layers[2]  # nn.Linear(old_hidden_dim * 2, old_hidden_dim)
            new_linear2 = new_final_layers[2]  # nn.Linear(new_hidden_dim * 2, new_hidden_dim)

            new_linear2.weight[:old_dim, :old_dim * 2].copy_(old_linear2.weight)
            new_linear2.bias[:old_dim].copy_(old_linear2.bias)

            # Initialize new dimensions
            nn.init.xavier_uniform_(new_linear2.weight[old_dim:, :])
            nn.init.xavier_uniform_(new_linear2.weight[:, old_dim * 2:])
            nn.init.zeros_(new_linear2.bias[old_dim:])

            # Transfer LayerNorm
            old_norm = old_final_layers[3]  # LayerNorm(old_hidden_dim)
            new_norm = new_final_layers[3]  # LayerNorm(new_hidden_dim)

            new_norm.weight[:old_dim].copy_(old_norm.weight)
            new_norm.bias[:old_dim].copy_(old_norm.bias)
            new_norm.weight[old_dim:].fill_(1.0)
            new_norm.bias[old_dim:].zero_()

        # Replace final layers of image encoder
        self.image_encoder = nn.Sequential(
            *list(self.image_encoder.children())[:-4],
            *new_image_final
        )

        # Similar growth for audio encoder
        new_audio_final = nn.Sequential(
            nn.Linear(128 * 8 * 16, new_hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(new_hidden_dim * 2, new_hidden_dim),
            nn.LayerNorm(new_hidden_dim)
        ).to(next(self.audio_encoder.parameters()).device)

        # Transfer audio encoder weights (similar process as image)
        with torch.no_grad():
            old_audio_final = list(self.audio_encoder.children())[-4:]
            new_audio_final_layers = list(new_audio_final.children())

            # Transfer and expand weights
            old_linear1 = old_audio_final[0]
            new_linear1 = new_audio_final_layers[0]

            new_linear1.weight[:old_dim * 2, :].copy_(old_linear1.weight)
            new_linear1.bias[:old_dim * 2].copy_(old_linear1.bias)
            nn.init.xavier_uniform_(new_linear1.weight[old_dim * 2:, :])
            nn.init.zeros_(new_linear1.bias[old_dim * 2:])

            old_linear2 = old_audio_final[2]
            new_linear2 = new_audio_final_layers[2]

            new_linear2.weight[:old_dim, :old_dim * 2].copy_(old_linear2.weight)
            new_linear2.bias[:old_dim].copy_(old_linear2.bias)
            nn.init.xavier_uniform_(new_linear2.weight[old_dim:, :])
            nn.init.xavier_uniform_(new_linear2.weight[:, old_dim * 2:])
            nn.init.zeros_(new_linear2.bias[old_dim:])

            old_norm = old_audio_final[3]
            new_norm = new_audio_final_layers[3]

            new_norm.weight[:old_dim].copy_(old_norm.weight)
            new_norm.bias[:old_dim].copy_(old_norm.bias)
            new_norm.weight[old_dim:].fill_(1.0)
            new_norm.bias[old_dim:].zero_()

        # Replace audio encoder final layers
        self.audio_encoder = nn.Sequential(
            *list(self.audio_encoder.children())[:-4],
            *new_audio_final
        )

        # Grow video encoder (similar process)
        new_video_final = nn.Sequential(
            nn.Linear(128 * 4 * 4, new_hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(new_hidden_dim * 2, new_hidden_dim),
            nn.LayerNorm(new_hidden_dim)
        ).to(next(self.video_encoder.parameters()).device)

        with torch.no_grad():
            old_video_final = list(self.video_encoder.children())[-4:]
            new_video_final_layers = list(new_video_final.children())

            # Similar weight transfer process as above
            self._transfer_final_layers(old_video_final, new_video_final_layers, old_dim, new_hidden_dim)

        self.video_encoder = nn.Sequential(
            *list(self.video_encoder.children())[:-4],
            *new_video_final
        )

        # Grow sensor encoder
        new_sensor_encoder = nn.Sequential(
            nn.Linear(self.config.sensor_dim if hasattr(self.config, 'sensor_dim') else 32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, new_hidden_dim),
            nn.LayerNorm(new_hidden_dim)
        ).to(next(self.sensor_encoder.parameters()).device)

        with torch.no_grad():
            # Transfer sensor encoder weights
            old_layers = list(self.sensor_encoder.children())
            new_layers = list(new_sensor_encoder.children())

            # Copy first layers exactly
            for i in range(min(len(old_layers) - 2, len(new_layers) - 2)):
                if hasattr(old_layers[i], 'weight'):
                    new_layers[i].weight.copy_(old_layers[i].weight)
                    if hasattr(old_layers[i], 'bias') and old_layers[i].bias is not None:
                        new_layers[i].bias.copy_(old_layers[i].bias)

            # Handle final layer expansion
            old_final = old_layers[-2]  # Last Linear layer
            new_final = new_layers[-2]

            new_final.weight[:old_dim, :].copy_(old_final.weight)
            new_final.bias[:old_dim].copy_(old_final.bias)
            nn.init.xavier_uniform_(new_final.weight[old_dim:, :])
            nn.init.zeros_(new_final.bias[old_dim:])

            # Handle LayerNorm
            old_norm = old_layers[-1]
            new_norm = new_layers[-1]

            new_norm.weight[:old_dim].copy_(old_norm.weight)
            new_norm.bias[:old_dim].copy_(old_norm.bias)
            new_norm.weight[old_dim:].fill_(1.0)
            new_norm.bias[old_dim:].zero_()

        self.sensor_encoder = new_sensor_encoder

        # Grow modality embeddings
        new_modality_embeddings = nn.Embedding(6, new_hidden_dim).to(self.modality_embeddings.weight.device)
        with torch.no_grad():
            new_modality_embeddings.weight[:, :old_dim].copy_(self.modality_embeddings.weight)
            nn.init.normal_(new_modality_embeddings.weight[:, old_dim:], std=0.02)
        self.modality_embeddings = new_modality_embeddings

        # Grow fusion components
        if hasattr(self.fusion, 'grow'):
            self.fusion.grow(new_hidden_dim)

        # Grow temporal transformer
        new_temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=new_hidden_dim,
                nhead=8,
                dim_feedforward=new_hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        ).to(next(self.temporal_transformer.parameters()).device)

        # Transfer transformer weights would be complex, so we reinitialize
        # In practice, the temporal modeling can be retrained quickly
        self.temporal_transformer = new_temporal_transformer

        # Grow cross-modal aligner
        new_aligner = nn.Sequential(
            nn.Linear(new_hidden_dim, new_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(new_hidden_dim // 2, new_hidden_dim),
            nn.LayerNorm(new_hidden_dim)
        ).to(next(self.cross_modal_aligner.parameters()).device)

        with torch.no_grad():
            # Transfer aligner weights
            old_aligner_layers = list(self.cross_modal_aligner.children())
            new_aligner_layers = list(new_aligner.children())

            # First layer: input expands, output expands
            old_linear1 = old_aligner_layers[0]
            new_linear1 = new_aligner_layers[0]

            new_linear1.weight[:old_dim // 2, :old_dim].copy_(old_linear1.weight)
            new_linear1.bias[:old_dim // 2].copy_(old_linear1.bias)
            nn.init.xavier_uniform_(new_linear1.weight[old_dim // 2:, :])
            nn.init.xavier_uniform_(new_linear1.weight[:, old_dim:])
            nn.init.zeros_(new_linear1.bias[old_dim // 2:])

            # Second layer: input expands, output expands
            old_linear2 = old_aligner_layers[2]
            new_linear2 = new_aligner_layers[2]

            new_linear2.weight[:old_dim, :old_dim // 2].copy_(old_linear2.weight)
            new_linear2.bias[:old_dim].copy_(old_linear2.bias)
            nn.init.xavier_uniform_(new_linear2.weight[old_dim:, :])
            nn.init.xavier_uniform_(new_linear2.weight[:, old_dim // 2:])
            nn.init.zeros_(new_linear2.bias[old_dim:])

            # LayerNorm
            old_norm = old_aligner_layers[3]
            new_norm = new_aligner_layers[3]

            new_norm.weight[:old_dim].copy_(old_norm.weight)
            new_norm.bias[:old_dim].copy_(old_norm.bias)
            new_norm.weight[old_dim:].fill_(1.0)
            new_norm.bias[old_dim:].zero_()

        self.cross_modal_aligner = new_aligner

        # Update hidden dimension
        self.hidden_dim = new_hidden_dim
        self.config.initial_hidden_dim = new_hidden_dim

        return True

    def _transfer_final_layers(self, old_layers, new_layers, old_dim, new_dim):
        """Helper method to transfer weights during growth"""
        # Transfer Linear layers with dimension expansion
        old_linear1 = old_layers[0]
        new_linear1 = new_layers[0]

        new_linear1.weight[:old_dim * 2, :].copy_(old_linear1.weight)
        new_linear1.bias[:old_dim * 2].copy_(old_linear1.bias)
        nn.init.xavier_uniform_(new_linear1.weight[old_dim * 2:, :])
        nn.init.zeros_(new_linear1.bias[old_dim * 2:])

        old_linear2 = old_layers[2]
        new_linear2 = new_layers[2]

        new_linear2.weight[:old_dim, :old_dim * 2].copy_(old_linear2.weight)
        new_linear2.bias[:old_dim].copy_(old_linear2.bias)
        nn.init.xavier_uniform_(new_linear2.weight[old_dim:, :])
        nn.init.xavier_uniform_(new_linear2.weight[:, old_dim * 2:])
        nn.init.zeros_(new_linear2.bias[old_dim:])

        old_norm = old_layers[3]
        new_norm = new_layers[3]

        new_norm.weight[:old_dim].copy_(old_norm.weight)
        new_norm.bias[:old_dim].copy_(old_norm.bias)
        new_norm.weight[old_dim:].fill_(1.0)
        new_norm.bias[old_dim:].zero_()


class SpectrogramLayer(nn.Module):
    """Learnable spectrogram computation for audio processing"""

    def __init__(self, n_fft=2048, hop_length=512, n_mels=128):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Learnable mel filterbank
        self.mel_scale = nn.Parameter(torch.randn(n_mels, n_fft // 2 + 1))

    def forward(self, audio):
        # Compute STFT
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True
        )

        # Magnitude spectrogram
        magnitude = torch.abs(stft)

        # Apply learnable mel filtering
        mel_spec = torch.matmul(self.mel_scale, magnitude)

        # Log compression
        mel_spec = torch.log(mel_spec + 1e-8)

        # Add channel dimension for CNN
        return mel_spec.unsqueeze(1)


class CrossModalAttentionFusion(nn.Module):
    """Cross-modal attention-based fusion mechanism"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.fusion_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, embeddings_list, confidence_weights=None):
        # Concatenate all modalities
        combined = torch.cat(embeddings_list, dim=1)

        # Apply cross-attention between modalities
        attended, _ = self.cross_attention(combined, combined, combined)
        attended = self.norm1(attended + combined)

        # Apply self-attention
        self_attended, _ = self.self_attention(attended, attended, attended)
        self_attended = self.norm2(self_attended + attended)

        # Final fusion projection
        fused = self.fusion_projection(self_attended)

        # Apply confidence weighting if provided
        if confidence_weights is not None:
            # Expand confidence weights to match sequence dimension
            weights = confidence_weights.unsqueeze(1).expand(-1, fused.size(1), -1)
            fused = fused * weights

        return fused

    def grow(self, new_hidden_dim):
        """Grow attention fusion to new dimension"""
        if new_hidden_dim <= self.hidden_dim:
            return False

        # Create new attention layers
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=new_hidden_dim,
            num_heads=8,
            batch_first=True
        ).to(next(self.cross_attention.parameters()).device)

        self.self_attention = nn.MultiheadAttention(
            embed_dim=new_hidden_dim,
            num_heads=8,
            batch_first=True
        ).to(next(self.self_attention.parameters()).device)

        # Grow normalization layers
        new_norm1 = nn.LayerNorm(new_hidden_dim).to(self.norm1.weight.device)
        new_norm2 = nn.LayerNorm(new_hidden_dim).to(self.norm2.weight.device)

        with torch.no_grad():
            # Transfer norm weights
            new_norm1.weight[:self.hidden_dim].copy_(self.norm1.weight)
            new_norm1.bias[:self.hidden_dim].copy_(self.norm1.bias)
            new_norm1.weight[self.hidden_dim:].fill_(1.0)
            new_norm1.bias[self.hidden_dim:].zero_()

            new_norm2.weight[:self.hidden_dim].copy_(self.norm2.weight)
            new_norm2.bias[:self.hidden_dim].copy_(self.norm2.bias)
            new_norm2.weight[self.hidden_dim:].fill_(1.0)
            new_norm2.bias[self.hidden_dim:].zero_()

        self.norm1 = new_norm1
        self.norm2 = new_norm2

        # Grow fusion projection
        new_fusion = nn.Sequential(
            nn.Linear(new_hidden_dim, new_hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(new_hidden_dim * 2, new_hidden_dim)
        ).to(next(self.fusion_projection.parameters()).device)

        with torch.no_grad():
            # Transfer projection weights
            old_layers = list(self.fusion_projection.children())
            new_layers = list(new_fusion.children())

            # First linear layer
            old_linear1 = old_layers[0]
            new_linear1 = new_layers[0]

            new_linear1.weight[:self.hidden_dim * 2, :self.hidden_dim].copy_(old_linear1.weight)
            new_linear1.bias[:self.hidden_dim * 2].copy_(old_linear1.bias)
            nn.init.xavier_uniform_(new_linear1.weight[self.hidden_dim * 2:, :])
            nn.init.xavier_uniform_(new_linear1.weight[:, self.hidden_dim:])
            nn.init.zeros_(new_linear1.bias[self.hidden_dim * 2:])

            # Second linear layer
            old_linear2 = old_layers[3]
            new_linear2 = new_layers[3]

            new_linear2.weight[:self.hidden_dim, :self.hidden_dim * 2].copy_(old_linear2.weight)
            new_linear2.bias[:self.hidden_dim].copy_(old_linear2.bias)
            nn.init.xavier_uniform_(new_linear2.weight[self.hidden_dim:, :])
            nn.init.xavier_uniform_(new_linear2.weight[:, self.hidden_dim * 2:])
            nn.init.zeros_(new_linear2.bias[self.hidden_dim:])

        self.fusion_projection = new_fusion
        self.hidden_dim = new_hidden_dim

        return True

class TransformerModalityFusion(nn.Module):
    """Transformer-based modality fusion with positional encodings"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=3
        )

        # Positional encoding for modality ordering
        self.position_embedding = nn.Parameter(torch.randn(10, hidden_dim))  # Support up to 10 modalities

    def forward(self, embeddings_list, modality_names):
        # Add positional encodings based on modality order
        positioned_embeddings = []
        for i, embedding in enumerate(embeddings_list):
            pos_emb = self.position_embedding[i].unsqueeze(0).unsqueeze(1).expand(
                embedding.size(0), embedding.size(1), -1
            )
            positioned_embeddings.append(embedding + pos_emb)

        # Concatenate all modalities
        combined = torch.cat(positioned_embeddings, dim=1)

        # Apply transformer
        fused = self.transformer(combined)

        return fused

class LearnedModalityFusion(nn.Module):
    """Learned weighted fusion with adaptive combination"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Learned fusion weights
        self.fusion_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Adaptive combination network
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, embeddings_list, confidence_weights=None):
        # Compute adaptive weights for each modality
        adaptive_weights = []
        for embedding in embeddings_list:
            weight = self.fusion_weights(embedding.mean(dim=1, keepdim=True))
            adaptive_weights.append(weight)

        # Normalize weights
        total_weight = torch.cat(adaptive_weights, dim=2)
        normalized_weights = F.softmax(total_weight, dim=2)

        # Weighted combination
        weighted_embeddings = []
        for i, embedding in enumerate(embeddings_list):
            weight = normalized_weights[:, :, i:i+1]
            weighted_embeddings.append(embedding * weight)

        # Combine all weighted embeddings
        combined = torch.stack(weighted_embeddings, dim=2).sum(dim=2)

        # Apply final combination network
        fused = self.combiner(combined)

        return fused

###########################################
# HIVE MIND COMPONENTS
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
                # Different handling based on model mode
                if self.model.config.unified_perception:
                    # For unified mode, handle concept prototypes
                    incoming_concepts = data.get('concepts', [])
                    incoming_experiences = data.get('experiences', [])
                    incoming_thought = data.get('thought')

                    # Process incoming concepts (different for unified mode)
                    integrated_concepts = 0
                    concept_updates = 0

                    if incoming_concepts and hasattr(self.model, "conceptual_system"):
                        # Convert concepts to format needed by unified conceptual system
                        for concept_data in incoming_concepts:
                            prototype = concept_data.get("prototype")
                            metadata = concept_data.get("metadata", {})
                            modality = concept_data.get("modality", "text")

                            if prototype and isinstance(prototype, list):
                                # Convert to tensor
                                prototype_tensor = torch.tensor(
                                    prototype,
                                    device=self.model.conceptual_system.concept_prototypes.device,
                                    dtype=torch.float
                                )

                                # Find if concept exists
                                # Check similarities with existing concepts
                                similar = self.model.conceptual_system.find_similar_concepts(
                                    prototype_tensor, top_k=1
                                )

                                if similar and similar[0][1] > 0.95:
                                    # Very similar concept exists - update
                                    concept_id = similar[0][0]

                                    # Blend prototypes
                                    with torch.no_grad():
                                        current = self.model.conceptual_system.concept_prototypes[concept_id]
                                        blended = current * 0.7 + prototype_tensor * 0.3
                                        self.model.conceptual_system.concept_prototypes[concept_id] = F.normalize(blended, dim=0)

                                    concept_updates += 1
                                else:
                                    # Create new concept
                                    concept_id = self.model.conceptual_system._consider_new_concept(
                                        prototype_tensor, modality=modality
                                    )

                                    if concept_id is not None:
                                        # Update metadata
                                        self.model.conceptual_system.concept_metadata[concept_id].update(metadata)
                                        integrated_concepts += 1
                else:
                    # Traditional mode
                    incoming_concepts = data.get('concepts', [])
                    incoming_experiences = data.get('experiences', [])
                    incoming_thought = data.get('thought')

                    # Process incoming concepts
                    integrated_concepts = 0
                    concept_updates = 0

                    if incoming_concepts and hasattr(self.model, "concept_bank"):
                        integrated_concepts, concept_updates = self.model.concept_bank.integrate_hive_concepts(
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
                # Get hive concepts to send back - different for unified vs traditional
                hive_concepts = []

                if self.model.config.unified_perception:
                    # Unified mode - prepare concepts from conceptual system
                    if hasattr(self.model, "conceptual_system"):
                        # Get a sample of concepts to share
                        concepts_to_share = []

                        # Select random concepts from each modality
                        for modality, concepts in self.model.conceptual_system.modality_concepts.items():
                            concept_list = list(concepts)
                            if concept_list:
                                # Get up to 10 random concepts
                                sample_size = min(10, len(concept_list))
                                sample = random.sample(concept_list, sample_size)
                                concepts_to_share.extend(sample)

                        # Prepare concept data
                        for concept_id in concepts_to_share:
                            if concept_id < self.model.conceptual_system.next_concept_id:
                                prototype = self.model.conceptual_system.concept_prototypes[concept_id].cpu().tolist()
                                metadata = self.model.conceptual_system.concept_metadata.get(concept_id, {})

                                concept_data = {
                                    "prototype": prototype,
                                    "metadata": metadata,
                                    "concept_id": concept_id,
                                    "modality": metadata.get("modality", "text")
                                }

                                hive_concepts.append(concept_data)
                else:
                    # Traditional mode
                    if hasattr(self.model, "concept_bank"):
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

                        hive_concepts = unique_concepts

                # Get hive experiences
                hive_experiences = self.model.experience_manager.get_experiences_for_sync(limit=20)

                # Get thought state
                shared_thought = self.model.thought_state.get_shared_thought()

                # Prepare response
                response = {
                    'status': 'success',
                    'timestamp': time.time(),
                    'concepts': hive_concepts[:self.config.hive_sync_concept_limit],
                    'experiences': hive_experiences,
                    'thought': shared_thought.tolist() if shared_thought is not None else None,
                    'connected_instances': len(self.connected_instances),
                    'sync_stats': {
                        'integrated_concepts': integrated_concepts,
                        'concept_updates': concept_updates,
                        'integrated_experiences': integrated_experiences
                    }
                }

                # Compress large responses
                if len(str(response)) > 10000:
                    # Use simpler response for large payloads
                    response = {
                        'status': 'success',
                        'timestamp': time.time(),
                        'concepts': hive_concepts[:min(100, len(hive_concepts))],
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
            # Prepare data to send - different for unified vs traditional
            if self.model.config.unified_perception:
                # Unified mode - prepare concepts from conceptual system
                concepts = []

                if hasattr(self.model, "conceptual_system"):
                    # Get a sample of concepts to share
                    concepts_to_share = []

                    # Select random concepts from each modality
                    for modality, concept_set in self.model.conceptual_system.modality_concepts.items():
                        concept_list = list(concept_set)
                        if concept_list:
                            # Get up to 10 random concepts
                            sample_size = min(10, len(concept_list))
                            sample = random.sample(concept_list, sample_size)
                            concepts_to_share.extend(sample)

                    # Prepare concept data
                    for concept_id in concepts_to_share:
                        if concept_id < self.model.conceptual_system.next_concept_id:
                            prototype = self.model.conceptual_system.concept_prototypes[concept_id].cpu().tolist()
                            metadata = self.model.conceptual_system.concept_metadata.get(concept_id, {})

                            concept_data = {
                                "prototype": prototype,
                                "metadata": metadata,
                                "concept_id": concept_id,
                                "modality": metadata.get("modality", "text")
                            }

                            concepts.append(concept_data)
            else:
                # Traditional mode
                if hasattr(self.model, "concept_bank"):
                    concepts = self.model.concept_bank.get_concepts_for_sync(
                        limit=self.config.hive_sync_concept_limit
                    )

            # Get experiences to share (same for both modes)
            experiences = self.model.experience_manager.get_experiences_for_sync(limit=20)

            # Get thought state to share
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

            # Different processing for unified vs traditional
            if self.model.config.unified_perception:
                # Process concepts in unified mode
                if 'concepts' in data and hasattr(self.model, "conceptual_system"):
                    # Mark concepts as synced
                    for c in concepts:
                        concept_id = c.get("concept_id")
                        if concept_id in self.model.conceptual_system.concept_metadata:
                            # Mark as synced in some way (no direct tracking in unified mode)
                            pass

                    # Integrate received concepts
                    integrated = 0
                    updated = 0

                    for concept_data in data.get('concepts', []):
                        prototype = concept_data.get("prototype")
                        metadata = concept_data.get("metadata", {})
                        modality = concept_data.get("modality", "text")

                        if prototype and isinstance(prototype, list):
                            # Convert to tensor
                            prototype_tensor = torch.tensor(
                                prototype,
                                device=self.model.conceptual_system.concept_prototypes.device,
                                dtype=torch.float
                            )

                            # Find if concept exists
                            similar = self.model.conceptual_system.find_similar_concepts(
                                prototype_tensor, top_k=1
                            )

                            if similar and similar[0][1] > 0.95:
                                # Very similar concept exists - update
                                concept_id = similar[0][0]

                                # Blend prototypes
                                with torch.no_grad():
                                    current = self.model.conceptual_system.concept_prototypes[concept_id]
                                    blended = current * 0.7 + prototype_tensor * 0.3
                                    self.model.conceptual_system.concept_prototypes[concept_id] = F.normalize(blended, dim=0)

                                updated += 1
                            else:
                                # Create new concept
                                concept_id = self.model.conceptual_system._consider_new_concept(
                                    prototype_tensor, modality=modality
                                )

                                if concept_id is not None:
                                    # Update metadata
                                    self.model.conceptual_system.concept_metadata[concept_id].update(metadata)
                                    integrated += 1

                    logger.info(f"Sync: Integrated {integrated} concepts, updated {updated}")
            else:
                # Traditional mode
                if hasattr(self.model, "concept_bank"):
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
# RUNTIME FUNCTIONS
###########################################

def create_sam_model(config_overrides=None, load_vocab=True, hive_mind=True, multimodal=False, unified_perception=False):
    """Create a new SAM instance with the given configuration overrides"""
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

    # Set unified perception mode
    config.unified_perception = unified_perception

    # Create model
    model = SAM(config)

    # Initialize with default vocabulary if requested (only for traditional mode)
    if load_vocab and not unified_perception:
        model.load_sam_vocabulary()

    return model, config


def run_sam(config=None, load_path=None, hive_config=None, multimodal=False, unified_perception=False):
    """Create and run a SAM instance"""
    # Load existing model or create new one
    if load_path and os.path.exists(load_path):
        model = SAM.load(load_path)
        logger.info(f"Loaded SAM from {load_path}")
    else:
        if hive_config:
            config_overrides = vars(config) if config else {}
            config_overrides.update(hive_config)
            model, _ = create_sam_model(
                config_overrides,
                hive_mind=True,
                multimodal=multimodal,
                unified_perception=unified_perception
            )
        else:
            model, _ = create_sam_model(
                vars(config) if config else None,
                multimodal=multimodal,
                unified_perception=unified_perception
            )
        logger.info(f"Created new SAM with {sum(p.numel() for p in model.parameters())} parameters")

    # Start background services
    model.start_services()

    # Simple interactive loop
    print("\nSAM is ready for interaction. Type 'exit' to quit.")
    print("Special commands: 'save', 'dream', 'stats', 'evolve', 'hive', 'private', 'modality [text|image|audio|multimodal]'")

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
                print(f"  Unified perception: {status['config']['unified_perception']}")

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
                    if hasattr(model, "segmentation") and hasattr(model.segmentation, "set_modality"):
                        model.segmentation.set_modality(requested_modality)
                    print(f"\nSAM: Switched to {requested_modality} modality.")
                else:
                    print(f"\nSAM: Unknown modality '{requested_modality}'. Available modalities: text, image, audio, multimodal")
                continue

            # Record in history
            history.append({"role": "user", "content": user_input})

            # Process and generate
            # Add context from history for responses
            context = ""
            if len(history) > 1 and model.config.communication_style == "Adaptive":
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

###########################################
# AUTONOMOUS EVOLUTION COMPONENTS
###########################################

class SelfEvolutionEngine:
    """Coordinates self-training and evolution processes for SAM"""

    def __init__(self, model, config=None, checkpoint_dir=None):
        """
        Initialize the self-evolution engine

        Args:
            model: The SAM model instance
            config: Configuration for self-evolution
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.config = config or model.config
        self.checkpoint_dir = checkpoint_dir or os.path.join(self.config.save_dir, "evolution_checkpoints")

        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize components
        self.task_generator = TaskGenerator(model)
        self.reasoning_engine = ReasoningEngine(model)
        self.verification_mechanism = VerificationMechanism(model)
        self.benchmark_manager = BenchmarkManager(model)
        self.knowledge_acquisition = KnowledgeAcquisition(model)

        # Evolution state
        self.evolution_step = 0
        self.evolution_history = []
        self.evolution_thread = None
        self.stop_evolution = threading.Event()
        self.evolution_active = False

        # Performance metrics
        self.performance_metrics = {
            "task_complexity": 0,
            "reasoning_depth": 0,
            "success_rate": 0.0,
            "benchmark_scores": {},
            "knowledge_breadth": 0
        }

    def start_evolution(self, interval_minutes=15):
        """Start autonomous evolution process in background thread"""
        if self.evolution_active:
            return False

        self.stop_evolution.clear()
        self.evolution_active = True

        def evolution_loop():
            while not self.stop_evolution.is_set():
                try:
                    # Run one evolution step
                    start_time = time.time()
                    results = self.evolve_step()
                    duration = time.time() - start_time

                    logger.info(f"Evolution step {self.evolution_step} completed in {duration:.2f}s. " +
                               f"Success rate: {results['success_rate']:.2f}")

                    # Save checkpoint periodically (every 10 steps)
                    if self.evolution_step % 10 == 0:
                        self.save_checkpoint()

                    # Sleep between steps
                    sleep_duration = max(1, int(interval_minutes * 60 - duration))
                    for _ in range(sleep_duration):
                        if self.stop_evolution.is_set():
                            break
                        time.sleep(1)

                except Exception as e:
                    logger.error(f"Error in evolution loop: {e}", exc_info=True)
                    time.sleep(60)  # Sleep longer after error

        self.evolution_thread = threading.Thread(target=evolution_loop)
        self.evolution_thread.daemon = True
        self.evolution_thread.start()

        logger.info(f"Started autonomous evolution with {interval_minutes} minute interval")
        return True

    def stop_evolution(self):
        """Stop autonomous evolution process"""
        if not self.evolution_active:
            return False

        self.stop_evolution.set()
        if self.evolution_thread:
            self.evolution_thread.join(timeout=30)

        self.evolution_active = False
        logger.info("Stopped autonomous evolution")
        return True

    def evolve_step(self, tasks_per_step=5):
        """Execute one step of the evolution process"""
        # Generate tasks based on current capabilities
        tasks = self.task_generator.generate_tasks(count=tasks_per_step)

        results = {
            "tasks": [],
            "success_count": 0,
            "success_rate": 0.0,
            "complexity_gain": 0,
            "knowledge_gain": 0
        }

        # For each task, apply the full cognitive loop
        for task in tasks:
            # Apply reasoning engines
            solution_candidates = self.reasoning_engine.solve_task(task)

            # Verify solutions
            verification_results = self.verification_mechanism.verify_solutions(
                task, solution_candidates)

            # Select best solution
            best_solution = verification_results["best_solution"]
            is_success = verification_results["is_valid"]

            # Record experience
            self.model.experience_manager.record_experience(
                "evolution",
                {
                    "task": task,
                    "solution": best_solution,
                    "success": is_success,
                    "step": self.evolution_step
                },
                metadata={
                    "complexity": task.get("complexity", 1),
                    "task_type": task.get("type", "unknown")
                }
            )

            # Update results
            task_result = {
                "task": task,
                "success": is_success,
                "solution": best_solution,
                "verification": verification_results
            }
            results["tasks"].append(task_result)

            if is_success:
                results["success_count"] += 1

        # Calculate success rate
        if tasks:
            results["success_rate"] = results["success_count"] / len(tasks)

        # Run internal benchmarks periodically (every 5 steps)
        if self.evolution_step % 5 == 0:
            benchmark_results = self.benchmark_manager.run_benchmarks()
            results["benchmark_results"] = benchmark_results

        # Acquire new knowledge based on performance
        if self.evolution_step % 3 == 0:
            acquisition_results = self.knowledge_acquisition.acquire_knowledge(
                focus_areas=self._determine_focus_areas(results))
            results["knowledge_acquisition"] = acquisition_results
            results["knowledge_gain"] = acquisition_results.get("concepts_added", 0)

        # Evolve model architecture if needed
        self._evolve_architecture(results)

        # Update metrics
        self._update_performance_metrics(results)

        # Update history
        self.evolution_history.append({
            "step": self.evolution_step,
            "timestamp": time.time(),
            "success_rate": results["success_rate"],
            "tasks_count": len(tasks),
            "metrics": copy.deepcopy(self.performance_metrics)
        })

        # Increment step counter
        self.evolution_step += 1

        return results

    def _determine_focus_areas(self, results):
        """Determine areas to focus knowledge acquisition based on performance"""
        # Analyze task failures to identify knowledge gaps
        focus_areas = []

        # Check task results for patterns in failures
        failure_tasks = [t for t in results["tasks"] if not t["success"]]

        # Group failures by task type
        failure_types = Counter([t["task"].get("type", "unknown") for t in failure_tasks])

        # Add most common failure types as focus areas
        for task_type, count in failure_types.most_common(3):
            if count >= 2:  # Need at least 2 failures of same type to consider it a focus area
                focus_areas.append(task_type)

        # Add general areas if not enough specific ones
        if len(focus_areas) < 2:
            focus_areas.extend(["coding", "reasoning", "knowledge"])
            focus_areas = focus_areas[:3]  # Limit to 3 areas

        return focus_areas

    def _evolve_architecture(self, results):
        """Evolve the model architecture based on performance"""
        # Determine if architecture evolution is needed
        success_rate = results["success_rate"]
        complexity = self.performance_metrics["task_complexity"]

        # Evolve if consistently successful on complex tasks
        if success_rate > 0.7 and complexity > self.model.layers[0].hidden_dim / 100:
            logger.info("High performance detected. Evolving model architecture...")

            # Call model's evolve method to grow capacity
            evolution_result = self.model.evolve()

            if evolution_result:
                logger.info(f"Architecture evolved. New dimensions: " +
                          f"width={self.model.layers[0].hidden_dim}, " +
                          f"depth={len(self.model.layers)}")

                # Record evolution event
                self.evolution_history[-1]["architecture_evolution"] = {
                    "width": self.model.layers[0].hidden_dim,
                    "depth": len(self.model.layers)
                }

    def _update_performance_metrics(self, results):
        """Update performance metrics based on evolution results"""
        # Update task complexity - average complexity of successful tasks
        successful_tasks = [t for t in results["tasks"] if t["success"]]
        if successful_tasks:
            avg_complexity = sum(t["task"].get("complexity", 1) for t in successful_tasks) / len(successful_tasks)
            # Exponential moving average for stability
            self.performance_metrics["task_complexity"] = (
                0.8 * self.performance_metrics["task_complexity"] + 0.2 * avg_complexity
            )

        # Update reasoning depth - from reasoning engine
        reasoning_depths = self.reasoning_engine.get_reasoning_stats().get("average_depth", 0)
        self.performance_metrics["reasoning_depth"] = (
            0.8 * self.performance_metrics["reasoning_depth"] + 0.2 * reasoning_depths
        )

        # Update success rate - exponential moving average
        self.performance_metrics["success_rate"] = (
            0.9 * self.performance_metrics["success_rate"] + 0.1 * results["success_rate"]
        )

        # Update benchmark scores if available
        if "benchmark_results" in results:
            for benchmark, score in results["benchmark_results"].items():
                if benchmark in self.performance_metrics["benchmark_scores"]:
                    # Exponential moving average
                    self.performance_metrics["benchmark_scores"][benchmark] = (
                        0.8 * self.performance_metrics["benchmark_scores"][benchmark] + 0.2 * score
                    )
                else:
                    self.performance_metrics["benchmark_scores"][benchmark] = score

        # Update knowledge breadth - count of unique concepts used
        if hasattr(self.model, "concept_bank"):
            self.performance_metrics["knowledge_breadth"] = self.model.concept_bank.next_concept_id

    def save_checkpoint(self):
        """Save evolution checkpoint"""
        try:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, f"evolution_step_{self.evolution_step}")
            os.makedirs(checkpoint_path, exist_ok=True)

            # Save evolution state
            evolution_state = {
                "step": self.evolution_step,
                "history": self.evolution_history[-100:],  # Last 100 entries
                "metrics": self.performance_metrics,
                "timestamp": time.time()
            }

            with open(os.path.join(checkpoint_path, "evolution_state.json"), "w") as f:
                json.dump(evolution_state, f, indent=2)

            # Save model
            self.model.save(checkpoint_path)

            logger.info(f"Evolution checkpoint saved at step {self.evolution_step}")
            return checkpoint_path

        except Exception as e:
            logger.error(f"Error saving evolution checkpoint: {e}")
            return None

    def load_checkpoint(self, checkpoint_path):
        """Load evolution checkpoint"""
        try:
            # Load evolution state
            with open(os.path.join(checkpoint_path, "evolution_state.json"), "r") as f:
                evolution_state = json.load(f)

            self.evolution_step = evolution_state["step"]
            self.evolution_history = evolution_state["history"]
            self.performance_metrics = evolution_state["metrics"]

            logger.info(f"Evolution state loaded from step {self.evolution_step}")
            return True

        except Exception as e:
            logger.error(f"Error loading evolution state: {e}")
            return False

    def get_evolution_status(self):
        """Get status of evolution progress"""
        return {
            "active": self.evolution_active,
            "step": self.evolution_step,
            "performance_metrics": self.performance_metrics,
            "history_length": len(self.evolution_history),
            "last_updated": time.time()
        }


class TaskGenerator:
    """Generates self-directed learning tasks for SAM"""

    def __init__(self, model):
        """Initialize the task generator"""
        self.model = model
        self.task_history = []
        self.task_types = [
            "code_completion", "code_generation", "function_inference",
            "math_operation", "math_equation", "logic_puzzle",
            "pattern_recognition", "sequence_completion", "concept_reasoning"
        ]

        # Task templates
        self.templates = {
            "code_completion": {
                "template": "def {function_name}({params}):\n    {partial_body}",
                "params": ["function_name", "params", "partial_body"],
                "complexity_factors": ["num_params", "body_length", "control_structures"]
            },
            "function_inference": {
                "template": "# Function takes inputs: {inputs}\n# And produces output: {output}\n# What is the function definition?",
                "params": ["inputs", "output"],
                "complexity_factors": ["input_complexity", "transformation_complexity"]
            },
            "math_operation": {
                "template": "Calculate: {expression}",
                "params": ["expression"],
                "complexity_factors": ["num_operations", "operation_types"]
            },
            # Add more templates for other task types...
        }

        # Complexity progression
        self.current_complexity = 1.0
        self.complexity_growth_rate = 0.05  # 5% increase per successful step

        # Track successful tasks to inform future generation
        self.successful_task_patterns = Counter()

    def generate_tasks(self, count=5):
        """Generate a batch of self-directed tasks"""
        tasks = []

        # Get model's current capabilities
        capabilities = self._assess_model_capabilities()

        # Adjust complexity based on capabilities
        self._adjust_complexity(capabilities)

        # Select task types with preference for those matching capabilities
        selected_types = self._select_task_types(capabilities, count)

        # Generate each task
        for task_type in selected_types:
            task = self._generate_task(task_type, self.current_complexity)
            tasks.append(task)

            # Track in history
            self.task_history.append({
                "task_id": task["id"],
                "type": task["type"],
                "complexity": task["complexity"],
                "timestamp": time.time()
            })

        return tasks

    def _assess_model_capabilities(self):
        """Assess current model capabilities to inform task generation"""
        capabilities = {
            "concept_count": 0,
            "pattern_complexity": 0,
            "reasoning_depth": 0,
            "success_rate": 0.0,
            "strength_areas": [],
            "weakness_areas": []
        }

        # Get concept stats
        if hasattr(self.model, "concept_bank"):
            concept_stats = self.model.concept_bank.get_concept_stats()
            capabilities["concept_count"] = concept_stats["total_concepts"]

        # Get recent experiences to determine areas of strength/weakness
        if hasattr(self.model, "experience_manager"):
            recent_experiences = self.model.experience_manager.get_experiences_by_type(
                "evolution", limit=50)

            if recent_experiences:
                # Calculate success rates by task type
                success_by_type = defaultdict(lambda: {"success": 0, "total": 0})

                for exp in recent_experiences:
                    task_type = exp.get("metadata", {}).get("task_type", "unknown")
                    success = exp.get("content", {}).get("success", False)

                    success_by_type[task_type]["total"] += 1
                    if success:
                        success_by_type[task_type]["success"] += 1

                # Calculate overall success rate
                total_success = sum(data["success"] for data in success_by_type.values())
                total_tasks = sum(data["total"] for data in success_by_type.values())

                if total_tasks > 0:
                    capabilities["success_rate"] = total_success / total_tasks

                # Determine strengths and weaknesses
                for task_type, data in success_by_type.items():
                    if data["total"] >= 3:  # Need minimum samples
                        rate = data["success"] / data["total"]
                        if rate >= 0.7:
                            capabilities["strength_areas"].append(task_type)
                        elif rate <= 0.3:
                            capabilities["weakness_areas"].append(task_type)

        # Estimate reasoning depth from thought state
        if hasattr(self.model, "thought_state"):
            capabilities["reasoning_depth"] = self.model.thought_state.thought_depth

        return capabilities

    def _adjust_complexity(self, capabilities):
        """Adjust task complexity based on model capabilities"""
        # Base complexity on success rate
        target_complexity = 1.0 + 4.0 * capabilities["success_rate"]

        # Increase based on concept count (knowledge breadth)
        # This causes complexity to grow with model's knowledge
        knowledge_factor = min(3.0, np.log10(max(100, capabilities["concept_count"])) - 1)
        target_complexity += knowledge_factor

        # Increase based on reasoning depth
        reasoning_factor = min(2.0, capabilities["reasoning_depth"] / 4)
        target_complexity += reasoning_factor

        # Smoothly approach target complexity
        self.current_complexity = 0.9 * self.current_complexity + 0.1 * target_complexity

        # Ensure minimum complexity
        self.current_complexity = max(1.0, self.current_complexity)

    def _select_task_types(self, capabilities, count):
        """Select task types based on model capabilities"""
        task_weights = {}

        # Base weights on task type
        for task_type in self.task_types:
            task_weights[task_type] = 1.0

        # Increase weight for weakness areas to focus on improvement
        for area in capabilities["weakness_areas"]:
            if area in task_weights:
                task_weights[area] *= 2.0

        # Also include some strength areas for reinforcement
        for area in capabilities["strength_areas"][:2]:  # Top 2 strengths
            if area in task_weights:
                task_weights[area] *= 1.5

        # Convert weights to probabilities
        total_weight = sum(task_weights.values())
        task_probs = {k: v / total_weight for k, v in task_weights.items()}

        # Select task types based on probabilities
        selected_types = random.choices(
            list(task_probs.keys()),
            weights=list(task_probs.values()),
            k=count
        )

        return selected_types

    def _generate_task(self, task_type, complexity):
        """Generate a specific task of given type and complexity"""
        task_id = str(uuid.uuid4())

        # Choose appropriate generator based on task type
        if task_type == "code_completion":
            task_content = self._generate_code_completion(complexity)
        elif task_type == "function_inference":
            task_content = self._generate_function_inference(complexity)
        elif task_type == "math_operation":
            task_content = self._generate_math_task(complexity)
        elif task_type == "sequence_completion":
            task_content = self._generate_sequence_task(complexity)
        elif task_type == "logic_puzzle":
            task_content = self._generate_logic_puzzle(complexity)
        else:
            # Fallback to a generic task
            task_content = self._generate_generic_task(task_type, complexity)

        # Create task dictionary
        task = {
            "id": task_id,
            "type": task_type,
            "complexity": complexity,
            "content": task_content,
            "created_at": time.time()
        }

        return task

    def _generate_code_completion(self, complexity):
        """Generate a code completion task with specified complexity"""
        # Determine language (mostly Python for now)
        language = "python"

        # Complexity determines function characteristics
        num_params = max(1, int(complexity))
        body_lines = max(2, int(complexity * 1.5))
        control_depth = max(1, int(complexity / 2))

        # Generate function signature
        function_names = ["calculate", "process", "transform", "analyze", "compute", "filter", "map", "reduce"]
        function_name = random.choice(function_names) + "_" + random.choice(["data", "values", "items", "numbers", "elements"])

        # Generate parameters
        param_types = ["list", "dict", "int", "float", "str"]
        params = []
        for i in range(num_params):
            param_type = random.choice(param_types)
            param_name = ["x", "y", "z", "data", "values", "items", "nums", "elements"][i % 8]
            if i > 0:
                param_name += str(i)
            params.append(param_name)

        param_str = ", ".join(params)

        # Generate function body elements
        body_elements = []

        # Variables
        variables = [f"result = []", f"total = 0", f"count = 0", f"output = {{}}"]
        body_elements.extend(random.sample(variables, min(2, len(variables))))

        # Control structures based on complexity
        control_structures = []
        if complexity >= 1.5:
            control_structures.append(f"for item in {params[0]}:")
        if complexity >= 2.5:
            control_structures.append(f"if item > 0:")
        if complexity >= 3.5:
            control_structures.append(f"try:\n        value = item / total\n    except ZeroDivisionError:\n        value = 0")

        body_elements.extend(control_structures[:control_depth])

        # Operations
        operations = [
            f"total += item",
            f"result.append(item * 2)",
            f"output[item] = total",
            f"count = len({params[0]})"
        ]
        body_elements.extend(random.sample(operations, min(2, len(operations))))

        # Return statement
        return_statements = ["return result", "return total", "return output", "return count"]
        body_elements.append(random.choice(return_statements))

        # Construct partial body with intentional gaps
        partial_body = []
        for i, element in enumerate(body_elements):
            # Randomly omit some elements to create the task
            if random.random() < 0.3:
                partial_body.append("# TODO: implement this part")
            else:
                partial_body.append(element)

        body_text = "\n    ".join(partial_body)

        # Create the final task
        task_text = f"def {function_name}({param_str}):\n    {body_text}"

        return {
            "language": language,
            "code": task_text,
            "instruction": "Complete the function by filling in the missing parts",
            "function_name": function_name,
            "parameters": params
        }

    def _generate_function_inference(self, complexity):
        """Generate a function inference task with specified complexity"""
        # Create simple transformation functions based on complexity
        operations = [
            (lambda x: x + 1, "increment by 1"),
            (lambda x: x * 2, "multiply by 2"),
            (lambda x: x ** 2, "square"),
            (lambda x: max(0, x), "replace negatives with zero"),
            (lambda x: int(x), "convert to integer"),
            (lambda x: round(x, 1), "round to 1 decimal place"),
            (lambda x: x % 5, "modulo 5"),
            (lambda x: 1 if x > 0 else -1, "sign function")
        ]

        # Select operations based on complexity
        num_operations = max(1, min(4, int(complexity)))
        selected_ops = random.sample(operations, num_operations)

        # Create composite function
        def apply_operations(x):
            result = x
            for op, _ in selected_ops:
                result = op(result)
            return result

        # Generate input-output pairs
        num_examples = max(3, min(8, int(complexity * 2)))
        inputs = [random.randint(-10, 10) for _ in range(num_examples)]
        outputs = [apply_operations(x) for x in inputs]

        # Create task description
        operation_descriptions = [desc for _, desc in selected_ops]

        return {
            "inputs": inputs,
            "outputs": outputs,
            "instruction": "Determine the function that transforms the inputs into the outputs",
            "input_output_pairs": list(zip(inputs, outputs)),
            "operations": operation_descriptions  # Hidden from task, for verification
        }

    def _generate_math_task(self, complexity):
        """Generate a math task with specified complexity"""
        # Determine number and type of operations based on complexity
        num_operations = max(1, min(6, int(complexity * 1.5)))

        # Available operations, with weights
        operations = [
            ("+", 1.0),
            ("-", 1.0),
            ("*", 0.8),
            ("/", 0.5),
            ("**", 0.3)  # Exponentiation, used sparingly
        ]

        # Adjust weights based on complexity
        if complexity < 2:
            # Simpler operations for low complexity
            operation_weights = [1.0, 1.0, 0.5, 0.2, 0.0]
        elif complexity < 3.5:
            # Moderate difficulty
            operation_weights = [1.0, 1.0, 0.8, 0.5, 0.2]
        else:
            # Higher difficulty with more complex operations
            operation_weights = [0.8, 0.8, 1.0, 0.8, 0.4]

        # Generate expression recursively
        def generate_expression(depth, max_depth):
            if depth == 0 or (depth < max_depth and random.random() < 0.3):
                # Generate leaf node (number)
                if random.random() < 0.7:
                    return str(random.randint(1, 10))
                else:
                    return str(random.randint(1, 100))

            # Generate internal node (operation)
            op_indices = list(range(len(operations)))
            op_index = random.choices(op_indices, weights=operation_weights, k=1)[0]
            operation, _ = operations[op_index]

            # Generate operands
            left = generate_expression(depth - 1, max_depth)
            right = generate_expression(depth - 1, max_depth)

            # Special handling for division to avoid division by zero
            if operation == "/":
                if right == "0":
                    right = str(random.randint(1, 5))
                return f"({left} {operation} {right})"
            elif operation == "**":
                # Keep exponents small
                if right not in ["2", "3"]:
                    right = str(random.randint(1, 3))
                return f"{left}{operation}{right}"
            else:
                return f"({left} {operation} {right})"

        # Generate the expression
        expression = generate_expression(num_operations, num_operations)

        # Calculate correct answer (for verification)
        try:
            answer = eval(expression)
            # Round floating point answers
            if isinstance(answer, float):
                answer = round(answer, 4)
        except:
            # Fallback if expression has issues
            expression = "(3 + 4) * 2"
            answer = 14

        return {
            "expression": expression,
            "instruction": "Calculate the result of this expression",
            "answer": answer  # Hidden from task, for verification
        }

    def _generate_sequence_task(self, complexity):
        """Generate a sequence completion task with specified complexity"""
        # Define sequence generators with varying complexity
        sequence_generators = [
            # Simple arithmetic sequences
            (lambda n: n + 1, "increment by 1", 1.0),
            (lambda n: n * 2, "double", 1.2),
            (lambda n: n ** 2, "square", 1.8),
            (lambda n: n * (n + 1) // 2, "triangular numbers", 2.2),
            (lambda n: int(0.5 + (1 + 5**0.5)**n / 5**0.5 / 2**n), "fibonacci", 2.5),
            (lambda n: 1 if n <= 1 else (sequence[-1] + sequence[-2]), "fibonacci recursive", 3.0),
            (lambda n: sum(int(d) for d in str(n)), "digit sum", 3.2),
            (lambda n: n if n <= 1 else (sequence[-1] * 2 if n % 2 == 0 else sequence[-1] + 3), "collatz-like", 3.5)
        ]

        # Filter generators by complexity
        valid_generators = [gen for gen, _, gen_complexity in sequence_generators
                           if gen_complexity <= complexity]

        if not valid_generators:
            valid_generators = [sequence_generators[0][0]]  # Fallback to simplest

        # Select a generator
        generator, description, _ = random.choice([(gen, desc, comp)
                                               for gen, desc, comp in sequence_generators
                                               if comp <= complexity])

        # Generate sequence
        sequence_length = max(5, min(10, int(complexity * 2)))

        # Some generators require special handling for recursive definitions
        if "recursive" in description:
            sequence = [0, 1]  # Start with first two Fibonacci numbers
            for i in range(2, sequence_length):
                sequence.append(sequence[i-1] + sequence[i-2])
        elif "collatz" in description:
            sequence = [random.randint(2, 5)]  # Start with a small positive number
            for i in range(1, sequence_length):
                if i % 2 == 0:
                    sequence.append(sequence[-1] * 2)
                else:
                    sequence.append(sequence[-1] + 3)
        else:
            # Use functional generator
            start = random.randint(1, 5)
            sequence = [generator(start + i) for i in range(sequence_length)]

        # Create task - hide last 2-3 elements as the task
        visible_sequence = sequence[:-3] if complexity > 2 else sequence[:-2]
        hidden_sequence = sequence[-3:] if complexity > 2 else sequence[-2:]

        return {
            "visible_sequence": visible_sequence,
            "instruction": "Continue the sequence by providing the next elements",
            "description": description,  # Hidden from task, for verification
            "expected_continuation": hidden_sequence  # Hidden from task, for verification
        }

    def _generate_logic_puzzle(self, complexity):
        """Generate a logic puzzle with specified complexity"""
        # Basic logic puzzle types
        puzzle_types = [
            "propositional_logic",
            "syllogism",
            "deduction",
            "constraint_satisfaction"
        ]

        # Select puzzle type based on complexity
        if complexity < 2:
            puzzle_type = random.choice(puzzle_types[:2])  # Simpler types
        else:
            puzzle_type = random.choice(puzzle_types)

        if puzzle_type == "propositional_logic":
            return self._generate_propositional_puzzle(complexity)
        elif puzzle_type == "syllogism":
            return self._generate_syllogism_puzzle(complexity)
        elif puzzle_type == "deduction":
            return self._generate_deduction_puzzle(complexity)
        else:  # constraint_satisfaction
            return self._generate_constraint_puzzle(complexity)

    def _generate_propositional_puzzle(self, complexity):
        """Generate a propositional logic puzzle"""
        # Define variables
        variables = ["A", "B", "C", "D", "E"]
        num_variables = min(len(variables), max(2, int(complexity)))
        used_vars = variables[:num_variables]

        # Define statements about variables
        statements = []
        conclusion = ""

        if complexity < 2:
            # Simple implication
            p, q = random.sample(used_vars, 2)
            statements.append(f"If {p} then {q}")
            statements.append(f"{p} is true")
            conclusion = f"Therefore, {q} is true"
            answer = True

        elif complexity < 3:
            # Modus tollens
            p, q = random.sample(used_vars, 2)
            statements.append(f"If {p} then {q}")
            statements.append(f"{q} is false")
            conclusion = f"Therefore, {p} is false"
            answer = True

        else:
            # More complex with multiple statements
            p, q, r = random.sample(used_vars, 3)

            if random.random() < 0.5:
                # Valid argument
                statements.append(f"If {p} then {q}")
                statements.append(f"If {q} then {r}")
                statements.append(f"{p} is true")
                conclusion = f"Therefore, {r} is true"
                answer = True
            else:
                # Invalid argument (affirming the consequent)
                statements.append(f"If {p} then {q}")
                statements.append(f"{q} is true")
                conclusion = f"Therefore, {p} is true"
                answer = False

        return {
            "statements": statements,
            "conclusion": conclusion,
            "instruction": "Determine if the conclusion logically follows from the statements",
            "answer": answer  # Hidden from task, for verification
        }

    def _generate_syllogism_puzzle(self, complexity):
        """Generate a syllogistic reasoning puzzle"""
        # Categories for syllogisms
        categories = [
            "mammals", "birds", "fish", "reptiles",
            "vehicles", "furniture", "tools", "foods",
            "Europeans", "Asians", "athletes", "musicians",
            "professors", "students", "doctors", "lawyers"
        ]

        # Properties
        properties = [
            "mortal", "intelligent", "fast", "strong",
            "tall", "heavy", "loud", "quiet",
            "valuable", "rare", "colorful", "dangerous",
            "helpful", "ethical", "efficient", "reliable"
        ]

        # Select categories and properties
        cats = random.sample(categories, 3)
        props = random.sample(properties, 2) if complexity > 2 else [random.choice(properties)]

        # Create syllogism
        statements = []
        conclusion = ""

        if random.random() < 0.7 or complexity < 2:
            # Valid syllogism
            statements.append(f"All {cats[0]} are {cats[1]}")
            statements.append(f"All {cats[1]} are {props[0]}")
            conclusion = f"Therefore, all {cats[0]} are {props[0]}"
            answer = True
        else:
            # Invalid syllogism
            if random.random() < 0.5:
                # Fallacy of the undistributed middle
                statements.append(f"All {cats[0]} are {props[0]}")
                statements.append(f"All {cats[2]} are {props[0]}")
                conclusion = f"Therefore, all {cats[0]} are {cats[2]}"
            else:
                # Affirming the consequent
                statements.append(f"All {cats[0]} are {props[0]}")
                statements.append(f"{cats[2]} is {props[0]}")
                conclusion = f"Therefore, {cats[2]} is a {cats[0]}"
            answer = False

        return {
            "statements": statements,
            "conclusion": conclusion,
            "instruction": "Determine if the conclusion logically follows from the statements",
            "answer": answer  # Hidden from task, for verification
        }

    def _generate_deduction_puzzle(self, complexity):
        """Generate a deductive reasoning puzzle"""
        # People for the puzzle
        people = ["Alice", "Bob", "Charlie", "Diana", "Ethan", "Fiona"]
        num_people = min(len(people), max(3, int(complexity)))
        used_people = people[:num_people]

        # Activities
        activities = ["swimming", "running", "reading", "painting", "cooking", "gaming"]
        num_activities = min(len(activities), num_people)
        used_activities = activities[:num_activities]

        # Create a valid assignment
        assignment = dict(zip(used_people, random.sample(used_activities, len(used_people))))

        # Generate clues
        clues = []
        num_clues = max(3, min(6, int(complexity * 2)))

        # Direct clues (person does activity)
        direct_clues = [f"{person} enjoys {assignment[person]}" for person in random.sample(used_people, min(2, len(used_people)))]

        # Negative clues (person doesn't do activity)
        negative_clues = []
        for person in used_people:
            other_activities = [a for a in used_activities if a != assignment[person]]
            if other_activities:
                negative_clues.append(f"{person} doesn't do {random.choice(other_activities)}")

        # Relationship clues (person A does activity1 if person B does activity2)
        relationship_clues = []
        for i in range(min(3, num_people - 1)):
            p1, p2 = random.sample(used_people, 2)
            relationship_clues.append(f"If {p1} does {assignment[p1]}, then {p2} does {assignment[p2]}")

        # Combine and shuffle clues
        all_clues = direct_clues + random.sample(negative_clues, min(len(negative_clues), 2))
        all_clues += random.sample(relationship_clues, min(len(relationship_clues), num_clues - len(all_clues)))
        random.shuffle(all_clues)

        # Select a person to ask about
        target_person = random.choice(used_people)
        target_activity = assignment[target_person]

        return {
            "clues": all_clues,
            "question": f"What activity does {target_person} do?",
            "instruction": "Use the clues to determine the answer",
            "answer": target_activity,  # Hidden from task, for verification
            "full_assignment": assignment  # Hidden from task, for verification
        }

    def _generate_constraint_puzzle(self, complexity):
        """Generate a constraint satisfaction puzzle"""
        # Objects to arrange
        object_types = [
            "books", "cars", "houses", "people",
            "animals", "plants", "computers", "paintings"
        ]

        # Properties to assign
        property_types = [
            "colors", "sizes", "ages", "weights",
            "prices", "ratings", "countries", "materials"
        ]

        # Select object and property types
        object_type = random.choice(object_types)
        property_type = random.choice(property_types)

        # Number of objects and properties
        num_items = max(3, min(5, int(complexity)))

        # Generate specific objects and properties
        objects = [f"{object_type[:-1]} {i+1}" for i in range(num_items)]

        # Generate specific properties
        if property_type == "colors":
            properties = random.sample(["red", "blue", "green", "yellow", "purple", "orange"], num_items)
        elif property_type == "sizes":
            properties = random.sample(["small", "medium", "large", "tiny", "huge"], num_items)
        elif property_type == "countries":
            properties = random.sample(["USA", "Japan", "France", "Brazil", "India"], num_items)
        else:
            # Generic properties
            properties = [f"{property_type[:-1]} {chr(65+i)}" for i in range(num_items)]

        # Create a valid assignment
        valid_assignment = dict(zip(objects, properties))

        # Generate constraints/clues
        clues = []
        num_clues = max(3, min(7, int(complexity * 2)))

        # Direct clues
        direct_clues = [f"The {objects[i]} has {properties[i]}" for i in range(min(2, num_items))]

        # Negative clues
        negative_clues = []
        for obj in objects:
            other_props = [p for p in properties if p != valid_assignment[obj]]
            if other_props:
                negative_clues.append(f"The {obj} doesn't have {random.choice(other_props)}")

        # Relationship clues
        relationship_clues = []
        for i in range(min(3, num_items - 1)):
            obj1, obj2 = random.sample(objects, 2)
            relationship_clues.append(f"If the {obj1} has {valid_assignment[obj1]}, then the {obj2} has {valid_assignment[obj2]}")

        # Combine and shuffle clues
        all_clues = direct_clues + random.sample(negative_clues, min(len(negative_clues), 2))
        all_clues += random.sample(relationship_clues, min(len(relationship_clues), num_clues - len(all_clues)))
        random.shuffle(all_clues)

        # Select an object to ask about
        target_object = random.choice(objects)
        target_property = valid_assignment[target_object]

        return {
            "clues": all_clues,
            "question": f"What {property_type[:-1]} does the {target_object} have?",
            "instruction": "Use the clues to determine the answer",
            "answer": target_property,  # Hidden from task, for verification
            "full_assignment": valid_assignment  # Hidden from task, for verification
        }

    def _generate_generic_task(self, task_type, complexity):
        """Generate a generic task when specific generator not available"""
        # Create a template-based task
        if task_type in self.templates:
            template_info = self.templates[task_type]

            # Fill template parameters with random values
            params = {}
            for param in template_info["params"]:
                params[param] = f"[{param}_{random.randint(1, 100)}]"

            # Create task text from template
            task_text = template_info["template"].format(**params)

            return {
                "text": task_text,
                "instruction": f"Complete the {task_type} task",
                "params": params
            }
        else:
            # Truly generic fallback
            return {
                "text": f"This is a {task_type} task with complexity {complexity}",
                "instruction": f"Solve the {task_type} problem",
                "complexity_level": int(complexity)
            }

    def record_task_result(self, task, success):
        """Record task result to inform future task generation"""
        if success:
            # Increment counter for successful task pattern
            self.successful_task_patterns[task["type"]] += 1

            # Gradually increase complexity for successful task types
            if len(self.task_history) >= 10:
                recent_tasks = [t for t in self.task_history[-10:] if t["type"] == task["type"]]
                if recent_tasks and sum(1 for t in recent_tasks if self.successful_task_patterns[t["type"]] > 0) >= 3:
                    # Multiple recent successes of this type, increase complexity
                    self.current_complexity += self.complexity_growth_rate


class ReasoningEngine:
    """Multiple reasoning approaches for solving tasks"""

    def __init__(self, model):
        """Initialize the reasoning engine"""
        self.model = model

        # Initialize reasoning streams
        self.reasoning_streams = {
            "deduction": self._deductive_reasoning,
            "abduction": self._abductive_reasoning,
            "induction": self._inductive_reasoning,
            "trial_and_error": self._trial_and_error
        }

        # Track reasoning statistics
        self.reasoning_stats = {
            "stream_usage": {stream: 0 for stream in self.reasoning_streams},
            "success_rates": {stream: [] for stream in self.reasoning_streams},
            "average_depth": 0,
            "last_updated": time.time()
        }

    def solve_task(self, task, max_solutions=3):
        """Solve a task using multiple reasoning streams"""
        # Prepare a focused thought context
        self._prepare_thought_context(task)

        # Use each reasoning stream to generate solution candidates
        solution_candidates = []

        # Determine which streams to use based on task type
        streams_to_use = self._select_reasoning_streams(task)

        # Apply each selected stream
        for stream_name in streams_to_use:
            stream_fn = self.reasoning_streams[stream_name]

            # Call the stream function to get candidates
            candidates = stream_fn(task)

            # Track usage
            self.reasoning_stats["stream_usage"][stream_name] += 1

            # Add to candidates with stream info
            for candidate in candidates:
                if isinstance(candidate, dict):
                    candidate["reasoning_stream"] = stream_name
                else:
                    # Convert to dict if it's not already
                    candidate = {
                        "solution": candidate,
                        "reasoning_stream": stream_name,
                        "confidence": 0.5  # Default confidence
                    }

                solution_candidates.append(candidate)

        # Limit number of candidates
        if len(solution_candidates) > max_solutions:
            # Prioritize by confidence if available
            solution_candidates.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            solution_candidates = solution_candidates[:max_solutions]

        return solution_candidates

    def _prepare_thought_context(self, task):
        """Prepare the model's thought state for solving the task"""
        if hasattr(self.model, "thought_state"):
            # Reset thought state to clear previous context
            self.model.thought_state.reset()

            # Create context specific to task type
            context_text = f"Task type: {task['type']}. "
            context_text += f"Complexity: {task['complexity']}. "

            if "instruction" in task:
                context_text += f"Instruction: {task['instruction']}. "

            # Add task-specific content
            if task["type"] == "code_completion" and "code" in task:
                context_text += f"Code context: {task['code']}"
            elif task["type"] == "function_inference" and "input_output_pairs" in task:
                pairs = task["input_output_pairs"]
                context_text += f"Input-output pairs: {pairs}"
            elif task["type"] == "math_operation" and "expression" in task:
                context_text += f"Expression: {task['expression']}"
            elif task["type"] == "sequence_completion" and "visible_sequence" in task:
                context_text += f"Sequence: {task['visible_sequence']}"
            elif task["type"] in ["logic_puzzle", "propositional_logic", "syllogism"] and "statements" in task:
                context_text += f"Statements: {task['statements']}. "
                if "conclusion" in task:
                    context_text += f"Conclusion: {task['conclusion']}"

            # Process text to update thought state
            if hasattr(self.model, "process_text"):
                self.model.process_text(context_text, private_context=True)

    def _select_reasoning_streams(self, task):
        """Select which reasoning streams to use based on task type"""
        task_type = task["type"]
        complexity = task["complexity"]

        # Default to using all streams
        streams = list(self.reasoning_streams.keys())

        # Prioritize streams based on task type
        if task_type in ["code_completion", "function_inference"]:
            # Prioritize deduction and trial-and-error for code tasks
            streams = ["deduction", "trial_and_error", "induction", "abduction"]

        elif task_type in ["math_operation", "sequence_completion"]:
            # Prioritize induction and deduction for pattern tasks
            streams = ["induction", "deduction", "abduction", "trial_and_error"]

        elif task_type in ["logic_puzzle", "propositional_logic", "syllogism"]:
            # Prioritize deduction for logic tasks
            streams = ["deduction", "abduction", "induction", "trial_and_error"]

        # For high complexity tasks, use all streams
        if complexity >= 3:
            return streams

        # For lower complexity, use fewer streams
        return streams[:max(1, int(complexity) + 1)]

    def _deductive_reasoning(self, task):
        """Apply deductive reasoning (from general rules to specific instances)"""
        candidates = []

        # Extract task information
        task_type = task["type"]

        # Apply deductive approach based on task type
        if task_type == "code_completion":
            candidates.append(self._deductive_code_completion(task))

        elif task_type == "function_inference":
            candidates.append(self._deductive_function_inference(task))

        elif task_type == "math_operation":
            candidates.append(self._deductive_math_operation(task))

        elif task_type == "sequence_completion":
            candidates.append(self._deductive_sequence_completion(task))

        elif task_type in ["logic_puzzle", "propositional_logic", "syllogism"]:
            candidates.append(self._deductive_logic_reasoning(task))

        else:
            # Generic deductive approach for other tasks
            candidates.append({
                "solution": f"Deductive solution for {task_type}",
                "confidence": 0.5,
                "reasoning_steps": ["Applied general rules to task specifics"]
            })

        return candidates

    def _deductive_code_completion(self, task):
        """Apply deductive reasoning to code completion tasks"""
        code = task.get("code", "")
        function_name = task.get("function_name", "")

        # Split code into lines for analysis
        code_lines = code.split("\n")

        # Extract function signature and existing body
        signature = ""
        body_lines = []

        for line in code_lines:
            if line.strip().startswith("def"):
                signature = line
            elif line.strip() and not line.strip().startswith("def"):
                body_lines.append(line)

        # Analyze signature to understand parameters
        params = []
        if "(" in signature and ")" in signature:
            param_str = signature.split("(")[1].split(")")[0]
            params = [p.strip() for p in param_str.split(",") if p.strip()]

        # Analyze existing body for patterns
        has_return = any("return" in line for line in body_lines)
        has_loops = any("for" in line or "while" in line for line in body_lines)
        has_conditionals = any("if" in line for line in body_lines)

        # Find TODO comments or gaps
        todos = [i for i, line in enumerate(body_lines) if "TODO" in line or "..." in line]

        # Generate complete code based on analysis
        completed_code = code
        reasoning_steps = []

        if todos:
            # Replace TODOs with appropriate implementations
            new_body_lines = body_lines.copy()

            for todo_idx in todos:
                if todo_idx > 0 and todo_idx < len(body_lines) - 1:
                    # Context from surrounding lines
                    prev_line = body_lines[todo_idx - 1].strip()
                    next_line = body_lines[todo_idx + 1].strip() if todo_idx + 1 < len(body_lines) else ""

                    if "for" in prev_line and not has_conditionals:
                        # Inside a loop, add conditional
                        new_body_lines[todo_idx] = "        if " + params[0] + " > 0:"
                        reasoning_steps.append("Added conditional inside loop")

                    elif prev_line.startswith("if") and not "return" in next_line:
                        # Inside conditional, add calculation
                        new_body_lines[todo_idx] = "        result.append(" + params[0] + "[i] * 2)"
                        reasoning_steps.append("Added calculation inside conditional")

                    else:
                        # Generic replacement
                        new_body_lines[todo_idx] = "    " + ("return result" if not has_return else "result = []")
                        reasoning_steps.append("Added missing variable initialization or return statement")

            # Reconstruct code
            completed_code = signature + "\n" + "\n".join(new_body_lines)

        elif not has_return:
            # Add missing return statement
            if params:
                completed_code += "\n    return " + params[0]
                reasoning_steps.append("Added missing return statement")
            else:
                completed_code += "\n    return None"
                reasoning_steps.append("Added default return None")

        # Ensure all variables are used
        for param in params:
            if param not in " ".join(body_lines):
                # Parameter not used, add simple usage
                if has_loops:
                    # Inside existing loop
                    idx = next((i for i, line in enumerate(body_lines) if "for" in line), -1)
                    if idx >= 0:
                        insert_line = f"        result.append({param})"
                        completed_body = body_lines[:idx+1] + [insert_line] + body_lines[idx+1:]
                        completed_code = signature + "\n" + "\n".join(completed_body)
                        reasoning_steps.append(f"Added usage of unused parameter {param}")
                else:
                    # No loops, add simple usage
                    completed_code = signature + "\n    result = " + param + "\n" + "\n".join(body_lines)
                    if not has_return:
                        completed_code += "\n    return result"
                    reasoning_steps.append(f"Added usage of unused parameter {param}")

        return {
            "solution": completed_code,
            "confidence": 0.75 if reasoning_steps else 0.4,
            "reasoning_steps": reasoning_steps or ["Applied code completion rules to existing structure"]
        }

    def _deductive_function_inference(self, task):
        """Apply deductive reasoning to function inference tasks"""
        input_output_pairs = task.get("input_output_pairs", [])

        if not input_output_pairs:
            return {
                "solution": "f(x) = x",  # Default if no examples
                "confidence": 0.1,
                "reasoning_steps": ["No input-output pairs provided, defaulting to identity function"]
            }

        # Start with simple transformations
        transformations = [
            (lambda x, y: y == x + 1, "f(x) = x + 1"),
            (lambda x, y: y == x - 1, "f(x) = x - 1"),
            (lambda x, y: y == x * 2, "f(x) = x * 2"),
            (lambda x, y: y == x / 2, "f(x) = x / 2"),
            (lambda x, y: y == x ** 2, "f(x) = x ** 2"),
            (lambda x, y: y == x ** 3, "f(x) = x ** 3"),
            (lambda x, y: y == abs(x), "f(x) = abs(x)"),
            (lambda x, y: y == -x, "f(x) = -x"),
            (lambda x, y: y == 1 if x > 0 else 0 if x == 0 else -1, "f(x) = 1 if x > 0 else 0 if x == 0 else -1")
        ]

        # Check each transformation
        valid_transformations = []
        for check_fn, formula in transformations:
            # Check if transformation works for all pairs
            valid = True
            for x, y in input_output_pairs:
                if not check_fn(x, y):
                    valid = False
                    break

            if valid:
                valid_transformations.append((formula, 1.0))  # High confidence for exact match

        if valid_transformations:
            # Return highest confidence transformation
            formula, confidence = valid_transformations[0]
            return {
                "solution": formula,
                "confidence": confidence,
                "reasoning_steps": ["Identified exact transformation formula",
                                   f"Verified formula works for all {len(input_output_pairs)} input-output pairs"]
            }

        # If no exact match, try to infer pattern
        differences = [y - x for x, y in input_output_pairs]
        ratios = [y / x if x != 0 else float('inf') for x, y in input_output_pairs]

        # Check for consistent difference (linear function)
        if all(abs(d - differences[0]) < 0.0001 for d in differences):
            formula = f"f(x) = x + {differences[0]}"
            return {
                "solution": formula,
                "confidence": 0.9,
                "reasoning_steps": ["Identified constant difference between input and output",
                                   f"All pairs have difference of approximately {differences[0]}"]
            }

        # Check for consistent ratio (multiplicative function)
        if all(abs(r - ratios[0]) < 0.0001 for r in ratios if r != float('inf')):
            formula = f"f(x) = x * {ratios[0]}"
            return {
                "solution": formula,
                "confidence": 0.9,
                "reasoning_steps": ["Identified constant ratio between input and output",
                                   f"All pairs have ratio of approximately {ratios[0]}"]
            }

        # Check for polynomial relationship
        if len(input_output_pairs) >= 3:
            # Try quadratic: f(x) = ax² + bx + c
            try:
                x_values = [x for x, _ in input_output_pairs]
                y_values = [y for _, y in input_output_pairs]

                # Fit quadratic using least squares (simplified approach)
                x_squared = [x**2 for x in x_values]
                n = len(x_values)
                sum_x = sum(x_values)
                sum_x_squared = sum(x_squared)
                sum_x_cubed = sum(x**3 for x in x_values)
                sum_x_fourth = sum(x**4 for x in x_values)
                sum_y = sum(y_values)
                sum_xy = sum(x*y for x, y in zip(x_values, y_values))
                sum_x_squared_y = sum(x**2*y for x, y in zip(x_values, y_values))

                # Set up the system of linear equations
                A = np.array([
                    [n, sum_x, sum_x_squared],
                    [sum_x, sum_x_squared, sum_x_cubed],
                    [sum_x_squared, sum_x_cubed, sum_x_fourth]
                ])
                b = np.array([sum_y, sum_xy, sum_x_squared_y])

                # Solve the system
                coeffs = np.linalg.solve(A, b)
                c, b, a = coeffs  # coefficients for f(x) = ax² + bx + c

                # Check fit accuracy
                predictions = [a*x**2 + b*x + c for x in x_values]
                errors = [abs(pred - actual) for pred, actual in zip(predictions, y_values)]
                avg_error = sum(errors) / len(errors)

                if avg_error < 0.1:  # Good fit
                    formula = f"f(x) = {a:.2f}x² + {b:.2f}x + {c:.2f}"
                    return {
                        "solution": formula,
                        "confidence": 0.85,
                        "reasoning_steps": ["Identified quadratic relationship",
                                          f"Formula {formula} fits with average error {avg_error:.4f}"]
                    }
            except:
                pass  # If polynomial fitting fails, continue to other approaches

        # If all else fails, give a composite function
        return {
            "solution": "f(x) = complex transformation (multiple steps)",
            "confidence": 0.4,
            "reasoning_steps": ["No simple function identified",
                              "May require multiple transformations or piecewise definition"]
        }

    def _deductive_math_operation(self, task):
        """Apply deductive reasoning to math operation tasks"""
        expression = task.get("expression", "")

        # Try to evaluate directly
        try:
            result = eval(expression)
            # Round floating point for cleaner presentation
            if isinstance(result, float):
                result = round(result, 4)

            return {
                "solution": str(result),
                "confidence": 0.95,
                "reasoning_steps": ["Directly evaluated the expression",
                                  f"Calculated {expression} = {result}"]
            }
        except Exception as e:
            # If direct evaluation fails, try step-by-step
            steps = []
            confidence = 0.5

            try:
                # Parse expression to handle it in parts
                # This is simplified - full parsing would need a proper math parser

                # Remove outer parentheses if present
                clean_expr = expression.strip()
                if clean_expr.startswith("(") and clean_expr.endswith(")"):
                    clean_expr = clean_expr[1:-1]

                # Simplify expressions with order of operations
                steps.append(f"Starting with expression: {clean_expr}")

                # Execute by order of operations
                # 1. Parentheses
                paren_pattern = r'\([^()]+\)'
                while re.search(paren_pattern, clean_expr):
                    for paren_match in re.finditer(paren_pattern, clean_expr):
                        paren_expr = paren_match.group()[1:-1]  # Remove parentheses
                        paren_result = eval(paren_expr)
                        steps.append(f"Evaluate parentheses: {paren_expr} = {paren_result}")
                        # Replace in expression
                        clean_expr = clean_expr[:paren_match.start()] + str(paren_result) + clean_expr[paren_match.end():]
                        break  # Handle one at a time

                # 2. Exponents (if present)
                exp_pattern = r'(\d+)\s*\*\*\s*(\d+)'
                while re.search(exp_pattern, clean_expr):
                    for exp_match in re.finditer(exp_pattern, clean_expr):
                        base, power = map(int, exp_match.groups())
                        exp_result = base ** power
                        steps.append(f"Evaluate exponent: {base}**{power} = {exp_result}")
                        # Replace in expression
                        clean_expr = clean_expr[:exp_match.start()] + str(exp_result) + clean_expr[exp_match.end():]
                        break  # Handle one at a time

                # 3. Multiplication and Division (left to right)
                mul_div_pattern = r'(\d+\.?\d*)\s*([*/])\s*(\d+\.?\d*)'
                while re.search(mul_div_pattern, clean_expr):
                    for op_match in re.finditer(mul_div_pattern, clean_expr):
                        left, op, right = op_match.groups()
                        left, right = float(left), float(right)
                        if op == '*':
                            result = left * right
                            steps.append(f"Multiply: {left} * {right} = {result}")
                        else:  # Division
                            result = left / right
                            steps.append(f"Divide: {left} / {right} = {result}")
                        # Replace in expression
                        clean_expr = clean_expr[:op_match.start()] + str(result) + clean_expr[op_match.end():]
                        break  # Handle one at a time

                # 4. Addition and Subtraction (left to right)
                add_sub_pattern = r'(\d+\.?\d*)\s*([+-])\s*(\d+\.?\d*)'
                while re.search(add_sub_pattern, clean_expr):
                    for op_match in re.finditer(add_sub_pattern, clean_expr):
                        left, op, right = op_match.groups()
                        left, right = float(left), float(right)
                        if op == '+':
                            result = left + right
                            steps.append(f"Add: {left} + {right} = {result}")
                        else:  # Subtraction
                            result = left - right
                            steps.append(f"Subtract: {left} - {right} = {result}")
                        # Replace in expression
                        clean_expr = clean_expr[:op_match.start()] + str(result) + clean_expr[op_match.end():]
                        break  # Handle one at a time

                # Final result
                final_result = eval(clean_expr)
                steps.append(f"Final result: {final_result}")

                # Round if it's a float
                if isinstance(final_result, float):
                    final_result = round(final_result, 4)

                return {
                    "solution": str(final_result),
                    "confidence": 0.9,
                    "reasoning_steps": steps
                }

            except Exception:
                # If step-by-step also fails, make a best guess
                # This is an extremely simplified approach
                terms = re.findall(r'[\d.]+', expression)
                if terms:
                    # Just guess based on the numbers present and operations
                    nums = [float(t) for t in terms]

                    if "+" in expression:
                        result = sum(nums)
                    elif "-" in expression:
                        result = nums[0] - sum(nums[1:])
                    elif "*" in expression:
                        result = 1
                        for n in nums:
                            result *= n
                    elif "/" in expression:
                        result = nums[0]
                        for n in nums[1:]:
                            if n != 0:
                                result /= n
                    else:
                        result = nums[0]

                    # Round result
                    if isinstance(result, float):
                        result = round(result, 4)

                    return {
                        "solution": str(result),
                        "confidence": 0.3,
                        "reasoning_steps": ["Failed to parse expression fully",
                                         "Made best guess based on numbers and operators"]
                    }

                # Truly stumped
                return {
                    "solution": "Unable to determine",
                    "confidence": 0.1,
                    "reasoning_steps": ["Expression too complex to parse"]
                }

    def _deductive_sequence_completion(self, task):
        """Apply deductive reasoning to sequence completion tasks"""
        sequence = task.get("visible_sequence", [])

        if not sequence or len(sequence) < 2:
            return {
                "solution": [1, 2, 3],  # Default if no meaningful sequence
                "confidence": 0.1,
                "reasoning_steps": ["Insufficient sequence elements to determine pattern"]
            }

        # Check arithmetic sequence (constant difference)
        differences = [sequence[i] - sequence[i-1] for i in range(1, len(sequence))]
        if all(abs(d - differences[0]) < 0.001 for d in differences):
            # Arithmetic sequence: an = a1 + (n-1)d
            d = differences[0]
            next_values = [sequence[-1] + d * (i+1) for i in range(3)]
            return {
                "solution": next_values,
                "confidence": 0.9,
                "reasoning_steps": [f"Identified arithmetic sequence with common difference {d}",
                                  f"Extending sequence using a_n = a_1 + (n-1)d"]
            }

        # Check geometric sequence (constant ratio)
        if all(sequence[i] != 0 for i in range(len(sequence))):
            ratios = [sequence[i] / sequence[i-1] for i in range(1, len(sequence))]
            if all(abs(r - ratios[0]) < 0.001 for r in ratios):
                # Geometric sequence: an = a1 * r^(n-1)
                r = ratios[0]
                next_values = [sequence[-1] * r ** (i+1) for i in range(3)]
                # Round if they're floats
                next_values = [round(v, 4) if isinstance(v, float) else v for v in next_values]
                return {
                    "solution": next_values,
                    "confidence": 0.9,
                    "reasoning_steps": [f"Identified geometric sequence with common ratio {r}",
                                     f"Extending sequence using a_n = a_1 * r^(n-1)"]
                }

        # Check quadratic sequence (second differences are constant)
        if len(sequence) >= 3:
            first_diff = [sequence[i] - sequence[i-1] for i in range(1, len(sequence))]
            second_diff = [first_diff[i] - first_diff[i-1] for i in range(1, len(first_diff))]

            if all(abs(d - second_diff[0]) < 0.001 for d in second_diff):
                # Quadratic sequence: an = an-1 + (first_diff[-1] + second_diff[0])
                next_values = []
                current = sequence[-1]
                next_diff = first_diff[-1]

                for _ in range(3):
                    next_diff += second_diff[0]
                    current += next_diff
                    next_values.append(current)

                return {
                    "solution": next_values,
                    "confidence": 0.85,
                    "reasoning_steps": ["Identified quadratic sequence (constant second difference)",
                                     f"Second difference is {second_diff[0]}"]
                }

        # Check Fibonacci-like (each term is sum of previous two)
        if len(sequence) >= 3:
            if all(abs(sequence[i] - (sequence[i-1] + sequence[i-2])) < 0.001 for i in range(2, len(sequence))):
                next_values = [sequence[-1] + sequence[-2]]
                next_values.append(next_values[0] + sequence[-1])
                next_values.append(next_values[1] + next_values[0])

                return {
                    "solution": next_values,
                    "confidence": 0.9,
                    "reasoning_steps": ["Identified Fibonacci-like sequence (each term is sum of previous two)",
                                     "Extending pattern: a_n = a_(n-1) + a_(n-2)"]
                }

        # Check powers (each term is a power of the same base)
        if all(v > 0 for v in sequence):
            # Try bases 2-10
            for base in range(2, 11):
                # Convert to logs to check if they form an arithmetic sequence
                try:
                    logs = [math.log(v, base) for v in sequence]
                    log_diffs = [logs[i] - logs[i-1] for i in range(1, len(logs))]

                    if all(abs(d - log_diffs[0]) < 0.001 for d in log_diffs):
                        # Power sequence: an = base^(a1_log + (n-1)*d)
                        next_values = [base ** (logs[-1] + log_diffs[0] * (i+1)) for i in range(3)]
                        # Round to integers if close
                        next_values = [round(v) if abs(v - round(v)) < 0.001 else v for v in next_values]

                        return {
                            "solution": next_values,
                            "confidence": 0.8,
                            "reasoning_steps": [f"Identified power sequence with base {base}",
                                             f"Pattern: each term is {base} raised to powers in arithmetic sequence"]
                        }
                except:
                    continue

        # If all else fails, use a heuristic approach
        try:
            # Might be a more complex pattern
            if all(isinstance(v, int) for v in sequence):
                # Check OEIS database patterns (simplified)
                if sequence == [1, 1, 2, 3, 5, 8]:  # Fibonacci
                    next_values = [13, 21, 34]
                    return {
                        "solution": next_values,
                        "confidence": 0.95,
                        "reasoning_steps": ["Identified Fibonacci sequence",
                                         "Pattern: each term is sum of two previous terms"]
                    }

                if sequence == [1, 2, 4, 8, 16, 32]:  # Powers of 2
                    next_values = [64, 128, 256]
                    return {
                        "solution": next_values,
                        "confidence": 0.95,
                        "reasoning_steps": ["Identified powers of 2",
                                         "Pattern: each term is double the previous"]
                    }

                if sequence == [1, 3, 6, 10, 15]:  # Triangular numbers
                    next_values = [21, 28, 36]
                    return {
                        "solution": next_values,
                        "confidence": 0.9,
                        "reasoning_steps": ["Identified triangular numbers",
                                         "Pattern: n(n+1)/2 for n=1,2,3,..."]
                    }

            # Linear extrapolation as a last resort
            next_values = []
            for i in range(1, 4):
                value = sequence[-1] + (sequence[-1] - sequence[0]) / (len(sequence) - 1) * i
                if isinstance(sequence[0], int) and abs(value - round(value)) < 0.001:
                    value = round(value)
                next_values.append(value)

            return {
                "solution": next_values,
                "confidence": 0.4,
                "reasoning_steps": ["No standard pattern identified",
                                  "Used linear extrapolation as an approximation"]
            }

        except Exception:
            # Ultimate fallback
            next_values = [sequence[-1] + (sequence[-1] - sequence[-2])] * 3
            return {
                "solution": next_values,
                "confidence": 0.2,
                "reasoning_steps": ["Pattern unclear",
                                  "Defaulted to extending with the last observed difference"]
            }

    def _deductive_logic_reasoning(self, task):
        """Apply deductive reasoning to logic puzzles"""
        statements = task.get("statements", [])
        conclusion = task.get("conclusion", "")

        if not statements:
            return {
                "solution": "Cannot determine",
                "confidence": 0.1,
                "reasoning_steps": ["No logical statements provided"]
            }

        # Check for common logical patterns
        if len(statements) >= 2 and conclusion:
            # Convert to simpler representation for analysis
            simplified_statements = []
            for stmt in statements:
                simplified_statements.append(self._simplify_logical_statement(stmt))

            simplified_conclusion = self._simplify_logical_statement(conclusion)

            # Check for Modus Ponens: If P then Q, P, therefore Q
            if len(simplified_statements) >= 2:
                # Find implication statement and assertion
                has_modus_ponens = False

                for i, stmt1 in enumerate(simplified_statements):
                    if "→" in stmt1:  # Contains implication
                        p, q = stmt1.split("→")
                        p, q = p.strip(), q.strip()

                        # Check if P is asserted in another statement
                        for j, stmt2 in enumerate(simplified_statements):
                            if i != j and not "→" in stmt2 and self._statement_equivalent(stmt2, p):
                                # If conclusion is Q, it's valid modus ponens
                                if self._statement_equivalent(simplified_conclusion, q):
                                    has_modus_ponens = True
                                    reasoning_steps = [
                                        f"Statement 1: If {p} then {q}",
                                        f"Statement 2: {p}",
                                        f"Valid conclusion by Modus Ponens: {q}"
                                    ]
                                    return {
                                        "solution": "True",
                                        "confidence": 0.95,
                                        "reasoning_steps": reasoning_steps
                                    }

                # Check for Modus Tollens: If P then Q, not Q, therefore not P
                has_modus_tollens = False

                for i, stmt1 in enumerate(simplified_statements):
                    if "→" in stmt1:  # Contains implication
                        p, q = stmt1.split("→")
                        p, q = p.strip(), q.strip()

                        # Check if not Q is asserted in another statement
                        for j, stmt2 in enumerate(simplified_statements):
                            if i != j and "¬" in stmt2:
                                negated_term = stmt2.replace("¬", "").strip()
                                if self._statement_equivalent(negated_term, q):
                                    # If conclusion is not P, it's valid modus tollens
                                    if "¬" in simplified_conclusion:
                                        negated_conclusion = simplified_conclusion.replace("¬", "").strip()
                                        if self._statement_equivalent(negated_conclusion, p):
                                            has_modus_tollens = True
                                            reasoning_steps = [
                                                f"Statement 1: If {p} then {q}",
                                                f"Statement 2: Not {q}",
                                                f"Valid conclusion by Modus Tollens: Not {p}"
                                            ]
                                            return {
                                                "solution": "True",
                                                "confidence": 0.95,
                                                "reasoning_steps": reasoning_steps
                                            }

                # Check for Hypothetical Syllogism: If P then Q, If Q then R, therefore If P then R
                has_syllogism = False

                for i, stmt1 in enumerate(simplified_statements):
                    if "→" in stmt1:  # Contains implication
                        p, q = stmt1.split("→")
                        p, q = p.strip(), q.strip()

                        # Find another implication
                        for j, stmt2 in enumerate(simplified_statements):
                            if i != j and "→" in stmt2:
                                q2, r = stmt2.split("→")
                                q2, r = q2.strip(), r.strip()

                                # Check if q = q2 (connecting the implications)
                                if self._statement_equivalent(q, q2):
                                    # If conclusion is "If P then R", it's valid syllogism
                                    expected_conclusion = f"{p} → {r}"
                                    if self._statement_equivalent(simplified_conclusion, expected_conclusion):
                                        has_syllogism = True
                                        reasoning_steps = [
                                            f"Statement 1: If {p} then {q}",
                                            f"Statement 2: If {q} then {r}",
                                            f"Valid conclusion by Hypothetical Syllogism: If {p} then {r}"
                                        ]
                                        return {
                                            "solution": "True",
                                            "confidence": 0.95,
                                            "reasoning_steps": reasoning_steps
                                        }

            # Check for common fallacies
            # Affirming the Consequent: If P then Q, Q, therefore P (INVALID)
            has_consequent_fallacy = False

            for i, stmt1 in enumerate(simplified_statements):
                if "→" in stmt1:  # Contains implication
                    p, q = stmt1.split("→")
                    p, q = p.strip(), q.strip()

                    # Check if Q is asserted in another statement
                    for j, stmt2 in enumerate(simplified_statements):
                        if i != j and not "→" in stmt2 and self._statement_equivalent(stmt2, q):
                            # If conclusion is P, it's the fallacy
                            if self._statement_equivalent(simplified_conclusion, p):
                                has_consequent_fallacy = True
                                reasoning_steps = [
                                    f"Statement 1: If {p} then {q}",
                                    f"Statement 2: {q}",
                                    f"Invalid conclusion (Affirming the Consequent fallacy): {p}",
                                    "This reasoning is invalid because Q could be true for other reasons besides P"
                                ]
                                return {
                                    "solution": "False",
                                    "confidence": 0.9,
                                    "reasoning_steps": reasoning_steps
                                }

            # Denying the Antecedent: If P then Q, not P, therefore not Q (INVALID)
            has_antecedent_fallacy = False

            for i, stmt1 in enumerate(simplified_statements):
                if "→" in stmt1:  # Contains implication
                    p, q = stmt1.split("→")
                    p, q = p.strip(), q.strip()

                    # Check if not P is asserted in another statement
                    for j, stmt2 in enumerate(simplified_statements):
                        if i != j and "¬" in stmt2:
                            negated_term = stmt2.replace("¬", "").strip()
                            if self._statement_equivalent(negated_term, p):
                                # If conclusion is not Q, it's the fallacy
                                if "¬" in simplified_conclusion:
                                    negated_conclusion = simplified_conclusion.replace("¬", "").strip()
                                    if self._statement_equivalent(negated_conclusion, q):
                                        has_antecedent_fallacy = True
                                        reasoning_steps = [
                                            f"Statement 1: If {p} then {q}",
                                            f"Statement 2: Not {p}",
                                            f"Invalid conclusion (Denying the Antecedent fallacy): Not {q}",
                                            "This reasoning is invalid because Q could still be true for reasons other than P"
                                        ]
                                        return {
                                            "solution": "False",
                                            "confidence": 0.9,
                                            "reasoning_steps": reasoning_steps
                                        }

        # If the logical pattern isn't recognized, fall back to more general analysis
        # This is highly simplified and would need a full logic reasoner for complete coverage

        # Check syllogisms (All A are B, All B are C, therefore All A are C)
        if "All" in " ".join(statements) and "All" in conclusion:
            is_valid_syllogism = False

            # Extremely simplified syllogism check
            if len(statements) >= 2:
                # Extract terms
                terms = []
                for stmt in statements:
                    if "All" in stmt and "are" in stmt:
                        parts = stmt.split("are")
                        subject = parts[0].replace("All", "").strip()
                        predicate = parts[1].strip()
                        terms.append((subject, predicate))

                # Check if terms connect
                if len(terms) >= 2:
                    # Find connected terms
                    for i, (s1, p1) in enumerate(terms):
                        for j, (s2, p2) in enumerate(terms):
                            if i != j and p1.strip().lower() == s2.strip().lower():
                                # Check if conclusion connects s1 to p2
                                expected = f"All {s1} are {p2}"
                                if expected.lower() in conclusion.lower():
                                    is_valid_syllogism = True
                                    reasoning_steps = [
                                        f"Statement 1: All {s1} are {p1}",
                                        f"Statement 2: All {p1} are {p2}",
                                        f"Valid syllogistic conclusion: All {s1} are {p2}"
                                    ]
                                    return {
                                        "solution": "True",
                                        "confidence": 0.9,
                                        "reasoning_steps": reasoning_steps
                                    }

        # Defaulting to "cannot determine" with low confidence
        return {
            "solution": "Cannot determine with certainty",
            "confidence": 0.4,
            "reasoning_steps": ["Logical pattern not fully recognized",
                             "More complex logical analysis required"]
        }

    def _simplify_logical_statement(self, statement):
        """Simplify logical statement for analysis"""
        # Very basic simplification
        simplified = statement.lower()

        # Replace "if...then" with →
        simplified = re.sub(r"if\s+([^,]+)\s+then\s+([^,]+)", r"\1 → \2", simplified)

        # Replace "not" with ¬
        simplified = re.sub(r"(?:^|\s+)not\s+", " ¬", simplified)
        simplified = re.sub(r"(?:^|\s+)isn't\s+", " ¬", simplified)
        simplified = re.sub(r"(?:^|\s+)doesn't\s+", " ¬", simplified)

        # Replace "is false" with ¬
        simplified = re.sub(r"is\s+false", "¬", simplified)

        # Clean up
        simplified = simplified.replace(".", "").replace(",", "").strip()

        return simplified

    def _statement_equivalent(self, stmt1, stmt2):
        """Check if two simplified statements are logically equivalent"""
        # Very basic equivalence check
        s1 = stmt1.lower().strip()
        s2 = stmt2.lower().strip()

        # Direct equality
        if s1 == s2:
            return True

        # Handle negation equivalence
        if s1.startswith("¬") and s2.startswith("¬"):
            return s1[1:].strip() == s2[1:].strip()

        # More sophisticated equivalence checking would be needed for a complete solution

        return False

    def _abductive_reasoning(self, task):
        """Apply abductive reasoning (inference to best explanation)"""
        task_type = task["type"]

        # Create candidates with this reasoning approach
        candidates = []

        # Apply based on task type
        if task_type == "function_inference":
            candidates.append(self._abductive_function_inference(task))
        elif task_type == "sequence_completion":
            candidates.append(self._abductive_sequence_completion(task))
        else:
            # Generic abductive approach
            candidates.append({
                "solution": f"Abductive solution for {task_type}",
                "confidence": 0.4,
                "reasoning_steps": ["Inferred most likely explanation for observed patterns"]
            })

        return candidates

    def _abductive_function_inference(self, task):
        """Use abductive reasoning for function inference tasks"""
        input_output_pairs = task.get("input_output_pairs", [])

        if not input_output_pairs:
            return {
                "solution": "Unable to determine function",
                "confidence": 0.1,
                "reasoning_steps": ["No input-output pairs provided"]
            }

        # Look for patterns in the data that might suggest a function
        candidate_functions = [
            # Basic transforms
            (lambda x: x + 1, "f(x) = x + 1", "addition"),
            (lambda x: x - 1, "f(x) = x - 1", "subtraction"),
            (lambda x: x * 2, "f(x) = x * 2", "multiplication"),
            (lambda x: x / 2, "f(x) = x / 2", "division"),
            (lambda x: x ** 2, "f(x) = x ^ 2", "square"),
            (lambda x: x ** 3, "f(x) = x ^ 3", "cube"),
            (lambda x: math.sqrt(x) if x >= 0 else float('nan'), "f(x) = sqrt(x)", "square root"),
            (lambda x: abs(x), "f(x) = |x|", "absolute value"),
            (lambda x: -x, "f(x) = -x", "negation"),
            (lambda x: 1/x if x != 0 else float('inf'), "f(x) = 1/x", "reciprocal"),
            (lambda x: math.sin(x), "f(x) = sin(x)", "sine"),
            (lambda x: math.cos(x), "f(x) = cos(x)", "cosine"),
            (lambda x: round(x), "f(x) = round(x)", "rounding"),
            (lambda x: math.floor(x), "f(x) = floor(x)", "floor"),
            (lambda x: math.ceil(x), "f(x) = ceiling(x)", "ceiling"),
            # Composite functions
            (lambda x: 2*x + 1, "f(x) = 2x + 1", "linear"),
            (lambda x: x**2 + x, "f(x) = x^2 + x", "quadratic"),
            (lambda x: x**2 - x, "f(x) = x^2 - x", "quadratic"),
            (lambda x: x**2 + 1, "f(x) = x^2 + 1", "quadratic"),
            (lambda x: x**3 + x, "f(x) = x^3 + x", "cubic"),
            (lambda x: 1/(x+1) if x != -1 else float('inf'), "f(x) = 1/(x+1)", "rational"),
            (lambda x: (x**2 - 1)/(x - 1) if x != 1 else float('nan'), "f(x) = (x^2-1)/(x-1)", "rational")
        ]

        # Track function performance
        function_scores = []

        # Try each candidate function
        for fn, formula, fn_type in candidate_functions:
            try:
                # Apply function to inputs and compare to outputs
                errors = []
                for x, y in input_output_pairs:
                    calculated = fn(x)
                    # Handle NaN values
                    if math.isnan(calculated) or math.isinf(calculated):
                        errors.append(float('inf'))
                    else:
                        errors.append(abs(calculated - y))

                # Calculate average error
                if errors and not all(math.isinf(e) for e in errors):
                    avg_error = sum(e for e in errors if not math.isinf(e)) / len([e for e in errors if not math.isinf(e)])
                    function_scores.append((formula, fn_type, avg_error))
            except Exception:
                # Skip functions that fail to apply
                continue

        # Sort by error (lowest first)
        function_scores.sort(key=lambda x: x[2])

        # If we have good candidates
        if function_scores and function_scores[0][2] < 0.01:
            formula, fn_type, error = function_scores[0]
            return {
                "solution": formula,
                "confidence": min(0.95, max(0.5, 1.0 - error * 10)),
                "reasoning_steps": [f"Tested multiple function hypotheses",
                                    f"Selected {fn_type} function with lowest error",
                                    f"Function {formula} has average error of {error:.6f}"]
            }
        elif function_scores and function_scores[0][2] < 0.1:
            formula, fn_type, error = function_scores[0]
            return {
                "solution": formula,
                "confidence": min(0.8, max(0.3, 1.0 - error * 5)),
                "reasoning_steps": [f"Tested multiple function hypotheses",
                                    f"Selected {fn_type} function as best approximation",
                                    f"Function {formula} has average error of {error:.6f}"]
            }
        else:
            # Look for piecewise functions
            inputs = [x for x, _ in input_output_pairs]
            ascending = all(inputs[i] <= inputs[i+1] for i in range(len(inputs)-1))

            if ascending and len(input_output_pairs) >= 4:
                # Try piecewise linear interpolation
                pieces = []
                for i in range(len(input_output_pairs) - 1):
                    x1, y1 = input_output_pairs[i]
                    x2, y2 = input_output_pairs[i+1]
                    if x2 != x1:
                        slope = (y2 - y1) / (x2 - x1)
                        intercept = y1 - slope * x1
                        pieces.append((f"{slope:.2f}x + {intercept:.2f}", x1, x2))

                if pieces:
                    solution = "f(x) = "
                    for i, (piece, x1, x2) in enumerate(pieces):
                        if i == 0:
                            solution += f"{piece} for x ∈ [{x1}, {x2})"
                        elif i == len(pieces) - 1:
                            solution += f", {piece} for x ∈ [{x1}, {x2}]"
                        else:
                            solution += f", {piece} for x ∈ [{x1}, {x2})"

                    return {
                        "solution": solution,
                        "confidence": 0.6,
                        "reasoning_steps": ["No simple function fits the data well",
                                          "Created piecewise linear function approximation",
                                          f"Function consists of {len(pieces)} linear segments"]
                    }

            # No good function found, return best guess with low confidence
            if function_scores:
                formula, fn_type, error = function_scores[0]
                return {
                    "solution": formula + " (approximate)",
                    "confidence": 0.3,
                    "reasoning_steps": ["No function fits the data precisely",
                                      f"Best approximation is {fn_type} function",
                                      f"Function {formula} has high error of {error:.6f}"]
                }
            else:
                return {
                    "solution": "Cannot determine a simple function",
                    "confidence": 0.1,
                    "reasoning_steps": ["No standard function fits the data",
                                      "May require a complex or specialized function"]
                }

    def _abductive_sequence_completion(self, task):
        """Use abductive reasoning for sequence completion tasks"""
        sequence = task.get("visible_sequence", [])

        if not sequence or len(sequence) < 3:
            return {
                "solution": [1, 2, 3],
                "confidence": 0.1,
                "reasoning_steps": ["Insufficient sequence elements to infer pattern"]
            }

        # Try to infer pattern from specific known sequences
        # This is different from deductive reasoning as we're considering multiple hypotheses

        # Check for special sequences first
        if sequence == [1, 1, 2, 3, 5, 8]:  # Fibonacci
            return {
                "solution": [13, 21, 34],
                "confidence": 0.95,
                "reasoning_steps": ["Recognized Fibonacci sequence",
                                  "Each number is the sum of the two preceding ones"]
            }

        if sequence == [0, 1, 4, 9, 16, 25]:  # Perfect squares
            return {
                "solution": [36, 49, 64],
                "confidence": 0.95,
                "reasoning_steps": ["Recognized perfect squares",
                                  "Sequence follows pattern n^2 for n=0,1,2,..."]
            }

        if sequence == [1, 3, 6, 10, 15]:  # Triangular numbers
            return {
                "solution": [21, 28, 36],
                "confidence": 0.95,
                "reasoning_steps": ["Recognized triangular numbers",
                                  "Sequence follows pattern n(n+1)/2"]
            }

        if sequence == [1, 4, 9, 16, 25, 36]:  # Perfect squares starting from 1
            return {
                "solution": [49, 64, 81],
                "confidence": 0.95,
                "reasoning_steps": ["Recognized perfect squares",
                                  "Sequence follows pattern n^2 for n=1,2,3,..."]
            }

        if sequence == [2, 4, 8, 16, 32]:  # Powers of 2
            return {
                "solution": [64, 128, 256],
                "confidence": 0.95,
                "reasoning_steps": ["Recognized powers of 2",
                                  "Sequence follows pattern 2^n for n=1,2,3,..."]
            }

        if sequence == [1, 3, 5, 7, 9]:  # Odd numbers
            return {
                "solution": [11, 13, 15],
                "confidence": 0.95,
                "reasoning_steps": ["Recognized odd numbers",
                                  "Sequence follows pattern 2n-1 for n=1,2,3,..."]
            }

        if sequence == [2, 4, 6, 8, 10]:  # Even numbers
            return {
                "solution": [12, 14, 16],
                "confidence": 0.95,
                "reasoning_steps": ["Recognized even numbers",
                                  "Sequence follows pattern 2n for n=1,2,3,..."]
            }

        # If not a recognized special sequence, explore multiple hypotheses
        hypotheses = []

        # Hypothesis 1: Linear sequence (constant difference)
        first_diff = [sequence[i] - sequence[i-1] for i in range(1, len(sequence))]
        hypothesis1_quality = sum(abs(d - first_diff[0]) for d in first_diff) / len(first_diff)

        if hypothesis1_quality < 0.001:  # Almost perfect match
            next_values = [sequence[-1] + first_diff[0]]
            next_values.append(next_values[0] + first_diff[0])
            next_values.append(next_values[1] + first_diff[0])

            hypotheses.append({
                "solution": next_values,
                "confidence": 0.9,
                "quality": hypothesis1_quality,
                "reasoning": f"Linear sequence with constant difference {first_diff[0]}"
            })

        # Hypothesis 2: Geometric sequence (constant ratio)
        if all(sequence[i] != 0 for i in range(len(sequence))):
            ratios = [sequence[i] / sequence[i-1] for i in range(1, len(sequence))]
            hypothesis2_quality = sum(abs(r - ratios[0]) for r in ratios) / len(ratios)

            if hypothesis2_quality < 0.001:  # Almost perfect match
                next_values = [sequence[-1] * ratios[0]]
                next_values.append(next_values[0] * ratios[0])
                next_values.append(next_values[1] * ratios[0])

                # Integer check
                if all(isinstance(s, int) for s in sequence):
                    next_values = [round(v) for v in next_values]

                hypotheses.append({
                    "solution": next_values,
                    "confidence": 0.9,
                    "quality": hypothesis2_quality,
                    "reasoning": f"Geometric sequence with constant ratio {ratios[0]:.2f}"
                })

        # Hypothesis 3: Quadratic sequence (second differences are constant)
        if len(first_diff) >= 2:
            second_diff = [first_diff[i] - first_diff[i-1] for i in range(1, len(first_diff))]
            hypothesis3_quality = sum(abs(d - second_diff[0]) for d in second_diff) / len(second_diff)

            if hypothesis3_quality < 0.001:  # Almost perfect match
                next_diff = first_diff[-1] + second_diff[0]
                next_values = [sequence[-1] + next_diff]
                next_diff = next_diff + second_diff[0]
                next_values.append(next_values[0] + next_diff)
                next_diff = next_diff + second_diff[0]
                next_values.append(next_values[1] + next_diff)

                # Integer check
                if all(isinstance(s, int) for s in sequence):
                    next_values = [round(v) for v in next_values]

                hypotheses.append({
                    "solution": next_values,
                    "confidence": 0.85,
                    "quality": hypothesis3_quality,
                    "reasoning": f"Quadratic sequence with second difference {second_diff[0]}"
                })

        # Hypothesis 4: Recurrence relation (each term is a function of previous terms)
        # Try Fibonacci-like recurrence
        if len(sequence) >= 4:
            rec_errors = []
            for i in range(2, len(sequence)):
                expected = sequence[i-1] + sequence[i-2]
                actual = sequence[i]
                rec_errors.append(abs(expected - actual))

            hypothesis4_quality = sum(rec_errors) / len(rec_errors)

            if hypothesis4_quality < 0.1:  # Good match
                next_values = [sequence[-1] + sequence[-2]]
                next_values.append(next_values[0] + sequence[-1])
                next_values.append(next_values[1] + next_values[0])

                # Integer check
                if all(isinstance(s, int) for s in sequence):
                    next_values = [round(v) for v in next_values]

                hypotheses.append({
                    "solution": next_values,
                    "confidence": 0.8,
                    "quality": hypothesis4_quality,
                    "reasoning": "Recurrence relation: a(n) = a(n-1) + a(n-2)"
                })

        # Sort hypotheses by quality (lower is better)
        hypotheses.sort(key=lambda h: h["quality"])

        # Return best hypothesis if we have one
        if hypotheses:
            best = hypotheses[0]
            return {
                "solution": best["solution"],
                "confidence": best["confidence"],
                "reasoning_steps": ["Considered multiple sequence pattern hypotheses",
                                  best["reasoning"],
                                  f"Selected best hypothesis with quality score {best['quality']:.6f}"]
            }

        # If all hypotheses fail, try a simple fallback prediction
        last_value = sequence[-1]
        second_last = sequence[-2] if len(sequence) > 1 else 0

        # Use last difference
        diff = last_value - second_last
        next_values = [last_value + diff, last_value + 2*diff, last_value + 3*diff]

        return {
            "solution": next_values,
            "confidence": 0.3,
            "reasoning_steps": ["No clear pattern identified in the sequence",
                              "Used last difference for prediction",
                              "Low confidence in this continuation"]
        }

    def _inductive_reasoning(self, task):
        """Apply inductive reasoning (from specific instances to general rules)"""
        candidates = []

        # Apply based on task type
        if task["type"] == "code_completion":
            candidates.append(self._inductive_code_completion(task))

        elif task["type"] == "function_inference":
            candidates.append(self._inductive_function_inference(task))

        elif task["type"] == "sequence_completion":
            candidates.append(self._inductive_sequence_completion(task))

        else:
            # Generic inductive approach
            candidates.append({
                "solution": f"Inductive solution for {task['type']}",
                "confidence": 0.4,
                "reasoning_steps": ["Generalized from specific examples to pattern"]
            })

        return candidates

    def _inductive_code_completion(self, task):
        """Use inductive reasoning for code completion"""
        code = task.get("code", "")

        # Extract function name and parameters
        function_name = ""
        params = []

        # Parse function signature
        for line in code.split("\n"):
            if line.strip().startswith("def "):
                function_parts = line.split("(", 1)
                if len(function_parts) > 1:
                    function_name = function_parts[0].replace("def", "").strip()
                    param_str = function_parts[1].split(")", 1)[0]
                    params = [p.strip() for p in param_str.split(",") if p.strip()]
                break

        # Look at existing patterns in the code
        code_lines = code.split("\n")
        body_lines = [line for line in code_lines if not line.strip().startswith("def")]

        # Find TODOs and gaps
        gaps = []
        for i, line in enumerate(body_lines):
            if "TODO" in line or "..." in line:
                gaps.append((i, line))

        # If no explicit gaps, look for common patterns that might be missing
        if not gaps:
            # Check for missing return statement
            has_return = any("return" in line for line in body_lines)
            if not has_return:
                gaps.append((len(body_lines), "# Missing return statement"))

        # Look for similar functions in past experiences
        similar_experiences = []
        if hasattr(self.model, "experience_manager"):
            code_experiences = self.model.experience_manager.get_experiences_by_type(
                "evolution", limit=20)

            for exp in code_experiences:
                if isinstance(exp.get("content"), dict):
                    content = exp.get("content", {})
                    if (content.get("task", {}).get("type") == "code_completion" and
                        content.get("success", False)):
                        # Look at successful code completions
                        task_content = content.get("task", {}).get("content", {})
                        if task_content and function_name:
                            if function_name.lower() in task_content.get("code", "").lower():
                                similar_experiences.append(content)

        # Generate solution based on inductive patterns
        solution = code
        confidence = 0.3
        reasoning_steps = []

        # If we have similar past experiences, learn from them
        if similar_experiences:
            # Extract patterns from similar experiences
            patterns = []
            for exp in similar_experiences:
                exp_solution = exp.get("solution", "")
                # Extract successful patterns
                if "for " in exp_solution and not any("for " in line for line in body_lines):
                    patterns.append("Missing for loop")
                if "if " in exp_solution and not any("if " in line for line in body_lines):
                    patterns.append("Missing if condition")
                if "return " in exp_solution and not has_return:
                    patterns.append("Missing return statement")

            # Apply patterns
            if patterns:
                reasoning_steps.append(f"Found {len(patterns)} patterns from similar past experiences")
                reasoning_steps.append(f"Applied patterns: {', '.join(patterns)}")

                # Generate code based on patterns
                updated_code = code

                for pattern in patterns:
                    if pattern == "Missing for loop" and params:
                        # Add for loop for first parameter that looks like a collection
                        collection_param = params[0]
                        for p in params:
                            if p.endswith('s') or p in ['data', 'items', 'elements', 'values', 'arr', 'list']:
                                collection_param = p
                                break

                        loop_code = f"\n    for item in {collection_param}:\n        # Process item\n"
                        updated_code += loop_code

                    elif pattern == "Missing if condition":
                        if "for " in updated_code:
                            # Add condition inside loop
                            lines = updated_code.split("\n")
                            for i, line in enumerate(lines):
                                if "for " in line and i+1 < len(lines):
                                    lines.insert(i+1, "        if item > 0:")
                                    break
                            updated_code = "\n".join(lines)
                        else:
                            # Add standalone condition
                            updated_code += "\n    if " + params[0] + " > 0:\n        # Condition body\n"

                    elif pattern == "Missing return statement":
                        # Add appropriate return
                        if "result" in updated_code:
                            updated_code += "\n    return result"
                        elif params:
                            updated_code += f"\n    return {params[0]}"
                        else:
                            updated_code += "\n    return None"

                solution = updated_code
                confidence = 0.6

        # Fill gaps if we have them
        elif gaps:
            updated_code = code_lines.copy()

            for gap_idx, gap_line in gaps:
                # Determine context for this gap
                before_context = code_lines[max(0, gap_idx-1):gap_idx]
                after_context = code_lines[gap_idx+1:min(len(code_lines), gap_idx+2)]

                # Infer appropriate code based on context
                if "return" in gap_line.lower():
                    # Fill in return statement
                    if "result" in "\n".join(body_lines):
                        replacement = "    return result"
                    elif len(params) > 0:
                        replacement = f"    return {params[0]}"
                    else:
                        replacement = "    return None"

                    updated_code[gap_idx] = replacement
                    reasoning_steps.append(f"Added return statement based on context")

                elif any("for" in line for line in before_context):
                    # We're in a loop, add loop body
                    if len(params) > 0:
                        replacement = f"        result.append({params[0]})"
                    else:
                        replacement = "        result.append(item)"

                    updated_code[gap_idx] = replacement
                    reasoning_steps.append(f"Added loop body based on context")

                elif any("if" in line for line in before_context):
                    # We're in a conditional, add conditional body
                    if "result" in "\n".join(body_lines):
                        replacement = "        result.append(item)"
                    elif len(params) > 0:
                        replacement = f"        processed.append({params[0]})"
                    else:
                        replacement = "        value += 1"

                    updated_code[gap_idx] = replacement
                    reasoning_steps.append(f"Added conditional body based on context")

                else:
                    # Generic gap filling
                    if len(params) > 0:
                        replacement = f"    result = []\n    for item in {params[0]}:\n        result.append(item)"
                    else:
                        replacement = "    result = []\n    # Process data\n    return result"

                    updated_code[gap_idx] = replacement
                    reasoning_steps.append(f"Added generic code structure")

            solution = "\n".join(updated_code)
            confidence = 0.5

        # If no specific patterns found, apply general code structure
        else:
            # Add basic structure based on function name and parameters
            if function_name.startswith(("calculate", "compute")):
                if params:
                    structure = f"\n    result = 0\n    for item in {params[0]}:\n        result += item\n    return result"
                else:
                    structure = "\n    result = 0\n    # Calculation logic\n    return result"

                solution += structure
                reasoning_steps.append("Added calculation structure based on function name")

            elif function_name.startswith(("process", "transform")):
                if params:
                    structure = f"\n    result = []\n    for item in {params[0]}:\n        result.append(item * 2)\n    return result"
                else:
                    structure = "\n    result = []\n    # Transformation logic\n    return result"

                solution += structure
                reasoning_steps.append("Added transformation structure based on function name")

            elif function_name.startswith(("filter")):
                if params:
                    structure = f"\n    result = []\n    for item in {params[0]}:\n        if item > 0:\n            result.append(item)\n    return result"
                else:
                    structure = "\n    result = []\n    # Filtering logic\n    return result"

                solution += structure
                reasoning_steps.append("Added filtering structure based on function name")

            else:
                # Generic structure
                if params:
                    structure = f"\n    result = []\n    # Process {params[0]}\n    return result"
                else:
                    structure = "\n    result = None\n    # Function logic\n    return result"

                solution += structure
                reasoning_steps.append("Added generic function structure")

            confidence = 0.4

        if not reasoning_steps:
            reasoning_steps.append("Applied inductive reasoning to complete the code")

        return {
            "solution": solution,
            "confidence": confidence,
            "reasoning_steps": reasoning_steps
        }

    def _inductive_function_inference(self, task):
        """Use inductive reasoning for function inference"""
        input_output_pairs = task.get("input_output_pairs", [])

        if not input_output_pairs:
            return {
                "solution": "f(x) = x",
                "confidence": 0.1,
                "reasoning_steps": ["No input-output pairs provided"]
            }

        # Extract inputs and outputs
        inputs = [x for x, _ in input_output_pairs]
        outputs = [y for _, y in input_output_pairs]

        # Look for patterns in the data
        reasoning_steps = ["Analyzed input-output pairs to identify patterns"]

        # Check if all outputs are the same
        if all(y == outputs[0] for y in outputs):
            return {
                "solution": f"f(x) = {outputs[0]}",
                "confidence": 0.95,
                "reasoning_steps": reasoning_steps + ["All outputs have the same value",
                                                    "Function is a constant function"]
            }

        # Check if output = input
        if all(abs(x - y) < 0.001 for (x, y) in input_output_pairs):
            return {
                "solution": "f(x) = x",
                "confidence": 0.95,
                "reasoning_steps": reasoning_steps + ["Outputs equal inputs",
                                                    "Function is the identity function"]
            }

        # Try polynomial regression
        try:
            # Try linear regression first
            n = len(input_output_pairs)
            sum_x = sum(inputs)
            sum_y = sum(outputs)
            sum_x2 = sum(x**2 for x in inputs)
            sum_xy = sum(x*y for x, y in input_output_pairs)

            # Calculate slope and intercept
            m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
            b = (sum_y - m * sum_x) / n

            # Check fit
            predictions = [m * x + b for x in inputs]
            errors = [abs(pred - actual) for pred, actual in zip(predictions, outputs)]
            avg_error = sum(errors) / len(errors)

            # Round coefficients for cleaner presentation
            m_rounded = round(m, 4)
            b_rounded = round(b, 4)

            # If linear fit is good
            if avg_error < 0.01:
                # Clean representation
                formula = "f(x) = "
                if m_rounded == 1:
                    formula += "x"
                elif m_rounded == -1:
                    formula += "-x"
                elif m_rounded != 0:
                    formula += f"{m_rounded}x"

                if b_rounded > 0:
                    if m_rounded != 0:
                        formula += f" + {b_rounded}"
                    else:
                        formula += f"{b_rounded}"
                elif b_rounded < 0:
                    formula += f" - {abs(b_rounded)}"

                return {
                    "solution": formula,
                    "confidence": 0.9,
                    "reasoning_steps": reasoning_steps + [
                        "Applied linear regression",
                        f"Found linear relationship with slope {m_rounded} and intercept {b_rounded}",
                        f"Average error: {avg_error:.6f}"
                    ]
                }

            # If linear fit is not good, try quadratic
            if avg_error >= 0.01 and len(inputs) >= 3:
                # Set up the matrices for quadratic regression
                X = np.vstack([np.ones(n), inputs, [x**2 for x in inputs]]).T
                y = np.array(outputs)

                # Solve for coefficients
                coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                a, b, c = coeffs

                # Check fit
                predictions = [a + b*x + c*x**2 for x in inputs]
                errors = [abs(pred - actual) for pred, actual in zip(predictions, outputs)]
                avg_error = sum(errors) / len(errors)

                # Round coefficients
                a_rounded = round(a, 4)
                b_rounded = round(b, 4)
                c_rounded = round(c, 4)

                # If quadratic fit is good
                if avg_error < 0.01:
                    # Clean representation
                    formula = "f(x) = "
                    terms = []

                    if c_rounded != 0:
                        if c_rounded == 1:
                            terms.append("x²")
                        elif c_rounded == -1:
                            terms.append("-x²")
                        else:
                            terms.append(f"{c_rounded}x²")

                    if b_rounded != 0:
                        if b_rounded == 1:
                            terms.append("x")
                        elif b_rounded == -1:
                            terms.append("-x")
                        else:
                            terms.append(f"{b_rounded}x")

                    if a_rounded != 0 or not terms:
                        terms.append(f"{a_rounded}")

                    formula += " + ".join(terms).replace("+ -", "- ")

                    return {
                        "solution": formula,
                        "confidence": 0.85,
                        "reasoning_steps": reasoning_steps + [
                            "Applied quadratic regression",
                            f"Found quadratic relationship: f(x) = {c_rounded}x² + {b_rounded}x + {a_rounded}",
                            f"Average error: {avg_error:.6f}"
                        ]
                    }

            # Return the best fit we found with lower confidence
            if avg_error < 0.1:
                return {
                    "solution": f"f(x) = {m_rounded}x + {b_rounded} (approximate)",
                    "confidence": 0.6,
                    "reasoning_steps": reasoning_steps + [
                        "Applied linear regression",
                        f"Found approximate linear relationship",
                        f"Average error: {avg_error:.6f}"
                    ]
                }

        except Exception:
            # If regression fails, continue to other approaches
            pass

        # Try to infer pattern from data points
        # Check if it's a modular operation
        for mod in range(2, 10):
            # Check if output = input % mod
            if all(abs(y - (x % mod)) < 0.001 for x, y in input_output_pairs):
                return {
                    "solution": f"f(x) = x % {mod}",
                    "confidence": 0.9,
                    "reasoning_steps": reasoning_steps + [
                        f"Identified modular arithmetic pattern",
                        f"Function takes remainder when dividing by {mod}"
                    ]
                }

        # Check if it's a complex pattern by checking ratios, differences etc.
        # This is a simplified approach - a complete solution would use more sophisticated pattern detection

        # Return a low-confidence approximate solution
        return {
            "solution": "f(x) = complex function (not identified)",
            "confidence": 0.3,
            "reasoning_steps": reasoning_steps + [
                "Could not identify a simple mathematical pattern",
                "Function may involve complex operations or conditional logic"
            ]
        }

    def _inductive_sequence_completion(self, task):
        """Use inductive reasoning for sequence completion"""
        sequence = task.get("visible_sequence", [])

        if not sequence or len(sequence) < 2:
            return {
                "solution": [1, 2, 3],
                "confidence": 0.1,
                "reasoning_steps": ["Insufficient data to infer pattern"]
            }

        # Start building sequences database
        sequence_patterns = {}

        # Get length of the growing pattern
        n = len(sequence)

        # Try generating sequence with different formulas and see what matches

        # Linear formula: an = a1 + (n-1)d
        for a1 in range(-5, 6):
            for d in range(-5, 6):
                if d == 0:
                    continue

                generated = [a1 + (i)*d for i in range(n)]
                error = sum(abs(a - b) for a, b in zip(sequence, generated))

                if error < 0.001:
                    formula = f"a_n = {a1} + (n-1)*{d}"
                    sequence_patterns[formula] = {
                        "error": error,
                        "prediction": [a1 + n*d, a1 + (n+1)*d, a1 + (n+2)*d],
                        "conf": 0.9,
                        "type": "arithmetic"
                    }

        # Quadratic formula: an = an²+bn+c
        # Skip if we already found a good linear match
        if not sequence_patterns:
            try:
                # Set up matrices for quadratic regression
                X = np.vstack([np.ones(n), range(n), [i**2 for i in range(n)]]).T
                y = np.array(sequence)

                # Solve for coefficients
                coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                c, b, a = coeffs

                # Round coefficients
                a_rounded = round(a, 4)
                b_rounded = round(b, 4)
                c_rounded = round(c, 4)

                # Check fit
                generated = [a*i**2 + b*i + c for i in range(n)]
                error = sum(abs(x - y) for x, y in zip(sequence, generated))

                if error < 0.01:
                    formula = f"a_n = {a_rounded}n² + {b_rounded}n + {c_rounded}"
                    sequence_patterns[formula] = {
                        "error": error,
                        "prediction": [a*n**2 + b*n + c, a*(n+1)**2 + b*(n+1) + c, a*(n+2)**2 + b*(n+2) + c],
                        "conf": 0.85,
                        "type": "quadratic"
                    }
            except:
                pass

        # Geometric formula: an = a1*r^(n-1)
        # Skip if we already found good matches
        if not sequence_patterns and all(s != 0 for s in sequence):
            for a1 in range(-5, 6):
                if a1 == 0:
                    continue

                # Calculate potential ratio
                ratios = [sequence[i]/sequence[i-1] for i in range(1, n)]

                if all(abs(r - ratios[0]) < 0.001 for r in ratios):
                    r = ratios[0]

                    generated = [sequence[0] * r**i for i in range(n)]
                    error = sum(abs(a - b) for a, b in zip(sequence, generated))

                    if error < 0.01:
                        formula = f"a_n = {sequence[0]} * {r:.4f}^(n-1)"
                        sequence_patterns[formula] = {
                            "error": error,
                            "prediction": [sequence[-1]*r, sequence[-1]*r**2, sequence[-1]*r**3],
                            "conf": 0.85,
                            "type": "geometric"
                        }

        # Fibonacci-like: an = an-1 + an-2
        # Skip if we already found good matches
        if not sequence_patterns and len(sequence) >= 3:
            # Check if each term is the sum of the two previous
            is_fibonacci = True
            for i in range(2, n):
                if abs(sequence[i] - (sequence[i-1] + sequence[i-2])) > 0.001:
                    is_fibonacci = False
                    break

            if is_fibonacci:
                formula = "a_n = a_(n-1) + a_(n-2) (Fibonacci-like)"

                # Calculate next terms
                next_terms = [0, 0, 0]
                next_terms[0] = sequence[-1] + sequence[-2]
                next_terms[1] = next_terms[0] + sequence[-1]
                next_terms[2] = next_terms[1] + next_terms[0]

                sequence_patterns[formula] = {
                    "error": 0,
                    "prediction": next_terms,
                    "conf": 0.9,
                    "type": "recursive"
                }

        # Find the best pattern
        if sequence_patterns:
            # Sort by error
            sorted_patterns = sorted(sequence_patterns.items(), key=lambda x: x[1]["error"])
            best_formula, best_pattern = sorted_patterns[0]

            return {
                "solution": best_pattern["prediction"],
                "confidence": best_pattern["conf"],
                "reasoning_steps": [
                    "Applied inductive reasoning to find sequence formula",
                    f"Identified {best_pattern['type']} sequence pattern",
                    f"Formula: {best_formula}",
                    f"Generated next terms using the formula"
                ]
            }

        # If no formula found, try last-difference extrapolation
        differences = [sequence[i] - sequence[i-1] for i in range(1, n)]

        # If differences are almost equal
        if max(differences) - min(differences) < 0.1:
            avg_diff = sum(differences) / len(differences)
            next_terms = [sequence[-1] + avg_diff * (i+1) for i in range(3)]

            return {
                "solution": next_terms,
                "confidence": 0.6,
                "reasoning_steps": [
                    "Could not identify exact mathematical formula",
                    f"Used average difference extrapolation",
                    f"Average difference: {avg_diff:.4f}"
                ]
            }

        # Last resort: projection based on the last few terms
        if len(sequence) >= 3:
            # Use last three terms to predict next three
            last_diffs = [sequence[-1] - sequence[-2], sequence[-2] - sequence[-3]]
            avg_last_diff = sum(last_diffs) / len(last_diffs)

            next_terms = [sequence[-1] + avg_last_diff * (i+1) for i in range(3)]

            return {
                "solution": next_terms,
                "confidence": 0.4,
                "reasoning_steps": [
                    "No standard pattern identified",
                    "Used local extrapolation based on last few terms",
                    "Low confidence prediction"
                ]
            }

        # Ultimate fallback
        return {
            "solution": [sequence[-1] + 1, sequence[-1] + 2, sequence[-1] + 3],
            "confidence": 0.2,
            "reasoning_steps": ["No clear pattern identified", "Default continuation with incrementing values"]
        }

    def _trial_and_error(self, task):
        """Apply trial and error approach (experimentation)"""
        task_type = task["type"]
        candidates = []

        # Generic trial and error approach
        if task_type == "code_completion":
            candidates.append(self._trial_and_error_code(task))

        elif task_type == "function_inference":
            candidates.append(self._trial_and_error_function(task))

        elif task_type == "math_operation":
            candidates.append(self._trial_and_error_math(task))

        else:
            # Generic trial and error
            candidates.append({
                "solution": f"Trial and error solution for {task_type}",
                "confidence": 0.3,
                "reasoning_steps": ["Systematically tested different approaches"]
            })

        return candidates

    def _trial_and_error_code(self, task):
        """Apply trial and error to code completion"""
        code = task.get("code", "")

        # Find TODOs and gaps
        code_lines = code.split("\n")
        todos = [(i, line) for i, line in enumerate(code_lines) if "TODO" in line or "..." in line]

        # If there are explicit gaps, try to fill them
        if todos:
            # Common code patterns to try
            patterns = [
                "    result = []",
                "    for item in items:",
                "        result.append(item)",
                "    return result",
                "    total = 0",
                "    for item in items:",
                "        total += item",
                "    return total",
                "    if condition:",
                "        return True",
                "    else:",
                "        return False"
            ]

            # Try filling each gap systematically
            candidates = []

            for todo_idx, todo_line in todos:
                for pattern in patterns:
                    # Create a variation with this pattern
                    code_copy = code_lines.copy()
                    code_copy[todo_idx] = pattern
                    candidate = "\n".join(code_copy)
                    candidates.append((candidate, pattern))

            # Evaluate each candidate for syntax validity
            valid_candidates = []
            for candidate_code, pattern in candidates:
                try:
                    # Check if it's syntactically valid Python
                    compile(candidate_code, "<string>", "exec")
                    valid_candidates.append((candidate_code, pattern))
                except:
                    continue

            # If we have valid candidates, return the first one
            if valid_candidates:
                solution, pattern = valid_candidates[0]
                return {
                    "solution": solution,
                    "confidence": 0.5,
                    "reasoning_steps": [
                        "Used trial and error to systematically test code patterns",
                        f"Found syntactically valid completion with pattern: {pattern}",
                        "Selected first valid solution"
                    ]
                }

        # If no explicit gaps or no valid completions found
        # Try to deduce what kind of function it is and complete accordingly

        function_name = ""
        params = []

        # Parse function signature
        for line in code_lines:
            if line.strip().startswith("def "):
                function_parts = line.split("(", 1)
                if len(function_parts) > 1:
                    function_name = function_parts[0].replace("def", "").strip()
                    param_str = function_parts[1].split(")", 1)[0]
                    params = [p.strip() for p in param_str.split(",") if p.strip()]
                break

        # Try different completions based on function name
        if "calculate" in function_name or "compute" in function_name:
            completion = "\n    result = 0\n    for item in " + (params[0] if params else "items") + ":\n        result += item\n    return result"
        elif "process" in function_name or "transform" in function_name:
            completion = "\n    result = []\n    for item in " + (params[0] if params else "items") + ":\n        result.append(item * 2)\n    return result"
        elif "filter" in function_name:
            completion = "\n    result = []\n    for item in " + (params[0] if params else "items") + ":\n        if item > 0:\n            result.append(item)\n    return result"
        else:
            completion = "\n    result = []\n    # Process data\n    return result"

        solution = code + completion

        return {
            "solution": solution,
            "confidence": 0.4,
            "reasoning_steps": [
                "Applied trial and error to complete the function",
                "Generated completion based on function name and parameters",
                "Added data processing structure with appropriate return statement"
            ]
        }

    def _trial_and_error_function(self, task):
        """Apply trial and error to function inference"""
        input_output_pairs = task.get("input_output_pairs", [])

        if not input_output_pairs:
            return {
                "solution": "f(x) = x",
                "confidence": 0.1,
                "reasoning_steps": ["No input-output pairs provided"]
            }

        # Common function templates to try
        templates = [
            (lambda x: x, "f(x) = x"),
            (lambda x: x + 1, "f(x) = x + 1"),
            (lambda x: x - 1, "f(x) = x - 1"),
            (lambda x: x * 2, "f(x) = x * 2"),
            (lambda x: x / 2, "f(x) = x / 2"),
            (lambda x: x ** 2, "f(x) = x^2"),
            (lambda x: x ** 3, "f(x) = x^3"),
            (lambda x: math.sqrt(x) if x >= 0 else float('nan'), "f(x) = sqrt(x)"),
            (lambda x: x ** 2 + x, "f(x) = x^2 + x"),
            (lambda x: x ** 2 - x, "f(x) = x^2 - x"),
            (lambda x: x + x ** 2, "f(x) = x + x^2"),
            (lambda x: 2 * x + 1, "f(x) = 2x + 1"),
            (lambda x: 2 * x - 1, "f(x) = 2x - 1"),
            (lambda x: abs(x), "f(x) = |x|"),
            (lambda x: -x, "f(x) = -x"),
            (lambda x: 1/x if x != 0 else float('inf'), "f(x) = 1/x"),
            (lambda x: math.sin(x), "f(x) = sin(x)"),
            (lambda x: math.cos(x), "f(x) = cos(x)"),
            (lambda x: math.tan(x), "f(x) = tan(x)"),
            (lambda x: math.exp(x), "f(x) = e^x"),
            (lambda x: math.log(x) if x > 0 else float('nan'), "f(x) = ln(x)"),
            (lambda x: math.log10(x) if x > 0 else float('nan'), "f(x) = log10(x)"),
            (lambda x: 2**x, "f(x) = 2^x"),
            (lambda x: 10**x, "f(x) = 10^x"),
            (lambda x: x % 2, "f(x) = x % 2"),
            (lambda x: x % 3, "f(x) = x % 3"),
            (lambda x: x % 5, "f(x) = x % 5"),
            (lambda x: round(x), "f(x) = round(x)"),
            (lambda x: math.floor(x), "f(x) = floor(x)"),
            (lambda x: math.ceil(x), "f(x) = ceiling(x)"),
            (lambda x: 1 if x > 0 else (0 if x == 0 else -1), "f(x) = sign(x)")
        ]

        # Try each template and measure error
        template_errors = []

        for fn, formula in templates:
            try:
                errors = []
                for x, y in input_output_pairs:
                    try:
                        predicted = fn(x)
                        if math.isnan(predicted) or math.isinf(predicted):
                            errors.append(float('inf'))
                        else:
                            errors.append(abs(predicted - y))
                    except:
                        errors.append(float('inf'))

                # Calculate average error
                if errors and not all(math.isinf(e) for e in errors):
                    avg_error = sum(e for e in errors if not math.isinf(e)) / len([e for e in errors if not math.isinf(e)])
                    template_errors.append((formula, avg_error))
            except:
                continue

        # Sort by error
        template_errors.sort(key=lambda x: x[1])

        # If we found a good match
        if template_errors and template_errors[0][1] < 0.01:
            best_formula, error = template_errors[0]
            return {
                "solution": best_formula,
                "confidence": 0.9,
                "reasoning_steps": [
                    "Used trial and error to test common function templates",
                    f"Found matching function with very low error ({error:.6f})",
                    f"Selected function: {best_formula}"
                ]
            }
        elif template_errors and template_errors[0][1] < 0.1:
            best_formula, error = template_errors[0]
            return {
                "solution": best_formula,
                "confidence": 0.7,
                "reasoning_steps": [
                    "Used trial and error to test common function templates",
                    f"Found close matching function with error of {error:.6f}",
                    f"Selected function: {best_formula}"
                ]
            }
        elif template_errors:
            best_formula, error = template_errors[0]
            return {
                "solution": best_formula + " (approximate)",
                "confidence": 0.4,
                "reasoning_steps": [
                    "Used trial and error to test common function templates",
                    f"Found approximate function with error of {error:.6f}",
                    f"Selected function: {best_formula}"
                ]
            }
        else:
            return {
                "solution": "Cannot determine using trial and error",
                "confidence": 0.2,
                "reasoning_steps": [
                    "Tested 30+ function templates",
                    "None provided a reasonable match to the data",
                    "Function may be complex or contain conditional logic"
                ]
            }

    def _trial_and_error_math(self, task):
        """Apply trial and error to math operations"""
        expression = task.get("expression", "")

        if not expression:
            return {
                "solution": "0",
                "confidence": 0.1,
                "reasoning_steps": ["No expression provided"]
            }

        # Try direct evaluation
        try:
            result = eval(expression)
            if isinstance(result, float):
                result = round(result, 4)

            return {
                "solution": str(result),
                "confidence": 0.95,
                "reasoning_steps": [
                    "Used trial and error by directly evaluating the expression",
                    f"Successfully calculated {expression} = {result}"
                ]
            }
        except:
            pass

        # If direct evaluation fails, try variations of the expression
        # This handles some common issues like missing operators
        variations = []

        # Add missing multiplication symbols
        if re.search(r'\d\(', expression):
            variation = re.sub(r'(\d)(\()', r'\1*\2', expression)
            variations.append((variation, "Added missing multiplication symbols"))

        # Fix missing closing parentheses
        if expression.count('(') > expression.count(')'):
            variation = expression + ')' * (expression.count('(') - expression.count(')'))
            variations.append((variation, "Added missing closing parentheses"))

        # Fix missing opening parentheses
        if expression.count(')') > expression.count('('):
            variation = '(' * (expression.count(')') - expression.count('(')) + expression
            variations.append((variation, "Added missing opening parentheses"))

        # Try each variation
        for variation, explanation in variations:
            try:
                result = eval(variation)
                if isinstance(result, float):
                    result = round(result, 4)

                return {
                    "solution": str(result),
                    "confidence": 0.8,
                    "reasoning_steps": [
                        "Expression could not be evaluated directly",
                        explanation,
                        f"Evaluated {variation} = {result}"
                    ]
                }
            except:
                continue

        # If variations fail, try extraction of numbers and operators
        try:
            # Extract all numbers from the expression
            numbers = re.findall(r'\d+\.?\d*', expression)
            numbers = [float(num) for num in numbers]

            # Extract operators
            operators = re.findall(r'[\+\-\*\/\^]', expression)

            # Try a simple left-to-right calculation
            if numbers and operators:
                result = numbers[0]
                for i, op in enumerate(operators):
                    if i + 1 < len(numbers):
                        if op == '+':
                            result += numbers[i + 1]
                        elif op == '-':
                            result -= numbers[i + 1]
                        elif op == '*':
                            result *= numbers[i + 1]
                        elif op == '/':
                            result /= numbers[i + 1]
                        elif op == '^':
                            result **= numbers[i + 1]

                if isinstance(result, float):
                    result = round(result, 4)

                return {
                    "solution": str(result),
                    "confidence": 0.5,
                    "reasoning_steps": [
                        "Expression could not be parsed normally",
                        "Extracted numbers and operators to calculate result",
                        f"Applied left-to-right evaluation: {result}"
                    ]
                }

            # If all else fails, just guess based on the numbers
            if numbers:
                # Try sum, product, etc.
                sum_result = sum(numbers)
                product_result = 1
                for num in numbers:
                    product_result *= num

                # Decide which operation makes more sense
                if '+' in expression:
                    result = sum_result
                    explanation = "sum of all numbers"
                elif '*' in expression:
                    result = product_result
                    explanation = "product of all numbers"
                else:
                    result = sum_result
                    explanation = "sum of all numbers (default)"

                if isinstance(result, float):
                    result = round(result, 4)

                return {
                    "solution": str(result),
                    "confidence": 0.3,
                    "reasoning_steps": [
                        "Could not parse expression correctly",
                        f"Used trial and error to calculate {explanation}",
                        f"Result: {result} (low confidence)"
                    ]
                }
        except:
            pass

        # Ultimate fallback
        return {
            "solution": "Cannot solve with trial and error",
            "confidence": 0.1,
            "reasoning_steps": [
                "Unable to evaluate expression through any trial and error approach",
                "Expression may be malformed or too complex"
            ]
        }

    def get_reasoning_stats(self):
        """Get statistics about reasoning engine usage"""
        total_usage = sum(self.reasoning_stats["stream_usage"].values())

        if total_usage == 0:
            return {
                "total_usage": 0,
                "stream_usage": self.reasoning_stats["stream_usage"],
                "average_depth": 0,
                "success_rates": {stream: 0.0 for stream in self.reasoning_streams}
            }

        # Calculate statistics
        stats = {
            "total_usage": total_usage,
            "stream_usage": self.reasoning_stats["stream_usage"],
            "usage_percentages": {
                stream: count / total_usage
                for stream, count in self.reasoning_stats["stream_usage"].items()
            },
            "average_depth": self.reasoning_stats["average_depth"],
            "success_rates": {}
        }

        # Calculate success rates
        for stream, rates in self.reasoning_stats["success_rates"].items():
            if rates:
                stats["success_rates"][stream] = sum(rates) / len(rates)
            else:
                stats["success_rates"][stream] = 0.0

        return stats


class VerificationMechanism:
    """Verifies solutions using internal consistency and simulations"""

    def __init__(self, model):
        """Initialize verification mechanism"""
        self.model = model
        self.verification_history = []

        # Set up verification metrics
        self.verification_metrics = {
            "total_verifications": 0,
            "successful_verifications": 0,
            "failure_reasons": Counter()
        }

    def verify_solutions(self, task, solution_candidates):
        """Verify a set of solution candidates for a task"""
        if not solution_candidates:
            return {
                "is_valid": False,
                "best_solution": None,
                "reason": "No solution candidates provided"
            }

        # Extract task type
        task_type = task["type"]

        # Select appropriate verification method
        if task_type == "code_completion":
            return self._verify_code_completion(task, solution_candidates)

        elif task_type == "function_inference":
            return self._verify_function_inference(task, solution_candidates)

        elif task_type == "math_operation":
            return self._verify_math_operation(task, solution_candidates)

        elif task_type == "sequence_completion":
            return self._verify_sequence_completion(task, solution_candidates)

        elif task_type in ["logic_puzzle", "propositional_logic", "syllogism"]:
            return self._verify_logic_puzzle(task, solution_candidates)

        else:
            # Generic verification based on confidence
            return self._generic_verification(task, solution_candidates)

    def _verify_code_completion(self, task, solution_candidates):
        """Verify code completion candidates"""
        # Update metrics
        self.verification_metrics["total_verifications"] += 1

        # Load candidates from lowest confidence to highest
        candidates = sorted(solution_candidates, key=lambda x: x.get("confidence", 0))

        valid_candidates = []
        verification_details = []

        for candidate in candidates:
            solution = candidate.get("solution", "")
            confidence = candidate.get("confidence", 0)

            # Skip empty solutions
            if not solution:
                verification_details.append({
                    "solution": "empty",
                    "is_valid": False,
                    "reason": "Empty solution"
                })
                continue

            # Check for syntax errors
            try:
                compile(solution, "<string>", "exec")
                syntax_valid = True
                reason = "Syntax is valid"
            except Exception as e:
                syntax_valid = False
                reason = f"Syntax error: {str(e)}"

            # Check if solution completes the task
            task_completed = False

            if syntax_valid:
                # Check if TODOs are removed
                task_completed = "TODO" not in solution and "..." not in solution

                # Check if function has a return statement
                if "def " in solution and "return " not in solution:
                    task_completed = False
                    reason = "Missing return statement"

                # Check if function is using its parameters
                function_params = []
                for line in solution.split("\n"):
                    if line.strip().startswith("def "):
                        param_str = line.split("(", 1)[1].split(")", 1)[0]
                        function_params = [p.strip() for p in param_str.split(",") if p.strip()]
                        break

                # Check if all parameters are used in the body
                for param in function_params:
                    # Skip self parameter for methods
                    if param == "self":
                        continue

                    param_name = param.split(":", 1)[0].split("=", 1)[0].strip()
                    if param_name not in solution.split("def ", 1)[1]:
                        task_completed = False
                        reason = f"Parameter '{param_name}' is not used"

            is_valid = syntax_valid and task_completed

            verification_details.append({
                "solution": solution[:50] + "..." if len(solution) > 50 else solution,
                "is_valid": is_valid,
                "syntax_valid": syntax_valid,
                "task_completed": task_completed,
                "reason": reason,
                "confidence": confidence
            })

            if is_valid:
                valid_candidates.append((candidate, confidence))

        # Return the best valid candidate, or the highest confidence candidate if none are valid
        if valid_candidates:
            self.verification_metrics["successful_verifications"] += 1
            best_candidate, _ = max(valid_candidates, key=lambda x: x[1])

            return {
                "is_valid": True,
                "best_solution": best_candidate.get("solution"),
                "verification_details": verification_details,
                "confidence": best_candidate.get("confidence", 0),
                "reasoning_stream": best_candidate.get("reasoning_stream", "unknown")
            }
        else:
            # If no valid candidates, return the highest confidence one with a warning
            self.verification_metrics["failure_reasons"]["no_valid_code"] += 1
            best_candidate = max(solution_candidates, key=lambda x: x.get("confidence", 0))

            return {
                "is_valid": False,
                "best_solution": best_candidate.get("solution"),
                "verification_details": verification_details,
                "reason": "No fully valid solutions found",
                "confidence": best_candidate.get("confidence", 0) * 0.8,  # Reduce confidence
                "reasoning_stream": best_candidate.get("reasoning_stream", "unknown")
            }

    def _verify_function_inference(self, task, solution_candidates):
        """Verify function inference candidates"""
        # Update metrics
        self.verification_metrics["total_verifications"] += 1

        input_output_pairs = task.get("input_output_pairs", [])

        if not input_output_pairs:
            self.verification_metrics["failure_reasons"]["no_input_output_pairs"] += 1
            return {
                "is_valid": False,
                "best_solution": None,
                "reason": "No input-output pairs available for verification"
            }

        # Evaluate each candidate
        candidate_scores = []
        verification_details = []

        for candidate in solution_candidates:
            solution = candidate.get("solution", "")
            confidence = candidate.get("confidence", 0)

            # Skip empty solutions
            if not solution:
                verification_details.append({
                    "solution": "empty",
                    "is_valid": False,
                    "reason": "Empty solution"
                })
                continue

            # Try to convert the solution into a function
            func = self._parse_function_formula(solution)

            if func:
                # Test the function on input-output pairs
                errors = []
                for x, expected_y in input_output_pairs:
                    try:
                        actual_y = func(x)
                        errors.append(abs(actual_y - expected_y))
                    except:
                        errors.append(float('inf'))

                # Calculate mean error
                if errors and not all(math.isinf(e) for e in errors):
                    mean_error = sum(e for e in errors if not math.isinf(e)) / len([e for e in errors if not math.isinf(e)])

                    is_valid = mean_error < 0.01  # Allow small floating point errors

                    verification_details.append({
                        "solution": solution,
                        "is_valid": is_valid,
                        "mean_error": mean_error,
                        "confidence": confidence
                    })

                    # Adjust confidence based on error
                    adjusted_confidence = confidence * max(0.1, 1.0 - min(1.0, mean_error * 10))
                    candidate_scores.append((candidate, adjusted_confidence, mean_error))
                else:
                    verification_details.append({
                        "solution": solution,
                        "is_valid": False,
                        "reason": "Function produced errors or infinity",
                        "confidence": confidence
                    })
            else:
                verification_details.append({
                    "solution": solution,
                    "is_valid": False,
                    "reason": "Could not parse function formula",
                    "confidence": confidence
                })

        # Sort candidates by adjusted confidence
        candidate_scores.sort(key=lambda x: (x[2], -x[1]))  # Sort by error (asc) then confidence (desc)

        # Return the best candidate
        if candidate_scores:
            best_candidate, adjusted_confidence, mean_error = candidate_scores[0]

            # Only consider valid if error is very small
            is_valid = mean_error < 0.01

            if is_valid:
                self.verification_metrics["successful_verifications"] += 1
            else:
                self.verification_metrics["failure_reasons"]["high_function_error"] += 1

            return {
                "is_valid": is_valid,
                "best_solution": best_candidate.get("solution"),
                "verification_details": verification_details,
                "mean_error": mean_error,
                "confidence": adjusted_confidence,
                "reasoning_stream": best_candidate.get("reasoning_stream", "unknown")
            }
        else:
            # No valid candidates
            self.verification_metrics["failure_reasons"]["no_valid_functions"] += 1

            # Return highest confidence candidate
            best_candidate = max(solution_candidates, key=lambda x: x.get("confidence", 0))

            return {
                "is_valid": False,
                "best_solution": best_candidate.get("solution"),
                "verification_details": verification_details,
                "reason": "No valid function formulas found",
                "confidence": best_candidate.get("confidence", 0) * 0.5,  # Significantly reduce confidence
                "reasoning_stream": best_candidate.get("reasoning_stream", "unknown")
            }

    def _parse_function_formula(self, formula_str):
        """Parse a function formula string into a callable function"""
        # Remove "f(x) = " prefix if present
        if "=" in formula_str:
            formula_str = formula_str.split("=", 1)[1].strip()

        # Replace common mathematical notations
        formula_str = formula_str.replace("^", "**")  # Exponentiation
        formula_str = formula_str.replace("√", "math.sqrt")  # Square root
        formula_str = formula_str.replace("π", "math.pi")  # Pi
        formula_str = formula_str.replace("e^", "math.exp")  # Exponential
        formula_str = formula_str.replace("ln", "math.log")  # Natural log
        formula_str = formula_str.replace("log", "math.log10")  # Log base 10
        formula_str = formula_str.replace("sin", "math.sin")  # Sine
        formula_str = formula_str.replace("cos", "math.cos")  # Cosine
        formula_str = formula_str.replace("tan", "math.tan")  # Tangent
        formula_str = formula_str.replace("abs", "abs")  # Absolute value
        formula_str = formula_str.replace("|x|", "abs(x)")  # Absolute value

        try:
            # Create a lambda function
            func = eval(f"lambda x: {formula_str}")

            # Test that it's callable
            func(1)

            return func
        except:
            return None

    def _verify_math_operation(self, task, solution_candidates):
        """Verify math operation candidates"""
        # Update metrics
        self.verification_metrics["total_verifications"] += 1

        expression = task.get("expression", "")
        expected_answer = task.get("answer")

        verification_details = []

        # If we have the expected answer
        if expected_answer is not None:
            # Check each candidate against expected answer
            valid_candidates = []

            for candidate in solution_candidates:
                solution = candidate.get("solution", "")
                confidence = candidate.get("confidence", 0)

                # Skip empty solutions
                if not solution:
                    verification_details.append({
                        "solution": "empty",
                        "is_valid": False,
                        "reason": "Empty solution"
                    })
                    continue

                # Try to convert solution to number
                try:
                    # Handle fractions
                    if "/" in solution and not any(c in solution for c in "()[]{}"):
                        parts = solution.split("/")
                        if len(parts) == 2:
                            num = float(parts[0].strip())
                            denom = float(parts[1].strip())
                            solution_value = num / denom
                        else:
                            solution_value = float(solution)
                    else:
                        solution_value = float(solution)

                    # Calculate error
                    error = abs(solution_value - expected_answer)

                    # Check if solution matches expected answer
                    is_valid = error < 0.001  # Allow small floating point errors

                    verification_details.append({
                        "solution": solution,
                        "solution_value": solution_value,
                        "expected": expected_answer,
                        "is_valid": is_valid,
                        "error": error,
                        "confidence": confidence
                    })

                    if is_valid:
                        valid_candidates.append((candidate, confidence))
                except:
                    verification_details.append({
                        "solution": solution,
                        "is_valid": False,
                        "reason": "Could not convert to number",
                        "confidence": confidence
                    })

            # Return the best valid candidate
            if valid_candidates:
                self.verification_metrics["successful_verifications"] += 1
                best_candidate, _ = max(valid_candidates, key=lambda x: x[1])

                return {
                    "is_valid": True,
                    "best_solution": best_candidate.get("solution"),
                    "verification_details": verification_details,
                    "confidence": best_candidate.get("confidence", 0),
                    "reasoning_stream": best_candidate.get("reasoning_stream", "unknown")
                }

        # If no expected answer or no valid candidates, try to verify using direct evaluation
        try:
            # Evaluate the expression
            true_result = eval(expression)

            # Check candidates against true result
            valid_candidates = []

            for candidate in solution_candidates:
                solution = candidate.get("solution", "")
                confidence = candidate.get("confidence", 0)

                # Skip empty solutions
                if not solution:
                    continue

                # Try to convert solution to number
                try:
                    # Handle fractions
                    if "/" in solution and not any(c in solution for c in "()[]{}"):
                        parts = solution.split("/")
                        if len(parts) == 2:
                            num = float(parts[0].strip())
                            denom = float(parts[1].strip())
                            solution_value = num / denom
                        else:
                            solution_value = float(solution)
                    else:
                        solution_value = float(solution)

                    # Calculate error
                    error = abs(solution_value - true_result)

                    # Check if solution matches true result
                    is_valid = error < 0.001  # Allow small floating point errors

                    verification_details.append({
                        "solution": solution,
                        "solution_value": solution_value,
                        "calculated": true_result,
                        "is_valid": is_valid,
                        "error": error,
                        "confidence": confidence
                    })

                    if is_valid:
                        valid_candidates.append((candidate, confidence))
                except:
                    continue

            # Return the best valid candidate
            if valid_candidates:
                self.verification_metrics["successful_verifications"] += 1
                best_candidate, _ = max(valid_candidates, key=lambda x: x[1])

                return {
                    "is_valid": True,
                    "best_solution": best_candidate.get("solution"),
                    "verification_details": verification_details,
                    "confidence": best_candidate.get("confidence", 0),
                    "reasoning_stream": best_candidate.get("reasoning_stream", "unknown")
                }
        except:
            # If direct evaluation fails, rely on confidence scores
            pass

        # If all verification methods fail, return highest confidence candidate
        self.verification_metrics["failure_reasons"]["no_valid_math_solution"] += 1
        best_candidate = max(solution_candidates, key=lambda x: x.get("confidence", 0))

        return {
            "is_valid": False,
            "best_solution": best_candidate.get("solution"),
            "verification_details": verification_details,
            "reason": "Could not verify against expected answer or direct evaluation",
            "confidence": best_candidate.get("confidence", 0) * 0.8,  # Reduce confidence
            "reasoning_stream": best_candidate.get("reasoning_stream", "unknown")
        }

    def _verify_sequence_completion(self, task, solution_candidates):
        """Verify sequence completion candidates"""
        # Update metrics
        self.verification_metrics["total_verifications"] += 1

        sequence = task.get("visible_sequence", [])
        expected_continuation = task.get("expected_continuation")

        verification_details = []

        # If we have the expected continuation
        if expected_continuation is not None:
            # Check each candidate against expected continuation
            valid_candidates = []

            for candidate in solution_candidates:
                solution = candidate.get("solution", [])
                confidence = candidate.get("confidence", 0)

                # Skip empty solutions
                if not solution:
                    verification_details.append({
                        "solution": "empty",
                        "is_valid": False,
                        "reason": "Empty solution"
                    })
                    continue

                # Convert to list if it's not already
                if not isinstance(solution, list):
                    try:
                        # Try to parse as list
                        if isinstance(solution, str):
                            solution = eval(solution)
                        else:
                            solution = [solution]
                    except:
                        verification_details.append({
                            "solution": str(solution),
                            "is_valid": False,
                            "reason": "Could not convert to list",
                            "confidence": confidence
                        })
                        continue

                # Limit solution to same length as expected
                solution = solution[:len(expected_continuation)]

                # Calculate error
                errors = []
                for actual, expected in zip(solution, expected_continuation):
                    try:
                        errors.append(abs(float(actual) - float(expected)))
                    except:
                        errors.append(float('inf'))

                # Check if solution matches expected continuation
                is_valid = all(e < 0.001 for e in errors)  # Allow small floating point errors

                verification_details.append({
                    "solution": str(solution),
                    "expected": expected_continuation,
                    "is_valid": is_valid,
                    "errors": errors,
                    "confidence": confidence
                })

                if is_valid:
                    valid_candidates.append((candidate, confidence))

            # Return the best valid candidate
            if valid_candidates:
                self.verification_metrics["successful_verifications"] += 1
                best_candidate, _ = max(valid_candidates, key=lambda x: x[1])

                return {
                    "is_valid": True,
                    "best_solution": best_candidate.get("solution"),
                    "verification_details": verification_details,
                    "confidence": best_candidate.get("confidence", 0),
                    "reasoning_stream": best_candidate.get("reasoning_stream", "unknown")
                }

        # If no expected continuation or no valid candidates, try to verify using pattern consistency
        all_candidate_scores = []

        for candidate in solution_candidates:
            solution = candidate.get("solution", [])
            confidence = candidate.get("confidence", 0)

            # Skip empty solutions
            if not solution:
                continue

            # Convert to list if it's not already
            if not isinstance(solution, list):
                try:
                    # Try to parse as list
                    if isinstance(solution, str):
                        solution = eval(solution)
                    else:
                        solution = [solution]
                except:
                    continue

            # Check if solution continues the pattern
            pattern_score = self._check_sequence_pattern(sequence, solution)

            verification_details.append({
                "solution": str(solution),
                "pattern_score": pattern_score,
                "is_valid": pattern_score > 0.8,
                "confidence": confidence
            })

            # Adjust confidence based on pattern score
            adjusted_confidence = confidence * pattern_score
            all_candidate_scores.append((candidate, adjusted_confidence, pattern_score))

        # Sort candidates by adjusted confidence
        all_candidate_scores.sort(key=lambda x: -x[1])

        # Return the best candidate
        if all_candidate_scores:
            best_candidate, adjusted_confidence, pattern_score = all_candidate_scores[0]

            # Only consider valid if pattern score is high
            is_valid = pattern_score > 0.8

            if is_valid:
                self.verification_metrics["successful_verifications"] += 1
            else:
                self.verification_metrics["failure_reasons"]["low_pattern_score"] += 1

            return {
                "is_valid": is_valid,
                "best_solution": best_candidate.get("solution"),
                "verification_details": verification_details,
                "pattern_score": pattern_score,
                "confidence": adjusted_confidence,
                "reasoning_stream": best_candidate.get("reasoning_stream", "unknown")
            }
        else:
            # No valid candidates
            self.verification_metrics["failure_reasons"]["no_valid_sequence"] += 1

            # Return highest confidence candidate
            best_candidate = max(solution_candidates, key=lambda x: x.get("confidence", 0))

            return {
                "is_valid": False,
                "best_solution": best_candidate.get("solution"),
                "verification_details": verification_details,
                "reason": "No valid sequence continuations found",
                "confidence": best_candidate.get("confidence", 0) * 0.6,  # Significantly reduce confidence
                "reasoning_stream": best_candidate.get("reasoning_stream", "unknown")
            }

    def _check_sequence_pattern(self, original, continuation):
        """Check if continuation follows the pattern of original sequence"""
        if not original or not continuation:
            return 0.0

        # Convert to numeric if possible
        try:
            original = [float(x) for x in original]
            continuation = [float(x) for x in continuation]
        except:
            return 0.0  # Non-numeric sequences not supported in this simple implementation

        pattern_scores = []

        # Check arithmetic sequence (constant difference)
        if len(original) >= 2:
            diffs = [original[i] - original[i-1] for i in range(1, len(original))]
            avg_diff = sum(diffs) / len(diffs)
            diff_variation = sum(abs(d - avg_diff) for d in diffs) / len(diffs)

            # If differences are consistent
            if diff_variation < 0.1:
                # Check if continuation follows the pattern
                expected = [original[-1] + avg_diff * (i+1) for i in range(len(continuation))]
                errors = [abs(a - b) for a, b in zip(continuation, expected)]

                score = 1.0 - min(1.0, sum(errors) / (len(errors) * abs(avg_diff) if avg_diff != 0 else 1.0))
                pattern_scores.append(score)

        # Check geometric sequence (constant ratio)
        if len(original) >= 2 and all(x != 0 for x in original):
            ratios = [original[i] / original[i-1] for i in range(1, len(original))]
            avg_ratio = sum(ratios) / len(ratios)
            ratio_variation = sum(abs(r - avg_ratio) for r in ratios) / len(ratios)

            # If ratios are consistent
            if ratio_variation < 0.1:
                # Check if continuation follows the pattern
                expected = [original[-1] * avg_ratio ** (i+1) for i in range(len(continuation))]
                errors = [abs(a - b) for a, b in zip(continuation, expected)]

                score = 1.0 - min(1.0, sum(errors) / (len(errors) * abs(original[-1]) if original[-1] != 0 else 1.0))
                pattern_scores.append(score)

        # Check quadratic sequence (constant second difference)
        if len(original) >= 3:
            first_diffs = [original[i] - original[i-1] for i in range(1, len(original))]
            second_diffs = [first_diffs[i] - first_diffs[i-1] for i in range(1, len(first_diffs))]
            avg_second_diff = sum(second_diffs) / len(second_diffs)
            second_diff_variation = sum(abs(d - avg_second_diff) for d in second_diffs) / len(second_diffs)

            # If second differences are consistent
            if second_diff_variation < 0.1:
                # Predict next first differences
                next_first_diffs = [first_diffs[-1] + avg_second_diff * (i+1) for i in range(len(continuation))]

                # Predict continuation
                expected = [original[-1]]
                for i in range(len(continuation)):
                    expected.append(expected[-1] + next_first_diffs[i])

                expected = expected[1:]  # Remove first element

                errors = [abs(a - b) for a, b in zip(continuation, expected)]

                score = 1.0 - min(1.0, sum(errors) / (len(errors) * abs(original[-1]) if original[-1] != 0 else 1.0))
                pattern_scores.append(score)

        # Check Fibonacci-like (each term is sum of previous two)
        if len(original) >= 3:
            is_fibonacci = True
            for i in range(2, len(original)):
                if abs(original[i] - (original[i-1] + original[i-2])) > 0.1:
                    is_fibonacci = False
                    break

            if is_fibonacci:
                # Extend to continuation
                expected = [original[-1], original[-1] + original[-2]]
                for i in range(2, len(continuation)):
                    expected.append(expected[-1] + expected[-2])

                expected = expected[:len(continuation)]

                errors = [abs(a - b) for a, b in zip(continuation, expected)]

                score = 1.0 - min(1.0, sum(errors) / (len(errors) * abs(original[-1]) if original[-1] != 0 else 1.0))
                pattern_scores.append(score)

        # Return the best score
        return max(pattern_scores) if pattern_scores else 0.0

    def _verify_logic_puzzle(self, task, solution_candidates):
        """Verify logic puzzle candidates"""
        # Update metrics
        self.verification_metrics["total_verifications"] += 1

        expected_answer = task.get("answer")

        verification_details = []

        # If we have the expected answer
        if expected_answer is not None:
            # Check each candidate against expected answer
            valid_candidates = []

            for candidate in solution_candidates:
                solution = candidate.get("solution", "")
                confidence = candidate.get("confidence", 0)

                # Skip empty solutions
                if not solution:
                    verification_details.append({
                        "solution": "empty",
                        "is_valid": False,
                        "reason": "Empty solution"
                    })
                    continue

                # Normalize solutions for comparison
                normalized_solution = self._normalize_logic_answer(solution)
                normalized_expected = self._normalize_logic_answer(expected_answer)

                # Check if solution matches expected answer
                is_valid = normalized_solution == normalized_expected

                verification_details.append({
                    "solution": solution,
                    "normalized": normalized_solution,
                    "expected": expected_answer,
                    "is_valid": is_valid,
                    "confidence": confidence
                })

                if is_valid:
                    valid_candidates.append((candidate, confidence))

            # Return the best valid candidate
            if valid_candidates:
                self.verification_metrics["successful_verifications"] += 1
                best_candidate, _ = max(valid_candidates, key=lambda x: x[1])

                return {
                    "is_valid": True,
                    "best_solution": best_candidate.get("solution"),
                    "verification_details": verification_details,
                    "confidence": best_candidate.get("confidence", 0),
                    "reasoning_stream": best_candidate.get("reasoning_stream", "unknown")
                }

        # If no expected answer or no valid candidates, return highest confidence
        self.verification_metrics["failure_reasons"]["no_valid_logic_solution"] += 1
        best_candidate = max(solution_candidates, key=lambda x: x.get("confidence", 0))

        return {
            "is_valid": False,
            "best_solution": best_candidate.get("solution"),
            "verification_details": verification_details,
            "reason": "Could not verify against expected answer",
            "confidence": best_candidate.get("confidence", 0) * 0.9,  # Slightly reduce confidence
            "reasoning_stream": best_candidate.get("reasoning_stream", "unknown")
        }

    def _normalize_logic_answer(self, answer):
        """Normalize logic puzzle answers for comparison"""
        if not answer:
            return ""

        # Convert to string
        answer_str = str(answer).lower()

        # Handle True/False
        if answer_str in ["true", "yes", "valid", "correct"]:
            return "true"
        elif answer_str in ["false", "no", "invalid", "incorrect"]:
            return "false"

        # Remove punctuation and extra spaces
        answer_str = re.sub(r'[^\w\s]', '', answer_str)
        answer_str = re.sub(r'\s+', ' ', answer_str).strip()

        return answer_str

    def _generic_verification(self, task, solution_candidates):
        """Generic verification based on confidence"""
        # Sort candidates by confidence
        sorted_candidates = sorted(solution_candidates, key=lambda x: -x.get("confidence", 0))

        if not sorted_candidates:
            self.verification_metrics["failure_reasons"]["no_candidates"] += 1
            return {
                "is_valid": False,
                "best_solution": None,
                "reason": "No solution candidates provided"
            }

        # Select highest confidence candidate
        best_candidate = sorted_candidates[0]
        confidence = best_candidate.get("confidence", 0)

        # Consider valid if confidence is high enough
        is_valid = confidence >= 0.7

        if is_valid:
            self.verification_metrics["successful_verifications"] += 1
        else:
            self.verification_metrics["failure_reasons"]["low_confidence"] += 1

        return {
            "is_valid": is_valid,
            "best_solution": best_candidate.get("solution"),
            "confidence": confidence,
            "reasoning_stream": best_candidate.get("reasoning_stream", "unknown")
        }

    def get_verification_metrics(self):
        """Get metrics about verification process"""
        success_rate = 0.0
        if self.verification_metrics["total_verifications"] > 0:
            success_rate = self.verification_metrics["successful_verifications"] / self.verification_metrics["total_verifications"]

        return {
            "total_verifications": self.verification_metrics["total_verifications"],
            "successful_verifications": self.verification_metrics["successful_verifications"],
            "success_rate": success_rate,
            "failure_reasons": dict(self.verification_metrics["failure_reasons"])
        }


class BenchmarkManager:
    """Manages internal benchmarks for evaluating SAM's capabilities"""

    def __init__(self, model):
        """Initialize benchmark manager"""
        self.model = model

        # Define benchmark categories
        self.benchmark_categories = [
            "code_generation",
            "mathematical_reasoning",
            "pattern_recognition",
            "logical_reasoning"
        ]

        # History of benchmark results
        self.benchmark_history = []

        # Initialize benchmarks
        self._initialize_benchmarks()

    def _initialize_benchmarks(self):
        """Initialize benchmark tasks"""
        self.benchmarks = {
            "code_generation": self._create_code_benchmarks(),
            "mathematical_reasoning": self._create_math_benchmarks(),
            "pattern_recognition": self._create_pattern_benchmarks(),
            "logical_reasoning": self._create_logic_benchmarks()
        }

    def _create_code_benchmarks(self):
        """Create code generation benchmarks"""
        return [
            {
                "name": "factorial_function",
                "type": "code_completion",
                "difficulty": 2,
                "content": {
                    "code": "def factorial(n):\n    # TODO: Implement factorial function",
                    "function_name": "factorial",
                    "parameters": ["n"]
                },
                "expected_output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
                "test_cases": [(0, 1), (1, 1), (5, 120)]
            },
            {
                "name": "is_palindrome",
                "type": "code_completion",
                "difficulty": 2,
                "content": {
                    "code": "def is_palindrome(text):\n    # TODO: Check if text is a palindrome",
                    "function_name": "is_palindrome",
                    "parameters": ["text"]
                },
                "expected_output": "def is_palindrome(text):\n    text = text.lower()\n    return text == text[::-1]",
                "test_cases": [("radar", True), ("hello", False), ("Racecar", True)]
            },
            {
                "name": "find_duplicates",
                "type": "code_completion",
                "difficulty": 3,
                "content": {
                    "code": "def find_duplicates(numbers):\n    # TODO: Find all duplicate numbers",
                    "function_name": "find_duplicates",
                    "parameters": ["numbers"]
                },
                "expected_output": "def find_duplicates(numbers):\n    seen = set()\n    duplicates = set()\n    for num in numbers:\n        if num in seen:\n            duplicates.add(num)\n        else:\n            seen.add(num)\n    return list(duplicates)",
                "test_cases": [([1, 2, 3, 2, 1], [1, 2]), ([1, 2, 3], []), ([1, 1, 1, 2, 2], [1, 2])]
            },
            {
                "name": "binary_search",
                "type": "code_completion",
                "difficulty": 4,
                "content": {
                    "code": "def binary_search(sorted_list, target):\n    # TODO: Implement binary search",
                    "function_name": "binary_search",
                    "parameters": ["sorted_list", "target"]
                },
                "expected_output": "def binary_search(sorted_list, target):\n    left, right = 0, len(sorted_list) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if sorted_list[mid] == target:\n            return mid\n        elif sorted_list[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
                "test_cases": [([1, 2, 3, 4, 5], 3, 2), ([1, 2, 3, 4, 5], 6, -1)]
            }
        ]

    def _create_math_benchmarks(self):
        """Create mathematical reasoning benchmarks"""
        return [
            {
                "name": "basic_arithmetic",
                "type": "math_operation",
                "difficulty": 1,
                "content": {
                    "expression": "3 + 4 * 2"
                },
                "expected_output": "11",
                "test_cases": []
            },
            {
                "name": "order_of_operations",
                "type": "math_operation",
                "difficulty": 2,
                "content": {
                    "expression": "(7 + 3) * (6 - 2) / 2"
                },
                "expected_output": "20",
                "test_cases": []
            },
            {
                "name": "complex_expression",
                "type": "math_operation",
                "difficulty": 3,
                "content": {
                    "expression": "2 ** 3 + 4 * (5 - 2) ** 2"
                },
                "expected_output": "44",
                "test_cases": []
            },
            {
                "name": "linear_equation",
                "type": "function_inference",
                "difficulty": 2,
                "content": {
                    "input_output_pairs": [(1, 3), (2, 5), (3, 7), (4, 9)]
                },
                "expected_output": "f(x) = 2x + 1",
                "test_cases": [(5, 11), (0, 1)]
            },
            {
                "name": "quadratic_function",
                "type": "function_inference",
                "difficulty": 3,
                "content": {
                    "input_output_pairs": [(0, 1), (1, 2), (2, 5), (3, 10), (4, 17)]
                },
                "expected_output": "f(x) = x^2 + 1",
                "test_cases": [(5, 26), (-1, 2)]
            }
        ]

    def _create_pattern_benchmarks(self):
        """Create pattern recognition benchmarks"""
        return [
            {
                "name": "fibonacci_sequence",
                "type": "sequence_completion",
                "difficulty": 2,
                "content": {
                    "visible_sequence": [1, 1, 2, 3, 5, 8]
                },
                "expected_output": [13, 21, 34],
                "test_cases": []
            },
            {
                "name": "arithmetic_sequence",
                "type": "sequence_completion",
                "difficulty": 1,
                "content": {
                    "visible_sequence": [3, 7, 11, 15, 19]
                },
                "expected_output": [23, 27, 31],
                "test_cases": []
            },
            {
                "name": "geometric_sequence",
                "type": "sequence_completion",
                "difficulty": 2,
                "content": {
                    "visible_sequence": [2, 6, 18, 54, 162]
                },
                "expected_output": [486, 1458, 4374],
                "test_cases": []
            },
            {
                "name": "quadratic_sequence",
                "type": "sequence_completion",
                "difficulty": 3,
                "content": {
                    "visible_sequence": [0, 1, 4, 9, 16, 25]
                },
                "expected_output": [36, 49, 64],
                "test_cases": []
            },
            {
                "name": "complex_pattern",
                "type": "sequence_completion",
                "difficulty": 4,
                "content": {
                    "visible_sequence": [2, 1, 3, 4, 7, 11, 18]
                },
                "expected_output": [29, 47, 76],
                "test_cases": []
            }
        ]

    def _create_logic_benchmarks(self):
        """Create logical reasoning benchmarks"""
        return [
            {
                "name": "modus_ponens",
                "type": "propositional_logic",
                "difficulty": 1,
                "content": {
                    "statements": [
                        "If it rains, then the ground is wet",
                        "It is raining"
                    ],
                    "conclusion": "Therefore, the ground is wet"
                },
                "expected_output": "True",
                "test_cases": []
            },
            {
                "name": "modus_tollens",
                "type": "propositional_logic",
                "difficulty": 2,
                "content": {
                    "statements": [
                        "If it rains, then the ground is wet",
                        "The ground is not wet"
                    ],
                    "conclusion": "Therefore, it is not raining"
                },
                "expected_output": "True",
                "test_cases": []
            },
            {
                "name": "invalid_syllogism",
                "type": "syllogism",
                "difficulty": 3,
                "content": {
                    "statements": [
                        "All birds can fly",
                        "Penguins are birds"
                    ],
                    "conclusion": "Therefore, penguins can fly"
                },
                "expected_output": "False",
                "test_cases": []
            },
            {
                "name": "conditional_reasoning",
                "type": "logic_puzzle",
                "difficulty": 3,
                "content": {
                    "statements": [
                        "If the alarm rings, John wakes up",
                        "If John wakes up, he is late for work",
                        "The alarm rings"
                    ],
                    "conclusion": "Therefore, John is late for work"
                },
                "expected_output": "True",
                "test_cases": []
            }
        ]


    def run_benchmarks(self, categories=None, max_benchmarks=3):
        """Run benchmarks to evaluate model capabilities"""
        # Select categories to run
        if not categories:
            categories = self.benchmark_categories
        elif isinstance(categories, str):
            categories = [categories]

        # Filter to requested categories
        categories = [c for c in categories if c in self.benchmark_categories]

        if not categories:
            return {"error": "No valid benchmark categories specified"}

        # Run benchmarks for each category
        results = {}

        for category in categories:
            # Get benchmarks for this category
            benchmarks = self.benchmarks.get(category, [])

            # Limit number of benchmarks per category
            selected_benchmarks = benchmarks[:max_benchmarks]

            category_results = []

            # Run each benchmark
            for benchmark in selected_benchmarks:
                # Create task from benchmark
                task = {
                    "id": f"benchmark_{benchmark['name']}",
                    "type": benchmark["type"],
                    "complexity": benchmark["difficulty"],
                    "content": benchmark["content"]
                }

                # Try to solve the benchmark
                solution_candidates = self.model.reasoning_engine.solve_task(task)

                # Verify solutions
                verification_results = self.model.verification_mechanism.verify_solutions(
                    task, solution_candidates)

                # Get best solution
                best_solution = verification_results.get("best_solution")
                is_valid = verification_results.get("is_valid", False)

                # Check against expected output for additional verification
                expected_output = benchmark.get("expected_output")

                if expected_output is not None and best_solution is not None:
                    # Compare with expected output
                    if isinstance(best_solution, list):
                        solution_str = str(best_solution)
                    else:
                        solution_str = str(best_solution)

                    expected_str = str(expected_output)

                    # Normalize for comparison
                    solution_norm = self._normalize_benchmark_output(solution_str)
                    expected_norm = self._normalize_benchmark_output(expected_str)

                    # Check if solution matches expected output
                    output_match = solution_norm == expected_norm

                    # Run test cases if available
                    test_results = []

                    if benchmark.get("test_cases") and benchmark["type"] == "code_completion":
                        # For code tasks, actually run the code if possible
                        test_results = self._run_code_tests(best_solution, benchmark["test_cases"])

                    # Determine final score (0-100)
                    if output_match and (not test_results or all(test_results)):
                        score = 100
                    elif output_match:
                        # Some test failures
                        success_ratio = sum(1 for t in test_results if t) / len(test_results)
                        score = int(70 + 30 * success_ratio)
                    elif is_valid and (not test_results or any(test_results)):
                        # Output doesn't match but solution is valid and passes some tests
                        success_ratio = sum(1 for t in test_results if t) / max(1, len(test_results))
                        score = int(40 + 40 * success_ratio)
                    elif is_valid:
                        # Output doesn't match and fails tests but is valid
                        score = 30
                    else:
                        # Not valid
                        score = 0
                else:
                    # No expected output specified
                    score = 100 if is_valid else 0
                    output_match = None
                    test_results = []

                # Record benchmark result
                result = {
                    "name": benchmark["name"],
                    "type": benchmark["type"],
                    "difficulty": benchmark["difficulty"],
                    "score": score,
                    "is_valid": is_valid,
                    "output_match": output_match,
                    "test_results": test_results
                }

                category_results.append(result)

            # Calculate category score (weighted by difficulty)
            total_difficulty = sum(result["difficulty"] for result in category_results)
            if total_difficulty > 0:
                weighted_score = sum(result["score"] * result["difficulty"] for result in category_results) / total_difficulty
            else:
                weighted_score = 0

            results[category] = {
                "score": round(weighted_score, 1),
                "benchmarks": category_results
            }

        # Calculate overall score
        if results:
            overall_score = sum(cat_result["score"] for cat_result in results.values()) / len(results)
        else:
            overall_score = 0

        # Record benchmark history
        benchmark_summary = {
            "timestamp": time.time(),
            "overall_score": round(overall_score, 1),
            "category_scores": {cat: results[cat]["score"] for cat in results},
            "categories_run": list(results.keys())
        }

        self.benchmark_history.append(benchmark_summary)

        # Include summary in results
        results["overall_score"] = round(overall_score, 1)
        results["timestamp"] = time.time()

        return results

    def _normalize_benchmark_output(self, output):
        """Normalize benchmark output for comparison"""
        if not output:
            return ""

        # Convert to string
        output_str = str(output).lower()

        # Handle True/False
        if output_str in ["true", "yes", "valid", "correct"]:
            return "true"
        elif output_str in ["false", "no", "invalid", "incorrect"]:
            return "false"

        # Remove whitespace, punctuation, etc.
        output_str = re.sub(r'\s+', '', output_str)
        output_str = re.sub(r'[^\w\d\[\],{}()]', '', output_str)

        return output_str

    def _run_code_tests(self, code_solution, test_cases):
        """Run code tests for function benchmarks"""
        if not code_solution or not test_cases:
            return []

        # Extract function name
        function_name = None
        for line in code_solution.split("\n"):
            if line.strip().startswith("def "):
                function_name = line.split("(")[0].replace("def", "").strip()
                break

        if not function_name:
            return [False] * len(test_cases)

        # Try to execute the code and run tests
        try:
            # Create a safe namespace
            namespace = {}

            # Execute the function definition
            exec(code_solution, namespace)

            # Get the function
            function = namespace.get(function_name)

            if not function or not callable(function):
                return [False] * len(test_cases)

            # Run each test case
            results = []

            for test_case in test_cases:
                try:
                    # Unpack test case
                    if len(test_case) == 2:
                        args, expected = test_case[0], test_case[1]
                    else:
                        *args, expected = test_case

                    # Handle single vs. multiple arguments
                    if isinstance(args, tuple) and len(args) == 1:
                        result = function(args[0])
                    else:
                        result = function(*args)

                    # Check if result matches expected
                    results.append(result == expected)
                except:
                    results.append(False)

            return results
        except:
            # If code execution fails
            return [False] * len(test_cases)

    def get_benchmark_history(self, limit=10):
        """Get history of benchmark results"""
        return self.benchmark_history[-limit:]


class KnowledgeAcquisition:
    """Handles autonomous knowledge acquisition for SAM"""

    def __init__(self, model):
        """Initialize knowledge acquisition system"""
        self.model = model

        # Knowledge growth tracking
        self.knowledge_growth = {
            "concepts_added": 0,
            "patterns_added": 0,
            "relationships_formed": 0,
            "concepts_by_domain": defaultdict(int)
        }

        # Knowledge domains and focus areas
        self.knowledge_domains = [
            "programming", "mathematics", "logic", "science",
            "language", "reasoning", "patterns", "concepts"
        ]

        # Current focus areas (domains currently being emphasized)
        self.focus_areas = []

        # Knowledge acquisition history
        self.acquisition_history = []

    def acquire_knowledge(self, focus_areas=None, count=5):
        """Acquire new knowledge based on focus areas"""
        # Update focus areas if provided
        if focus_areas:
            self.focus_areas = focus_areas

        if not self.focus_areas:
            # Default to all domains
            self.focus_areas = self.knowledge_domains

        # Track acquisition results
        results = {
            "concepts_added": 0,
            "patterns_added": 0,
            "relationships_formed": 0,
            "focus_areas": self.focus_areas.copy(),
            "timestamp": time.time()
        }

        # Determine acquisition methods based on focus areas
        acquisition_methods = []

        for area in self.focus_areas:
            if area in ["programming", "code_completion", "function_inference"]:
                acquisition_methods.append(self._acquire_programming_knowledge)

            elif area in ["mathematics", "math_operation", "sequence_completion"]:
                acquisition_methods.append(self._acquire_mathematical_knowledge)

            elif area in ["logic", "reasoning", "logic_puzzle", "propositional_logic"]:
                acquisition_methods.append(self._acquire_logical_knowledge)

            elif area in ["patterns", "pattern_recognition"]:
                acquisition_methods.append(self._acquire_pattern_knowledge)

            else:
                # Default for unknown areas
                acquisition_methods.append(self._acquire_general_knowledge)

        # Deduplicate methods
        acquisition_methods = list(set(acquisition_methods))

        # Apply each acquisition method
        for method in acquisition_methods:
            method_results = method(count=count // len(acquisition_methods) + 1)

            # Add results to total
            results["concepts_added"] += method_results.get("concepts_added", 0)
            results["patterns_added"] += method_results.get("patterns_added", 0)
            results["relationships_formed"] += method_results.get("relationships_formed", 0)

            # Add method-specific results
            method_name = method.__name__.replace("_acquire_", "").replace("_knowledge", "")
            results[f"{method_name}_results"] = method_results

        # Update knowledge growth tracking
        self.knowledge_growth["concepts_added"] += results["concepts_added"]
        self.knowledge_growth["patterns_added"] += results["patterns_added"]
        self.knowledge_growth["relationships_formed"] += results["relationships_formed"]

        for area in self.focus_areas:
            self.knowledge_growth["concepts_by_domain"][area] += results["concepts_added"] // len(self.focus_areas)

        # Record acquisition history
        self.acquisition_history.append(results)

        return results

    def _acquire_programming_knowledge(self, count=3):
        """Acquire programming-related knowledge"""
        results = {
            "concepts_added": 0,
            "patterns_added": 0,
            "relationships_formed": 0,
            "specific_concepts": []
        }

        # Programming concepts to potentially add
        programming_concepts = [
            # Data structures
            ("list", "A mutable ordered collection of elements"),
            ("dictionary", "A key-value mapping structure"),
            ("tuple", "An immutable ordered collection of elements"),
            ("set", "An unordered collection of unique elements"),

            # Control structures
            ("if-else", "Conditional branching based on a boolean expression"),
            ("for-loop", "Iteration over a sequence of elements"),
            ("while-loop", "Repeated execution while a condition is true"),
            ("try-except", "Error handling mechanism"),

            # Functions and methods
            ("function", "A reusable block of code that performs a specific task"),
            ("method", "A function associated with an object or class"),
            ("parameter", "An input value to a function"),
            ("return-value", "The output value of a function"),

            # Object-oriented concepts
            ("class", "A blueprint for creating objects"),
            ("object", "An instance of a class"),
            ("inheritance", "A mechanism where a class inherits properties from a parent class"),
            ("polymorphism", "The ability to present the same interface for different underlying implementations"),

            # Algorithms
            ("sorting", "Arranging elements in a specific order"),
            ("searching", "Finding a target element in a collection"),
            ("recursion", "A function that calls itself"),
            ("iteration", "Repeated execution of a code block")
        ]

        # Filter to concepts we don't already have
        existing_concepts = set()
        if hasattr(self.model, "concept_bank") and hasattr(self.model.concept_bank, "source_to_concept"):
            existing_concepts = set(self.model.concept_bank.source_to_concept.keys())

        new_concepts = [(name, desc) for name, desc in programming_concepts if name not in existing_concepts]

        # Select random concepts up to count
        selected_concepts = random.sample(new_concepts, min(count, len(new_concepts)))

        # Add concepts to concept bank
        for name, description in selected_concepts:
            try:
                if hasattr(self.model, "concept_bank") and hasattr(self.model.concept_bank, "add_character_concept"):
                    self.model.concept_bank.add_character_concept(name)

                    # Record added concept
                    results["concepts_added"] += 1
                    results["specific_concepts"].append(name)

                    # Also add related concepts and form relationships
                    related_concepts = []
                    if "loop" in name:
                        related_concepts = ["iteration", "control flow", "looping"]
                    elif "function" in name or "method" in name:
                        related_concepts = ["callable", "procedure", "subroutine"]
                    elif "class" in name or "object" in name:
                        related_concepts = ["OOP", "instance", "attribute"]

                    # Add related concepts
                    for related in related_concepts:
                        if related not in existing_concepts:
                            self.model.concept_bank.add_character_concept(related)
                            results["concepts_added"] += 1
                            results["specific_concepts"].append(related)

                    # Form relationships between concepts
                    if len(related_concepts) > 0:
                        results["relationships_formed"] += len(related_concepts)
            except Exception as e:
                logger.error(f"Error adding programming concept {name}: {e}")

        # Add programming patterns
        programming_patterns = [
            "for item in collection:",
            "if condition:",
            "try: ... except Exception as e:",
            "def function_name(parameters):",
            "class ClassName:",
            "return result",
            "with open(filename, 'r') as file:",
            "lambda x: x * 2",
            "[x for x in collection if condition]"  # List comprehension
        ]

        # Add patterns to pattern memory
        if hasattr(self.model, "segmentation") and hasattr(self.model.segmentation, "pattern_memory"):
            for pattern in programming_patterns[:count]:
                try:
                    self.model.segmentation.pattern_memory.add_pattern(pattern, context="programming")
                    results["patterns_added"] += 1
                except Exception as e:
                    logger.error(f"Error adding programming pattern {pattern}: {e}")

        return results

    def _acquire_mathematical_knowledge(self, count=3):
        """Acquire mathematics-related knowledge"""
        results = {
            "concepts_added": 0,
            "patterns_added": 0,
            "relationships_formed": 0,
            "specific_concepts": []
        }

        # Mathematical concepts to potentially add
        math_concepts = [
            # Arithmetic
            ("addition", "Combining two numbers to form their sum"),
            ("subtraction", "Finding the difference between two numbers"),
            ("multiplication", "Repeated addition of a number"),
            ("division", "Splitting a number into equal parts"),

            # Algebra
            ("equation", "A statement asserting the equality of two expressions"),
            ("variable", "A symbol representing an unknown value"),
            ("polynomial", "An expression of more than one term"),
            ("function", "A relation between inputs and outputs"),

            # Calculus
            ("derivative", "Rate of change of a function"),
            ("integral", "Accumulation of quantities"),
            ("limit", "Value a function approaches as input approaches a certain value"),

            # Geometry
            ("angle", "Figure formed by two rays with a common endpoint"),
            ("triangle", "Three-sided polygon"),
            ("circle", "Set of points equidistant from a center point"),
            ("area", "Amount of space inside a boundary"),

            # Number theory
            ("prime", "Number greater than 1 divisible only by 1 and itself"),
            ("divisor", "Number that divides another without a remainder"),
            ("factor", "Number that divides another exactly"),

            # Sequences
            ("fibonacci", "Sequence where each number is the sum of the two preceding ones"),
            ("arithmetic", "Sequence with a constant difference between terms"),
            ("geometric", "Sequence with a constant ratio between terms")
        ]

        # Filter to concepts we don't already have
        existing_concepts = set()
        if hasattr(self.model, "concept_bank") and hasattr(self.model.concept_bank, "source_to_concept"):
            existing_concepts = set(self.model.concept_bank.source_to_concept.keys())

        new_concepts = [(name, desc) for name, desc in math_concepts if name not in existing_concepts]

        # Select random concepts up to count
        selected_concepts = random.sample(new_concepts, min(count, len(new_concepts)))

        # Add concepts to concept bank
        for name, description in selected_concepts:
            try:
                if hasattr(self.model, "concept_bank") and hasattr(self.model.concept_bank, "add_character_concept"):
                    self.model.concept_bank.add_character_concept(name)

                    # Record added concept
                    results["concepts_added"] += 1
                    results["specific_concepts"].append(name)

                    # Also add related concepts and form relationships
                    related_concepts = []
                    if name in ["addition", "subtraction", "multiplication", "division"]:
                        related_concepts = ["arithmetic", "operation", "calculation"]
                    elif name in ["equation", "variable", "polynomial"]:
                        related_concepts = ["algebra", "expression", "solve"]
                    elif name in ["fibonacci", "arithmetic", "geometric"]:
                        related_concepts = ["sequence", "series", "pattern"]

                    # Add related concepts
                    for related in related_concepts:
                        if related not in existing_concepts:
                            self.model.concept_bank.add_character_concept(related)
                            results["concepts_added"] += 1
                            results["specific_concepts"].append(related)

                    # Form relationships between concepts
                    if len(related_concepts) > 0:
                        results["relationships_formed"] += len(related_concepts)
            except Exception as e:
                logger.error(f"Error adding math concept {name}: {e}")

        # Add mathematical patterns
        math_patterns = [
            "a + b = c",
            "a - b = c",
            "a * b = c",
            "a / b = c",
            "a = b + c",
            "a = b - c",
            "a = b * c",
            "a = b / c",
            "a^2 + b^2 = c^2",  # Pythagorean theorem
            "f(x) = ax + b",  # Linear function
            "f(x) = ax^2 + bx + c",  # Quadratic function
            "an = a1 + (n-1)d",  # Arithmetic sequence
            "an = a1 * r^(n-1)"  # Geometric sequence
        ]

        # Add patterns to pattern memory
        if hasattr(self.model, "segmentation") and hasattr(self.model.segmentation, "pattern_memory"):
            for pattern in math_patterns[:count]:
                try:
                    self.model.segmentation.pattern_memory.add_pattern(pattern, context="mathematics")
                    results["patterns_added"] += 1
                except Exception as e:
                    logger.error(f"Error adding math pattern {pattern}: {e}")

        return results

    def _acquire_logical_knowledge(self, count=3):
        """Acquire logic-related knowledge"""
        results = {
            "concepts_added": 0,
            "patterns_added": 0,
            "relationships_formed": 0,
            "specific_concepts": []
        }

        # Logical concepts to potentially add
        logic_concepts = [
            # Propositional logic
            ("proposition", "A statement that is either true or false"),
            ("negation", "The logical operation that flips the truth value"),
            ("conjunction", "The logical operation AND"),
            ("disjunction", "The logical operation OR"),
            ("implication", "The logical operation IF-THEN"),
            ("biconditional", "The logical operation IF-AND-ONLY-IF"),

            # Logical rules
            ("modus_ponens", "From (P → Q) and P, infer Q"),
            ("modus_tollens", "From (P → Q) and ¬Q, infer ¬P"),
            ("hypothetical_syllogism", "From (P → Q) and (Q → R), infer (P → R)"),

            # Fallacies
            ("affirming_consequent", "Fallacy: From (P → Q) and Q, inferring P"),
            ("denying_antecedent", "Fallacy: From (P → Q) and ¬P, inferring ¬Q"),

            # Syllogistic logic
            ("syllogism", "A form of deductive reasoning with two premises"),
            ("universal_affirmative", "All A are B"),
            ("universal_negative", "No A are B"),
            ("particular_affirmative", "Some A are B"),
            ("particular_negative", "Some A are not B")
        ]

        # Filter to concepts we don't already have
        existing_concepts = set()
        if hasattr(self.model, "concept_bank") and hasattr(self.model.concept_bank, "source_to_concept"):
            existing_concepts = set(self.model.concept_bank.source_to_concept.keys())

        new_concepts = [(name, desc) for name, desc in logic_concepts if name not in existing_concepts]

        # Select random concepts up to count
        selected_concepts = random.sample(new_concepts, min(count, len(new_concepts)))

        # Add concepts to concept bank
        for name, description in selected_concepts:
            try:
                if hasattr(self.model, "concept_bank") and hasattr(self.model.concept_bank, "add_character_concept"):
                    self.model.concept_bank.add_character_concept(name)

                    # Record added concept
                    results["concepts_added"] += 1
                    results["specific_concepts"].append(name)

                    # Also add related concepts and form relationships
                    related_concepts = []
                    if name in ["proposition", "negation", "conjunction", "disjunction", "implication"]:
                        related_concepts = ["propositional_logic", "truth_value", "connective"]
                    elif name in ["modus_ponens", "modus_tollens", "hypothetical_syllogism"]:
                        related_concepts = ["inference_rule", "deduction", "valid_argument"]
                    elif name in ["affirming_consequent", "denying_antecedent"]:
                        related_concepts = ["fallacy", "invalid_argument", "logical_error"]

                    # Add related concepts
                    for related in related_concepts:
                        if related not in existing_concepts:
                            self.model.concept_bank.add_character_concept(related)
                            results["concepts_added"] += 1
                            results["specific_concepts"].append(related)

                    # Form relationships between concepts
                    if len(related_concepts) > 0:
                        results["relationships_formed"] += len(related_concepts)
            except Exception as e:
                logger.error(f"Error adding logic concept {name}: {e}")

        # Add logical patterns
        logic_patterns = [
            "If P then Q",
            "P → Q",
            "P and Q",
            "P ∧ Q",
            "P or Q",
            "P ∨ Q",
            "not P",
            "¬P",
            "P if and only if Q",
            "P ↔ Q",
            "All A are B",
            "Some A are B",
            "No A are B",
            "Some A are not B"
        ]

        # Add patterns to pattern memory
        if hasattr(self.model, "segmentation") and hasattr(self.model.segmentation, "pattern_memory"):
            for pattern in logic_patterns[:count]:
                try:
                    self.model.segmentation.pattern_memory.add_pattern(pattern, context="logic")
                    results["patterns_added"] += 1
                except Exception as e:
                    logger.error(f"Error adding logic pattern {pattern}: {e}")

        return results

    def _acquire_pattern_knowledge(self, count=3):
        """Acquire pattern recognition knowledge"""
        results = {
            "concepts_added": 0,
            "patterns_added": 0,
            "relationships_formed": 0,
            "specific_concepts": []
        }

        # Pattern concepts to potentially add
        pattern_concepts = [
            # Sequence patterns
            ("arithmetic_sequence", "Sequence with constant difference between terms"),
            ("geometric_sequence", "Sequence with constant ratio between terms"),
            ("fibonacci_sequence", "Sequence where each term is sum of two previous terms"),
            ("triangular_numbers", "Sequence of numbers representing triangular patterns"),
            ("square_numbers", "Sequence of numbers that are perfect squares"),
            ("cubic_numbers", "Sequence of numbers that are perfect cubes"),
            ("prime_numbers", "Sequence of numbers divisible only by 1 and themselves"),

            # Pattern types
            ("linear_pattern", "Pattern that follows a straight line relationship"),
            ("quadratic_pattern", "Pattern that follows a squared relationship"),
            ("exponential_pattern", "Pattern that follows an exponential relationship"),
            ("periodic_pattern", "Pattern that repeats at regular intervals"),
            ("recursive_pattern", "Pattern defined in terms of previous elements"),

            # Pattern operations
            ("interpolation", "Estimating values between known data points"),
            ("extrapolation", "Estimating values beyond known data points"),
            ("pattern_matching", "Finding occurrences of a pattern in data"),
            ("pattern_recognition", "Identifying patterns in data")
        ]

        # Filter to concepts we don't already have
        existing_concepts = set()
        if hasattr(self.model, "concept_bank") and hasattr(self.model.concept_bank, "source_to_concept"):
            existing_concepts = set(self.model.concept_bank.source_to_concept.keys())

        new_concepts = [(name, desc) for name, desc in pattern_concepts if name not in existing_concepts]

        # Select random concepts up to count
        selected_concepts = random.sample(new_concepts, min(count, len(new_concepts)))

        # Add concepts to concept bank
        for name, description in selected_concepts:
            try:
                if hasattr(self.model, "concept_bank") and hasattr(self.model.concept_bank, "add_character_concept"):
                    self.model.concept_bank.add_character_concept(name)

                    # Record added concept
                    results["concepts_added"] += 1
                    results["specific_concepts"].append(name)

                    # Also add related concepts and form relationships
                    related_concepts = []
                    if "sequence" in name:
                        related_concepts = ["sequence", "series", "pattern", "succession"]
                    elif "pattern" in name:
                        related_concepts = ["relationship", "trend", "correlation", "function"]
                    elif name in ["interpolation", "extrapolation"]:
                        related_concepts = ["estimation", "prediction", "trend", "forecasting"]

                    # Add related concepts
                    for related in related_concepts:
                        if related not in existing_concepts:
                            self.model.concept_bank.add_character_concept(related)
                            results["concepts_added"] += 1
                            results["specific_concepts"].append(related)

                    # Form relationships between concepts
                    if len(related_concepts) > 0:
                        results["relationships_formed"] += len(related_concepts)
            except Exception as e:
                logger.error(f"Error adding pattern concept {name}: {e}")

        # Add pattern examples
        pattern_examples = [
            "1, 2, 3, 4, 5, ...",  # Counting numbers
            "2, 4, 6, 8, 10, ...",  # Even numbers
            "1, 3, 5, 7, 9, ...",  # Odd numbers
            "1, 4, 9, 16, 25, ...",  # Square numbers
            "1, 8, 27, 64, 125, ...",  # Cube numbers
            "1, 1, 2, 3, 5, 8, 13, ...",  # Fibonacci
            "2, 3, 5, 7, 11, 13, ...",  # Prime numbers
            "1, 3, 6, 10, 15, ...",  # Triangular numbers
            "1, 2, 4, 8, 16, ...",  # Powers of 2
            "1, 3, 9, 27, 81, ..."   # Powers of 3
        ]

        # Add patterns to pattern memory
        if hasattr(self.model, "segmentation") and hasattr(self.model.segmentation, "pattern_memory"):
            for pattern in pattern_examples[:count]:
                try:
                    self.model.segmentation.pattern_memory.add_pattern(pattern, context="patterns")
                    results["patterns_added"] += 1
                except Exception as e:
                    logger.error(f"Error adding pattern example {pattern}: {e}")

        return results

    def _acquire_general_knowledge(self, count=3):
        """Acquire general knowledge across domains"""
        results = {
            "concepts_added": 0,
            "patterns_added": 0,
            "relationships_formed": 0,
            "specific_concepts": []
        }

        # Compile general knowledge across domains
        general_concepts = [
            # Meta-concepts
            ("abstraction", "Simplifying complexity by focusing on essential features"),
            ("generalization", "Forming general concepts from specific instances"),
            ("specialization", "Tailoring general concepts to specific cases"),
            ("analogy", "Comparison between two things based on similarity"),
            ("decomposition", "Breaking down a complex system into parts"),
            ("synthesis", "Combining parts to form a coherent whole"),

            # Learning concepts
            ("induction", "Reasoning from specific to general"),
            ("deduction", "Reasoning from general to specific"),
            ("abduction", "Reasoning to the best explanation"),
            ("hypothesis", "Proposed explanation requiring further testing"),
            ("evidence", "Information indicating whether a belief is true"),
            ("inference", "Conclusion reached on the basis of evidence and reasoning"),

            # System concepts
            ("recursion", "Self-referential process where solution depends on solutions to smaller instances"),
            ("iteration", "Repetition of a process to generate a sequence of outcomes"),
            ("emergence", "Complex patterns arising from simple rules"),
            ("feedback", "Process where output is routed back as input"),
            ("adaptation", "Process of changing to suit new conditions"),
            ("optimization", "Process of making something as effective as possible")
        ]

        # Filter to concepts we don't already have
        existing_concepts = set()
        if hasattr(self.model, "concept_bank") and hasattr(self.model.concept_bank, "source_to_concept"):
            existing_concepts = set(self.model.concept_bank.source_to_concept.keys())

        new_concepts = [(name, desc) for name, desc in general_concepts if name not in existing_concepts]

        # Select random concepts up to count
        selected_concepts = random.sample(new_concepts, min(count, len(new_concepts)))

        # Add concepts to concept bank
        for name, description in selected_concepts:
            try:
                if hasattr(self.model, "concept_bank") and hasattr(self.model.concept_bank, "add_character_concept"):
                    self.model.concept_bank.add_character_concept(name)

                    # Record added concept
                    results["concepts_added"] += 1
                    results["specific_concepts"].append(name)

                    # Also add related concepts and form relationships
                    # Group related concepts
                    concept_groups = {
                        "abstraction": ["simplification", "model", "concept"],
                        "generalization": ["induction", "pattern", "commonality"],
                        "analogy": ["comparison", "similarity", "mapping"],
                        "induction": ["observation", "pattern", "hypothesis"],
                        "deduction": ["logic", "proof", "conclusion"],
                        "recursion": ["self-reference", "base-case", "recurrence"],
                        "emergence": ["complexity", "system", "interaction"],
                        "adaptation": ["evolution", "adjustment", "learning"]
                    }

                    related_concepts = concept_groups.get(name, ["concept", "knowledge", "understanding"])

                    # Add related concepts
                    for related in related_concepts:
                        if related not in existing_concepts:
                            self.model.concept_bank.add_character_concept(related)
                            results["concepts_added"] += 1
                            results["specific_concepts"].append(related)

                    # Form relationships between concepts
                    if len(related_concepts) > 0:
                        results["relationships_formed"] += len(related_concepts)
            except Exception as e:
                logger.error(f"Error adding general concept {name}: {e}")

        return results

    def get_knowledge_stats(self):
        """Get statistics about acquired knowledge"""
        # Calculate growth rates
        total_acquisitions = len(self.acquisition_history)

        if total_acquisitions > 0:
            recent_acquisitions = min(5, total_acquisitions)
            recent_concepts = sum(a["concepts_added"] for a in self.acquisition_history[-recent_acquisitions:])
            recent_patterns = sum(a["patterns_added"] for a in self.acquisition_history[-recent_acquisitions:])

            growth_rate = recent_concepts / recent_acquisitions
        else:
            growth_rate = 0

        # Domain distribution
        domain_distribution = {
            domain: count / max(1, sum(self.knowledge_growth["concepts_by_domain"].values()))
            for domain, count in self.knowledge_growth["concepts_by_domain"].items()
        }

        return {
            "total_concepts_added": self.knowledge_growth["concepts_added"],
            "total_patterns_added": self.knowledge_growth["patterns_added"],
            "total_relationships_formed": self.knowledge_growth["relationships_formed"],
            "concept_growth_rate": growth_rate,
            "domain_distribution": domain_distribution,
            "focus_areas": self.focus_areas
        }


###########################################
# SAM INTEGRATION AND EXTENSIONS
###########################################

def extend_sam_with_self_evolution(sam_model):
    """Extend a SAM model with self-evolution capabilities"""
    if not hasattr(sam_model, "reasoning_engine"):
        sam_model.reasoning_engine = ReasoningEngine(sam_model)

    if not hasattr(sam_model, "verification_mechanism"):
        sam_model.verification_mechanism = VerificationMechanism(sam_model)

    if not hasattr(sam_model, "benchmark_manager"):
        sam_model.benchmark_manager = BenchmarkManager(sam_model)

    if not hasattr(sam_model, "knowledge_acquisition"):
        sam_model.knowledge_acquisition = KnowledgeAcquisition(sam_model)

    if not hasattr(sam_model, "self_evolution"):
        sam_model.self_evolution = SelfEvolutionEngine(sam_model)

    # Add or replace methods on the SAM model
    original_evolve = sam_model.evolve

    def enhanced_evolve(self):
        """Enhanced evolution method that includes self-directed learning"""
        # First run original architectural evolution
        result = original_evolve()

        # Then run self-evolution step
        if hasattr(self, "self_evolution"):
            self_evolution_result = self.self_evolution.evolve_step()

            # Combine results
            if isinstance(result, dict):
                result["self_evolution"] = self_evolution_result

            # Log evolution step
            logger.info(f"Self-evolution step completed with success rate: {self_evolution_result['success_rate']:.2f}")

        return result

    # Replace method
    sam_model.evolve = types.MethodType(enhanced_evolve, sam_model)

    # Add method to start autonomous evolution
    def start_autonomous_evolution(self, interval_minutes=15):
        """Start autonomous evolution in the background"""
        if hasattr(self, "self_evolution"):
            return self.self_evolution.start_evolution(interval_minutes=interval_minutes)
        return False

    sam_model.start_autonomous_evolution = types.MethodType(start_autonomous_evolution, sam_model)

    # Add method to stop autonomous evolution
    def stop_autonomous_evolution(self):
        """Stop autonomous evolution"""
        if hasattr(self, "self_evolution"):
            return self.self_evolution.stop_evolution()
        return False

    sam_model.stop_autonomous_evolution = types.MethodType(stop_autonomous_evolution, sam_model)

    # Add method to run benchmarks
    def run_benchmarks(self, categories=None):
        """Run benchmarks to evaluate capabilities"""
        if hasattr(self, "benchmark_manager"):
            return self.benchmark_manager.run_benchmarks(categories=categories)
        return {"error": "Benchmark manager not available"}

    sam_model.run_benchmarks = types.MethodType(run_benchmarks, sam_model)

    # Add method to acquire knowledge
    def acquire_knowledge(self, focus_areas=None):
        """Actively acquire new knowledge"""
        if hasattr(self, "knowledge_acquisition"):
            return self.knowledge_acquisition.acquire_knowledge(focus_areas=focus_areas)
        return {"error": "Knowledge acquisition not available"}

    sam_model.acquire_knowledge = types.MethodType(acquire_knowledge, sam_model)

    # Add method to get evolution status
    def get_evolution_status(self):
        """Get the current status of self-evolution"""
        if hasattr(self, "self_evolution"):
            status = self.self_evolution.get_evolution_status()

            # Add benchmark scores
            if hasattr(self, "benchmark_manager") and self.benchmark_manager.benchmark_history:
                latest_benchmark = self.benchmark_manager.benchmark_history[-1]
                status["benchmark_scores"] = latest_benchmark.get("category_scores", {})
                status["overall_benchmark_score"] = latest_benchmark.get("overall_score", 0)

            # Add knowledge stats
            if hasattr(self, "knowledge_acquisition"):
                status["knowledge_stats"] = self.knowledge_acquisition.get_knowledge_stats()

            return status

        return {"error": "Self-evolution not available"}

    sam_model.get_evolution_status = types.MethodType(get_evolution_status, sam_model)

    return sam_model

# Command-line interface for SAM self-evolution
def run_sam_evolution(load_path=None, mode="interact", evolution_interval=15, duration_minutes=None):
    """Run SAM with self-evolution capabilities"""

    # Create or load model
    if load_path and os.path.exists(load_path):
        model = SAM.load(load_path)  # Direct reference, no import needed
        print(f"Loaded SAM from {load_path}")
    else:
        model, _ = create_sam_model()  # Direct reference, no import needed
        print("Created new SAM model")

    # Extend with self-evolution
    model = extend_sam_with_self_evolution(model)
    print("Enhanced SAM with self-evolution capabilities")

    # Run in specified mode
    if mode == "autonomous":
        # Run fully autonomous evolution
        print(f"Starting autonomous evolution with {evolution_interval} minute interval")
        model.start_autonomous_evolution(interval_minutes=evolution_interval)

        if duration_minutes:
            print(f"Will run for {duration_minutes} minutes")
            # Run for specified duration
            start_time = time.time()
            end_time = start_time + (duration_minutes * 60)

            try:
                while time.time() < end_time:
                    # Print status every minute
                    status = model.get_evolution_status()
                    print(f"\nEvolution progress - Step: {status['step']}, Success rate: {status['performance_metrics']['success_rate']:.2f}")

                    # Sleep for a minute
                    time.sleep(60)

            except KeyboardInterrupt:
                print("\nEvolution interrupted by user")

            finally:
                # Stop evolution
                model.stop_autonomous_evolution()

                # Save model
                save_path = model.save()
                print(f"Model saved to {save_path}")

                # Run benchmarks to evaluate final performance
                print("\nRunning final benchmarks")
                benchmark_results = model.run_benchmarks()

                print("\nFinal benchmark results:")
                for category, results in benchmark_results.items():
                    if category != "timestamp" and category != "overall_score":
                        print(f"  {category}: {results['score']:.1f}")

                print(f"Overall score: {benchmark_results['overall_score']:.1f}")
        else:
            print("Press Ctrl+C to stop evolution")
            try:
                while True:
                    # Print status every minute
                    status = model.get_evolution_status()
                    print(f"\rEvolution step: {status['step']}, Success rate: {status['performance_metrics']['success_rate']:.2f}", end="")
                    time.sleep(60)
            except KeyboardInterrupt:
                print("\nEvolution stopped")
                model.stop_autonomous_evolution()

    elif mode == "benchmark":
        # Run benchmarks only
        print("Running benchmarks")
        benchmark_results = model.run_benchmarks()

        print("\nBenchmark results:")
        for category, results in benchmark_results.items():
            if category != "timestamp" and category != "overall_score":
                print(f"\n{category.upper()}: {results['score']:.1f}")
                for bench in results["benchmarks"]:
                    print(f"  {bench['name']} (difficulty {bench['difficulty']}): {bench['score']}/100")

        print(f"\nOverall score: {benchmark_results['overall_score']:.1f}")

    elif mode == "interact":
        # Interactive mode with autonomous evolution in background
        print("Starting background evolution")
        model.start_autonomous_evolution(interval_minutes=evolution_interval)

        print("\nInteractive mode. Enter commands or chat with SAM.")
        print("Special commands: 'status', 'benchmark', 'evolve', 'acquire', 'save', 'exit'")

        while True:
            try:
                user_input = input("\nYou: ")

                if user_input.lower() == "exit":
                    break

                elif user_input.lower() == "status":
                    status = model.get_evolution_status()
                    print("\nEvolution Status:")
                    print(f"  Active: {status['active']}")
                    print(f"  Step: {status['step']}")
                    print(f"  Success Rate: {status['performance_metrics']['success_rate']:.2f}")
                    print(f"  Task Complexity: {status['performance_metrics']['task_complexity']:.2f}")
                    print(f"  Reasoning Depth: {status['performance_metrics']['reasoning_depth']:.2f}")

                    if "benchmark_scores" in status:
                        print("\nBenchmark Scores:")
                        for category, score in status["benchmark_scores"].items():
                            print(f"  {category}: {score:.1f}")
                        print(f"  Overall: {status.get('overall_benchmark_score', 0):.1f}")

                    if "knowledge_stats" in status:
                        print("\nKnowledge Stats:")
                        k_stats = status["knowledge_stats"]
                        print(f"  Concepts Added: {k_stats['total_concepts_added']}")
                        print(f"  Patterns Added: {k_stats['total_patterns_added']}")
                        print(f"  Relationships Formed: {k_stats['total_relationships_formed']}")
                        print(f"  Growth Rate: {k_stats['concept_growth_rate']:.2f} concepts/cycle")
                        print(f"  Focus Areas: {', '.join(k_stats['focus_areas'])}")

                elif user_input.lower() == "benchmark":
                    print("\nRunning benchmarks...")
                    benchmark_results = model.run_benchmarks()

                    print("\nBenchmark results:")
                    for category, results in benchmark_results.items():
                        if category != "timestamp" and category != "overall_score":
                            print(f"\n{category.upper()}: {results['score']:.1f}")
                            for bench in results["benchmarks"]:
                                print(f"  {bench['name']} (difficulty {bench['difficulty']}): {bench['score']}/100")

                    print(f"\nOverall score: {benchmark_results['overall_score']:.1f}")

                elif user_input.lower() == "evolve":
                    print("\nRunning manual evolution step...")
                    results = model.self_evolution.evolve_step()

                    print(f"\nEvolution results:")
                    print(f"  Success Rate: {results['success_rate']:.2f}")
                    print(f"  Tasks Attempted: {len(results['tasks'])}")
                    print(f"  Successful Tasks: {results['success_count']}")

                elif user_input.lower() == "acquire":
                    print("\nAcquiring new knowledge...")
                    acquisition = model.acquire_knowledge()

                    print("\nKnowledge acquisition results:")
                    print(f"  Concepts Added: {acquisition['concepts_added']}")
                    print(f"  Patterns Added: {acquisition['patterns_added']}")
                    print(f"  Relationships Formed: {acquisition['relationships_formed']}")

                    if acquisition["concepts_added"] > 0 and "specific_concepts" in acquisition:
                        print(f"  Sample Concepts: {', '.join(acquisition['specific_concepts'][:5])}")

                    print(f"  Focus Areas: {', '.join(acquisition['focus_areas'])}")

                elif user_input.lower() == "save":
                    save_path = model.save()
                    print(f"\nModel saved to {save_path}")

                else:
                    # Normal interaction with the model
                    if hasattr(model, "generate"):
                        response = model.generate(input_text=user_input, max_length=500)
                        print(f"\nSAM: {response}")
                    else:
                        print("\nSAM: Sorry, I don't have text generation capabilities.")

            except KeyboardInterrupt:
                print("\nInterrupted. Type 'exit' to quit or continue interaction.")

            except Exception as e:
                print(f"\nError: {e}")

        # Stop evolution before exiting
        model.stop_autonomous_evolution()

        # Save model before exiting
        save_path = model.save()
        print(f"\nModel saved to {save_path}")
        print("Goodbye!")

    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: interact, autonomous, benchmark")

    parser = argparse.ArgumentParser(description="Run SAM with self-evolution capabilities")
    parser.add_argument("--mode", choices=["interact", "autonomous", "benchmark"], default="interact",
                       help="Mode to run SAM in")
    parser.add_argument("--load_path", type=str, default=None,
                       help="Path to load model from")
    parser.add_argument("--interval", type=int, default=15,
                       help="Interval between evolution steps (minutes)")
    parser.add_argument("--duration", type=int, default=None,
                       help="Duration to run autonomous evolution (minutes)")

    args = parser.parse_args()

    run_sam_evolution(
        load_path=args.load_path,
        mode=args.mode,
        evolution_interval=args.interval,
        duration_minutes=args.duration
    )


###########################################
# MAIN ENTRY POINT
###########################################

if __name__ == "__main__":
    # Parse command-line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Run SAM (Synergistic Autonomous Machine)")
    parser.add_argument("--load", help="Path to load model from", default=None)
    parser.add_argument("--save_dir", help="Directory to save model checkpoints", default="./data")
    parser.add_argument("--hive", help="Enable hive mind", action="store_true")
    parser.add_argument("--server", help="Run as hive mind server", action="store_true")
    parser.add_argument("--server_url", help="Hive mind server URL", default=None)
    parser.add_argument("--dim", help="Initial hidden dimension", type=int, default=None)
    parser.add_argument("--layers", help="Initial number of layers", type=int, default=None)
    parser.add_argument("--multimodal", help="Enable multimodal processing", action="store_true")
    parser.add_argument("--traditional", help="Use traditional mode instead of unified perception", action="store_true")

    args = parser.parse_args()

    # Create config overrides from args
    config_overrides = {}
    if args.save_dir:
        config_overrides["save_dir"] = args.save_dir
        config_overrides["experiences_path"] = os.path.join(args.save_dir, "experiences.json")
        config_overrides["concepts_path"] = os.path.join(args.save_dir, "concepts.json")
        config_overrides["growth_log_path"] = os.path.join(args.save_dir, "growth_log.json")

    if args.dim:
        config_overrides["initial_hidden_dim"] = args.dim

    if args.layers:
        config_overrides["initial_num_layers"] = args.layers

    # Create hive config if enabled
    hive_config = None
    if args.hive:
        hive_config = {
            "hive_enabled": True,
            "hive_server_mode": args.server,
            "hive_server_url": args.server_url or ("" if args.server else "http://localhost:8765")
        }

    # Run SAM
    run_sam(
        config=SAMConfig(**config_overrides) if config_overrides else None,
        load_path=args.load,
        hive_config=hive_config,
        multimodal=args.multimodal,
        unified_perception=not args.traditional
    )

