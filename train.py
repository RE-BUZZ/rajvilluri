"""
Complete Training Module with v3.1 Optimizations
Based on arch3.md with v3.1 enhancements integrated
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
from typing import Dict, List, Tuple, Optional
import gc

# Import v3.1 optimized models
from model import (
    CompleteLipSyncModel,
    AdaptiveWGANGPDiscriminator,  # v3.1: Adaptive gradient penalty
    SyncDiscriminator
)

# Import v3.1 utilities
from optimizations import MultiMetricMonitor  # v3.1: FID + MS-SSIM monitoring

# Import preprocessing utilities at module level to avoid importing in workers
from pipeline import CompletePreprocessor


class LazyLipSyncDataset(Dataset):
    """Memory-efficient dataset with lazy loading"""
    
    def __init__(self, data_dir: str, split: str = 'train'):
        self.data_dir = Path(data_dir)
        self.split = split
        self.cache_file = self.data_dir / f'.index_cache_{split}.pkl'
        self.sequence_index = self._load_or_build_index()
        print(f"Found {len(self.sequence_index)} sequences in {split} split")
    
    def _load_or_build_index(self) -> List[Tuple[Path, int]]:
        """Load cached index or build new one with smart validation"""
        # Get current files - look in subdirectories (train/ or val/)
        split_dir = self.data_dir / self.split
        if split_dir.exists():
            # New structure: data_dir/train/*.pkl and data_dir/val/*.pkl
            current_files = list(split_dir.glob('*.pkl'))
        else:
            # Old structure: data_dir/*.pkl with _val suffix
            current_files = list(self.data_dir.glob('*.pkl'))
            if self.split == 'train':
                current_files = [f for f in current_files if not f.stem.endswith('_val')]
            else:
                current_files = [f for f in current_files if f.stem.endswith('_val')]
        current_files_set = set(current_files)

        # Check if cache exists
        if self.cache_file.exists():
            try:
                print(f"Loading cached {self.split} index...", flush=True)
                with open(self.cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                # Smart validation: compare actual file lists
                cached_files_set = set(cached_data['files'])

                # Files that exist now but not in cache
                new_files = current_files_set - cached_files_set
                # Files that were in cache but don't exist now
                removed_files = cached_files_set - current_files_set

                # Calculate overlap percentage
                overlap_count = len(cached_files_set & current_files_set)
                total_count = len(current_files_set)
                overlap_pct = (overlap_count / total_count * 100) if total_count > 0 else 0

                # Smart decision tree
                if len(new_files) == 0 and len(removed_files) == 0:
                    # Perfect match - use cache
                    print(f"  ✓ Loaded {len(cached_data['index'])} sequences from cache (perfect match)", flush=True)
                    return cached_data['index']

                elif overlap_pct >= 85 and len(new_files) <= 100:  # More lenient for growing dataset
                    # Small change - incrementally update
                    print(f"  ⚡ Cache mostly valid ({overlap_pct:.1f}% match), incrementally updating...", flush=True)
                    print(f"     New files: {len(new_files)}, Removed: {len(removed_files)}", flush=True)
                    return self._incremental_update(cached_data, new_files, removed_files, current_files)

                elif overlap_pct >= 5:  # LOWERED: Accept cache even with many new files
                    # Decent overlap - use cache anyway (good enough)
                    print(f"  ✓ Cache good enough ({overlap_pct:.1f}% match), using cached index", flush=True)
                    print(f"     Note: {len(new_files)} new files, {len(removed_files)} removed (acceptable drift)", flush=True)
                    return cached_data['index']

                else:
                    # Significant change - rebuild
                    print(f"  ⚠ Cache stale ({overlap_pct:.1f}% match), rebuilding from scratch...", flush=True)
                    print(f"     New files: {len(new_files)}, Removed: {len(removed_files)}", flush=True)
                    return self._build_index(current_files)

            except Exception as e:
                print(f"  ⚠ Cache load failed ({e}), rebuilding...", flush=True)
                return self._build_index(current_files)

        # No cache - build from scratch
        return self._build_index(current_files)

    def _incremental_update(self, cached_data, new_files, removed_files, current_files):
        """Incrementally update index for small changes"""
        index = cached_data['index']

        # Remove entries for deleted files
        if removed_files:
            removed_files_set = set(removed_files)
            index = [(path, idx) for path, idx in index if path not in removed_files_set]
            print(f"     Removed {len(removed_files)} deleted files from index", flush=True)

        # Add entries for new files
        if new_files:
            new_count = 0
            for file_path in new_files:
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                        num_sequences = len(data.get('sequences', []))
                        for seq_idx in range(num_sequences):
                            index.append((file_path, seq_idx))
                        new_count += num_sequences
                except Exception as e:
                    print(f"     ⚠ Failed to index {file_path.name}: {e}", flush=True)
            print(f"     Added {len(new_files)} new files ({new_count} sequences) to index", flush=True)

        # Save updated cache
        try:
            cache_data = {'files': list(current_files), 'index': index}
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"  ✓ Saved updated index cache ({len(index)} total sequences)", flush=True)
        except Exception as e:
            print(f"  ⚠ Failed to save updated cache: {e}", flush=True)

        return index

    def _build_index(self, files=None) -> List[Tuple[Path, int]]:
        """Build index of (file_path, sequence_idx) tuples with incremental saving"""
        if files is None:
            split_dir = self.data_dir / self.split
            if split_dir.exists():
                # New structure: data_dir/train/*.pkl and data_dir/val/*.pkl
                files = list(split_dir.glob('*.pkl'))
            else:
                # Old structure: data_dir/*.pkl with _val suffix
                files = list(self.data_dir.glob('*.pkl'))
                if self.split == 'train':
                    files = [f for f in files if not f.stem.endswith('_val')]
                else:
                    files = [f for f in files if f.stem.endswith('_val')]

        print(f"Building {self.split} index from {len(files)} files...", flush=True)
        print(f"  💾 Saving cache every 10 files to prevent data loss", flush=True)
        index = []
        for i, file_path in enumerate(files):
            # Progress update every 50 files
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  Progress: {i+1}/{len(files)} files ({100*(i+1)/len(files):.1f}%)", flush=True)

            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                num_sequences = len(data.get('sequences', []))
                for seq_idx in range(num_sequences):
                    index.append((file_path, seq_idx))

            # Save cache every 10 files
            if (i + 1) % 10 == 0:
                try:
                    cache_data = {'files': files, 'index': index}
                    with open(self.cache_file, 'wb') as f:
                        pickle.dump(cache_data, f)
                    if (i + 1) % 50 == 0:  # Only print every 50 to avoid spam
                        print(f"    💾 Cache saved ({len(index)} sequences)", flush=True)
                except Exception as e:
                    print(f"    ⚠ Failed to save cache: {e}", flush=True)

        print(f"  Completed: {len(files)} files indexed, {len(index)} total sequences", flush=True)

        # Final save
        try:
            cache_data = {'files': files, 'index': index}
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"  ✓ Saved final index cache to {self.cache_file}", flush=True)
        except Exception as e:
            print(f"  Warning: Failed to save final cache: {e}", flush=True)

        return index
    
    def __len__(self):
        return len(self.sequence_index)
    
    def __getitem__(self, idx):
        file_path, seq_idx = self.sequence_index[idx]

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        sequence = data['sequences'][seq_idx]

        # Check if data is JPEG compressed
        is_compressed = sequence.get('is_jpeg_compressed', False)

        if is_compressed:
            # Decompress JPEG bytes back to numpy arrays
            face_crops_list = [CompletePreprocessor.decompress_image_jpeg(jpeg_bytes)
                              for jpeg_bytes in sequence['face_crops']]
            mouth_crops_list = [CompletePreprocessor.decompress_image_jpeg(jpeg_bytes)
                               for jpeg_bytes in sequence['mouth_crops']]

            face_crops_raw = np.array(face_crops_list)
            mouth_crops_raw = np.array(mouth_crops_list)
        else:
            # Use uncompressed data directly (backward compatibility)
            face_crops_raw = sequence['face_crops']
            mouth_crops_raw = sequence['mouth_crops']

        # Convert to contiguous numpy arrays first
        face_crops_np = np.ascontiguousarray(face_crops_raw.transpose(0, 3, 1, 2), dtype=np.float32)
        mouth_crops_np = np.ascontiguousarray(mouth_crops_raw.transpose(0, 3, 1, 2), dtype=np.float32)

        # Handle mel_windows with shape validation
        mel_raw = sequence['mel_windows']
        # Expected: [T, 80, 7] for 3D mel spectrograms with context windows
        if len(mel_raw.shape) != 3:
            # Log warning but try to handle gracefully
            print(f"WARNING: Unexpected mel_windows shape {mel_raw.shape} in {file_path.name}, seq {seq_idx}")
            # If 2D [T, features], reshape to [T, 80, 7] assuming features=560
            if len(mel_raw.shape) == 2 and mel_raw.shape[1] == 560:
                mel_raw = mel_raw.reshape(mel_raw.shape[0], 80, 7)
            # If 4D [1, T, 80, 7], squeeze first dim
            elif len(mel_raw.shape) == 4 and mel_raw.shape[0] == 1:
                mel_raw = mel_raw.squeeze(0)

        mel_windows_np = np.ascontiguousarray(mel_raw, dtype=np.float32)
        visemes_np = np.ascontiguousarray(sequence['visemes'], dtype=np.int64)
        reference_frame_np = np.ascontiguousarray(face_crops_raw[0].transpose(2, 0, 1), dtype=np.float32)
        audio_raw_np = np.ascontiguousarray(sequence.get('audio_raw', np.zeros(16000)), dtype=np.float32)

        # Normalize and convert to torch (do math in torch to avoid numpy view issues)
        return {
            'face_crops': torch.from_numpy(face_crops_np).float() / 127.5 - 1.0,
            'mouth_crops': torch.from_numpy(mouth_crops_np).float() / 127.5 - 1.0,
            'mel_windows': torch.from_numpy(mel_windows_np),
            'visemes': torch.from_numpy(visemes_np).long(),
            'reference_frame': torch.from_numpy(reference_frame_np).float() / 127.5 - 1.0,
            'audio_raw': torch.from_numpy(audio_raw_np)
        }


def collate_pad_sequences(batch):
    """Custom collate function to pad variable-length sequences"""
    # Find max length for each tensor type (they have different temporal resolutions)
    max_video_len = max([item['face_crops'].shape[0] for item in batch])
    max_audio_len = max([item['mel_windows'].shape[0] for item in batch])

    max_lens = {
        'face_crops': max_video_len,
        'mouth_crops': max_video_len,
        'mel_windows': max_audio_len,
        'visemes': max_video_len
    }

    padded_batch = {}
    for key in batch[0].keys():
        tensors = [item[key] for item in batch]

        if key in ['face_crops', 'mouth_crops']:
            # Pad video sequences: [T, C, H, W]
            max_len = max_lens[key]
            padded = []
            for tensor in tensors:
                T = tensor.shape[0]
                if T < max_len:
                    # Pad with zeros
                    pad_amount = max_len - T
                    padding = torch.zeros((pad_amount, *tensor.shape[1:]), dtype=tensor.dtype)
                    padded.append(torch.cat([tensor, padding], dim=0))
                else:
                    padded.append(tensor)
            padded_batch[key] = torch.stack(padded, dim=0)

        elif key == 'mel_windows':
            # Pad mel spectrograms: [T, mel_bins, ...]
            max_len = max_lens[key]
            padded = []
            for tensor in tensors:
                T = tensor.shape[0]
                if T < max_len:
                    pad_amount = max_len - T
                    padding = torch.zeros((pad_amount, *tensor.shape[1:]), dtype=tensor.dtype)
                    padded.append(torch.cat([tensor, padding], dim=0))
                else:
                    padded.append(tensor)
            padded_batch[key] = torch.stack(padded, dim=0)

        elif key == 'visemes':
            # Pad viseme labels: [T]
            max_len = max_lens[key]
            padded = []
            for tensor in tensors:
                T = tensor.shape[0]
                if T < max_len:
                    pad_amount = max_len - T
                    padding = torch.zeros((pad_amount,), dtype=tensor.dtype)
                    padded.append(torch.cat([tensor, padding], dim=0))
                else:
                    padded.append(tensor)
            padded_batch[key] = torch.stack(padded, dim=0)

        elif key == 'reference_frame':
            # Reference frame: [C, H, W] - no padding needed
            padded_batch[key] = torch.stack(tensors, dim=0)

        elif key == 'audio_raw':
            # Audio: [audio_samples] - no padding needed (fixed size)
            padded_batch[key] = torch.stack(tensors, dim=0)

    return padded_batch


class FixedSyncLoss(nn.Module):
    """Fixed sync loss computation"""

    def __init__(self, syncnet):
        super().__init__()
        self.syncnet = syncnet
    
    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: [B, T, C, H, W]
            audio: [B, T, mel_bins]
        """
        self.syncnet.eval()
        with torch.no_grad():
            sync_score = self.syncnet(video, audio)
        
        # Higher sync score = better sync, so minimize negative score
        loss = -torch.mean(sync_score)
        return loss


class CompleteLoss(nn.Module):
    """
    Enhanced loss function with WGAN-GP discriminator loss and v3.1 adaptive penalty
    """
    
    def __init__(self, syncnet, vgg_for_perceptual=True, use_wgan_gp=True):
        super().__init__()
        
        self.syncnet = syncnet
        self.use_wgan_gp = use_wgan_gp
        
        # VGG for perceptual loss
        if vgg_for_perceptual:
            try:
                from torchvision.models import vgg16, VGG16_Weights
                vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
                self.vgg = nn.Sequential(*list(vgg.features)[:23]).eval()
                for param in self.vgg.parameters():
                    param.requires_grad = False
            except:
                print("Warning: VGG16 not available, perceptual loss disabled")
                self.vgg = None
        else:
            self.vgg = None
        
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        
        # Loss weights
        self.lambda_l1 = 100.0
        self.lambda_perceptual = 10.0
        self.lambda_sync = 5.0
        self.lambda_viseme = 2.0
        self.lambda_flow = 1.0
        self.lambda_adv = 1.0
        
        # Perceptual warmup (gradually increase weight)
        self.perceptual_warmup = 0.0
    
    def perceptual_loss(self, pred, target):
        """VGG perceptual loss"""
        if self.vgg is None:
            return torch.tensor(0.0, device=pred.device)
        
        pred_feat = self.vgg(pred)
        target_feat = self.vgg(target)
        return self.l2(pred_feat, target_feat) * self.perceptual_warmup
    
    def sync_loss(self, video, audio):
        """Audio-visual sync loss"""
        self.syncnet.eval()
        with torch.no_grad():
            sync_score = self.syncnet(video, audio)
        return -torch.mean(sync_score)
    
    def optical_flow_loss(self, flows):
        """Temporal smoothness loss"""
        if flows is None or len(flows) == 0:
            return torch.tensor(0.0)
        
        flow_tensor = torch.stack(flows, dim=1)
        
        # Smoothness: penalize large flow magnitudes
        smoothness = torch.mean(torch.abs(flow_tensor))
        
        # Consistency: penalize temporal variations
        if flow_tensor.size(1) > 1:
            flow_diff = flow_tensor[:, 1:] - flow_tensor[:, :-1]
            consistency = torch.mean(torch.abs(flow_diff))
        else:
            consistency = torch.tensor(0.0, device=flow_tensor.device)
        
        return smoothness * 0.5 + consistency * 0.5
    
    def wgan_gp_discriminator_loss(self, disc_real, disc_fake, gradient_penalty):
        """
        WGAN-GP discriminator loss with v3.1 adaptive gradient penalty
        
        Args:
            disc_real: Discriminator output for real samples
            disc_fake: Discriminator output for fake samples
            gradient_penalty: Computed gradient penalty (already includes adaptive lambda)
        """
        real_score = disc_real['real_fake'].mean()
        fake_score = disc_fake['real_fake'].mean()
        
        # Wasserstein loss: maximize distance between real and fake
        wasserstein_loss = -(real_score - fake_score)
        
        # Total discriminator loss
        disc_loss = wasserstein_loss + gradient_penalty
        
        return disc_loss, {'wasserstein': wasserstein_loss.item(), 'gp': gradient_penalty.item()}
    
    def wgan_gp_generator_loss(self, disc_fake):
        """
        WGAN-GP generator loss (fool discriminator)
        """
        fake_score = disc_fake['real_fake'].mean()
        return -fake_score
    
    def discriminator_loss(self, disc_real, disc_fake):
        """BCE discriminator loss (fallback if not using WGAN-GP)"""
        real_loss = F.binary_cross_entropy_with_logits(
            disc_real['real_fake'],
            torch.ones_like(disc_real['real_fake'])
        )
        
        fake_loss = F.binary_cross_entropy_with_logits(
            disc_fake['real_fake'],
            torch.zeros_like(disc_fake['real_fake'])
        )
        
        d_loss = (real_loss + fake_loss) / 2
        
        # Viseme classification auxiliary loss
        if 'viseme_logits' in disc_real and 'target_visemes' in disc_real:
            viseme_loss = self.ce(
                disc_real['viseme_logits'],
                disc_real['target_visemes']
            )
            d_loss += viseme_loss * 0.1
        
        return d_loss
    
    def generator_adversarial_loss(self, disc_fake):
        """BCE generator adversarial loss (fallback if not using WGAN-GP)"""
        fake_loss = F.binary_cross_entropy_with_logits(
            disc_fake['real_fake'],
            torch.ones_like(disc_fake['real_fake'])
        )
        
        # Viseme classification loss for generator
        if 'viseme_logits' in disc_fake and 'target_visemes' in disc_fake:
            viseme_loss = self.ce(
                disc_fake['viseme_logits'],
                disc_fake['target_visemes']
            )
            fake_loss += viseme_loss * 0.1
        
        return fake_loss
    
    def forward(self, outputs, targets, discriminator_outputs=None, stage='full'):
        """
        Compute all losses
        
        Args:
            outputs: Generator outputs dict
            targets: Ground truth targets dict
            discriminator_outputs: Dict with 'real' and 'fake' discriminator outputs
            stage: Training stage ('coarse', 'sync', 'full')
        """
        losses = {}
        
        generated = outputs['generated_mouths']
        target = targets['mouth_crops']
        
        # Reshape for loss computation
        B, T = generated.shape[:2]
        generated_flat = generated.reshape(B * T, *generated.shape[2:])
        target_flat = target.reshape(B * T, *target.shape[2:])
        
        # L1 reconstruction loss
        losses['l1'] = self.l1(generated_flat, target_flat) * self.lambda_l1
        
        # Perceptual loss
        if self.vgg is not None and self.perceptual_warmup > 0:
            losses['perceptual'] = self.perceptual_loss(generated_flat, target_flat) * self.lambda_perceptual
        
        # Sync loss (stage 2 onwards)
        if stage in ['sync', 'full']:
            audio = targets.get('mel_windows', targets.get('audio_raw'))
            losses['sync'] = self.sync_loss(generated, audio) * self.lambda_sync
        
        # Optical flow smoothness loss
        if 'optical_flows' in outputs and outputs['optical_flows'] is not None:
            losses['flow'] = self.optical_flow_loss(outputs['optical_flows']) * self.lambda_flow
        
        # Adversarial loss (stage 3 only)
        if stage == 'full' and discriminator_outputs is not None:
            if self.use_wgan_gp:
                losses['adv'] = self.wgan_gp_generator_loss(discriminator_outputs['fake']) * self.lambda_adv
            else:
                losses['adv'] = self.generator_adversarial_loss(discriminator_outputs['fake']) * self.lambda_adv
        
        # Total generator loss
        losses['total'] = sum(losses.values())
        
        return losses


class CompleteTrainer:
    """
    Enhanced progressive training with v3.1 optimizations:
    - Sparse attention for 30-50% compute reduction
    - Adaptive λ_gp for 10% faster convergence
    - Multi-metric monitoring (FID + MS-SSIM)
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_gpus = torch.cuda.device_count()

        # Create directories
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        fid_dir = Path(config['checkpoint_dir']) / 'fid_real'
        fid_dir.mkdir(exist_ok=True)

        # v3.1: Initialize models with sparse attention (default enabled)
        print("Initializing models with v3.1 optimizations...")
        print(f"  Found {self.num_gpus} GPUs")
        self.generator = CompleteLipSyncModel(
            num_visemes=21,
            use_checkpoint=True,
            use_sparse_attention=config.get('use_sparse_attention', True)  # v3.1
        ).to(self.device)
        
        # v3.1: Use AdaptiveWGANGPDiscriminator instead of standard discriminator
        if config.get('use_wgan_gp', True):
            self.discriminator = AdaptiveWGANGPDiscriminator(num_visemes=21).to(self.device)
            print("  ✓ Using AdaptiveWGANGPDiscriminator with adaptive gradient penalty")
        else:
            # Fallback to standard discriminator
            from model import MultiScaleDiscriminator
            self.discriminator = MultiScaleDiscriminator(num_visemes=21).to(self.device)
            print("  ✓ Using standard discriminator")
        
        self.syncnet = SyncDiscriminator().to(self.device)

        # Multi-GPU support with DataParallel
        if self.num_gpus > 1:
            print(f"  ✓ Using DataParallel across {self.num_gpus} GPUs")
            self.generator = nn.DataParallel(self.generator)
            self.discriminator = nn.DataParallel(self.discriminator)
            self.syncnet = nn.DataParallel(self.syncnet)
            # Adjust batch size for multi-GPU
            effective_batch_size = config.get('batch_size', 4) * self.num_gpus
            print(f"  ✓ Effective batch size: {effective_batch_size} ({config.get('batch_size', 4)} per GPU)")

        # Compile models for additional speedup (PyTorch 2.0+) - disabled for DataParallel
        if config.get('use_compile', True) and self.num_gpus == 1:
            try:
                self.generator = torch.compile(self.generator)
                self.discriminator = torch.compile(self.discriminator)
                print("  ✓ Models compiled with torch.compile")
            except Exception as e:
                print(f"  ⚠ torch.compile not available: {e}")
        elif config.get('use_compile', True) and self.num_gpus > 1:
            print("  ⚠ torch.compile disabled (incompatible with DataParallel)")
        
        # Load pretrained SyncNet if available
        syncnet_path = config.get('syncnet_checkpoint')
        if syncnet_path and os.path.exists(syncnet_path):
            self.syncnet.load_state_dict(torch.load(syncnet_path))
            print(f"  ✓ Loaded pretrained SyncNet from {syncnet_path}")
        
        # Optimizers
        self.opt_g = optim.Adam(
            self.generator.parameters(),
            lr=config.get('lr_g', 1e-4),
            betas=(0.5, 0.999)
        )
        self.opt_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config.get('lr_d', 1e-4),
            betas=(0.5, 0.999)
        )
        
        # Learning rate schedulers
        self.scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_g, mode='min', factor=0.5, patience=5
        )
        self.scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_d, mode='min', factor=0.5, patience=5
        )
        
        # Loss function
        self.criterion = CompleteLoss(
            self.syncnet,
            vgg_for_perceptual=True,
            use_wgan_gp=config.get('use_wgan_gp', True)
        ).to(self.device)
        
        # v3.1: Multi-metric monitoring (FID + MS-SSIM)
        if not config.get('no_monitor_metrics', False):
            self.metric_monitor = MultiMetricMonitor(str(fid_dir), device=self.device)
            print("  ✓ Multi-metric monitoring enabled (FID + MS-SSIM)")
        else:
            self.metric_monitor = None
            print("  ⚠ Multi-metric monitoring disabled")
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Data loaders
        train_dataset = LazyLipSyncDataset(config['data_dir'], split='train')
        val_dataset = LazyLipSyncDataset(config['data_dir'], split='val')
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('batch_size', 4),
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_pad_sequences
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('batch_size', 4),
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=collate_pad_sequences
        )
        
        # Training state
        self.current_epoch = 0
        self.current_stage = 'coarse'
        self.best_val_loss = float('inf')
        self.n_critic = config.get('n_critic', 5)  # Train discriminator n times per generator
        self.lambda_gp = config.get('lambda_gp', 10)  # Base gradient penalty coefficient

        # Load latest checkpoint if available
        self._load_latest_checkpoint()

        # Wandb logging
        if config.get('use_wandb', False) and WANDB_AVAILABLE:
            wandb.init(
                project="lipsync-v31",
                name=config.get('run_name', 'training'),
                config=config
            )
        elif config.get('use_wandb', False) and not WANDB_AVAILABLE:
            print("  ⚠ Wandb not available, logging disabled")
        
        print("\n" + "="*60)
        print("Training Configuration:")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {config.get('batch_size', 4)}")
        print(f"  Learning rate (G): {config.get('lr_g', 1e-4)}")
        print(f"  Learning rate (D): {config.get('lr_d', 1e-4)}")
        print(f"  Use WGAN-GP: {config.get('use_wgan_gp', True)}")
        print(f"  Use sparse attention: {config.get('use_sparse_attention', True)}")
        print(f"  n_critic: {self.n_critic}")
        print(f"  Lambda GP: {self.lambda_gp}")
        print("="*60 + "\n")
    
    def _setup_metric_real_images(self):
        """Save real images for FID/metric computation"""
        if self.metric_monitor is None:
            return

        print("Setting up real images for metric computation...")
        real_images = []
        for batch in self.train_loader:
            mouth_crops = batch['mouth_crops']
            B, T = mouth_crops.shape[:2]
            mouth_flat = mouth_crops.reshape(B * T, *mouth_crops.shape[2:])
            real_images.append(mouth_flat[:10])  # Save 10 samples per batch
            if len(real_images) >= 10:
                break

        if real_images:
            real_images = torch.cat(real_images, dim=0)
            self.metric_monitor.save_real_images(real_images)
        print(f"  ✓ Saved {len(real_images)} real images for metrics")

    def _load_latest_checkpoint(self):
        """Load the latest checkpoint from checkpoint directory"""
        checkpoint_dir = Path(self.config['checkpoint_dir'])
        if not checkpoint_dir.exists():
            print("  No checkpoint directory found, starting from scratch")
            return

        # Find all checkpoints
        checkpoints = list(checkpoint_dir.glob('checkpoint_epoch*_*.pt'))
        if not checkpoints:
            print("  No checkpoints found, starting from scratch")
            return

        # Sort by epoch number (extract from filename)
        def get_epoch(path):
            try:
                # Extract epoch number from checkpoint_epoch{N}_{stage}.pt
                parts = path.stem.split('_')
                epoch_str = parts[1].replace('epoch', '')
                return int(epoch_str)
            except:
                return -1

        checkpoints = sorted(checkpoints, key=get_epoch, reverse=True)
        latest_checkpoint = checkpoints[0]

        print(f"\n  ✓ Loading checkpoint from: {latest_checkpoint.name}")

        try:
            checkpoint = torch.load(latest_checkpoint, map_location=self.device, weights_only=False)

            # Load model states
            self.generator.load_state_dict(checkpoint['generator_state'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state'])
            self.opt_g.load_state_dict(checkpoint['opt_g_state'])
            self.opt_d.load_state_dict(checkpoint['opt_d_state'])

            # Load training state
            self.current_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
            self.current_stage = checkpoint['stage']
            self.best_val_loss = checkpoint['best_val_loss']

            print(f"  ✓ Resumed from Epoch {checkpoint['epoch']} ({self.current_stage} stage)")
            print(f"  ✓ Best validation loss: {self.best_val_loss:.4f}")
            print(f"  ✓ Will continue from Epoch {self.current_epoch}\n")

        except Exception as e:
            print(f"  ⚠ Failed to load checkpoint: {e}")
            print(f"  Starting from scratch")
            self.current_epoch = 0
            self.current_stage = 'coarse'
            self.best_val_loss = float('inf')

    def monitor_gradients(self):
        """Monitor gradient magnitudes for debugging"""
        total_norm = 0.0
        for p in self.generator.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def update_perceptual_warmup(self, epoch):
        """Gradually increase perceptual loss weight"""
        warmup_epochs = 10
        self.criterion.perceptual_warmup = min(1.0, epoch / warmup_epochs)
    
    def train_step(self, batch, stage='full'):
        """Single training step with v3.1 adaptive WGAN-GP"""
        # SAFETY: Validate batch dimensions to skip corrupted data
        try:
            if 'mel_windows' in batch:
                mel_shape = batch['mel_windows'].shape
                if len(mel_shape) == 5:
                    print(f"WARNING: mel_windows has 5D shape {mel_shape}, squeezing")
                    batch['mel_windows'] = batch['mel_windows'].squeeze(2)
                elif len(mel_shape) not in [3, 4]:
                    print(f"SKIP BATCH: Invalid mel_windows shape {mel_shape}")
                    return {'g_total': 0.0, 'g_l1': 0.0, 'g_perceptual': 0.0}
        except Exception as e:
            print(f"SKIP CORRUPTED BATCH: {e}")
            return {'g_total': 0.0, 'g_l1': 0.0, 'g_perceptual': 0.0}
        
        
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        
        all_losses = {}
        
        # ==================== Train Discriminator ====================
        if stage == 'full':
            # v3.1: Update discriminator epoch for adaptive λ_gp
            if hasattr(self.discriminator, 'module'):
                self.discriminator.module.update_epoch(self.current_epoch)
            else:
                self.discriminator.update_epoch(self.current_epoch)
            
            for _ in range(self.n_critic):
                self.opt_d.zero_grad()
                
                with autocast() if self.use_amp else torch.cuda.amp.autocast(enabled=False):
                    # Generate fake samples
                    with torch.no_grad():
                        outputs = self.generator(batch)
                        fake_mouths = outputs['generated_mouths'].detach()
                    
                    real_mouths = batch['mouth_crops']
                    
                    # Discriminator predictions
                    disc_real = self.discriminator(real_mouths)
                    disc_fake = self.discriminator(fake_mouths)
                    
                    if self.config.get('use_wgan_gp', True):
                        # v3.1: Compute adaptive gradient penalty (returns tuple)
                        if hasattr(self.discriminator, 'module'):
                            gp, adaptive_lambda = self.discriminator.module.compute_gradient_penalty(
                                real_mouths, fake_mouths,
                                base_lambda_gp=self.lambda_gp,
                                max_epochs=50
                            )
                        else:
                            gp, adaptive_lambda = self.discriminator.compute_gradient_penalty(
                                real_mouths, fake_mouths,
                                base_lambda_gp=self.lambda_gp,
                                max_epochs=50
                            )
                        
                        # WGAN-GP discriminator loss
                        d_loss, d_loss_dict = self.criterion.wgan_gp_discriminator_loss(
                            disc_real, disc_fake, gp
                        )
                        
                        all_losses.update({
                            'd_loss': d_loss.item(),
                            'wasserstein': d_loss_dict['wasserstein'],
                            'gradient_penalty': d_loss_dict['gp'],
                            'adaptive_lambda_gp': adaptive_lambda  # v3.1: Track adaptive lambda
                        })
                    else:
                        # Standard BCE discriminator loss
                        d_loss = self.criterion.discriminator_loss(disc_real, disc_fake)
                        all_losses['d_loss'] = d_loss.item()
                
                # Backward pass for discriminator
                if self.use_amp:
                    self.scaler.scale(d_loss).backward()
                    self.scaler.step(self.opt_d)
                    self.scaler.update()
                else:
                    d_loss.backward()
                    self.opt_d.step()
        
        # ==================== Train Generator ====================
        self.opt_g.zero_grad()
        
        with autocast() if self.use_amp else torch.cuda.amp.autocast(enabled=False):
            # Forward pass
            outputs = self.generator(batch)
            
            # Prepare discriminator outputs if in full training
            discriminator_outputs = None
            if stage == 'full':
                fake_mouths = outputs['generated_mouths']
                disc_fake = self.discriminator(fake_mouths)
                discriminator_outputs = {'fake': disc_fake}
            
            # Compute generator losses
            targets = {
                'mouth_crops': batch['mouth_crops'],
                'mel_windows': batch['mel_windows'],
                'visemes': batch['visemes']
            }
            
            g_losses = self.criterion(outputs, targets, discriminator_outputs, stage=stage)
        
        # Backward pass for generator
        g_loss = g_losses['total']
        if self.use_amp:
            self.scaler.scale(g_loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.opt_g)
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            
            self.scaler.step(self.opt_g)
            self.scaler.update()
        else:
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            self.opt_g.step()
        
        # Collect generator losses
        for key, value in g_losses.items():
            if isinstance(value, torch.Tensor):
                all_losses[f'g_{key}'] = value.item()
            else:
                all_losses[f'g_{key}'] = value
        
        # Monitor gradients
        grad_norm = self.monitor_gradients()
        all_losses['grad_norm'] = grad_norm
        
        return all_losses
    
    def validate(self, stage='full', compute_metrics=False):
        """Validation step with v3.1 multi-metric computation"""
        
        self.generator.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validating")):
                # Move to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Forward pass
                outputs = self.generator(batch)
                
                # Compute losses (no discriminator in validation)
                targets = {
                    'mouth_crops': batch['mouth_crops'],
                    'mel_windows': batch['mel_windows'],
                    'visemes': batch['visemes']
                }
                
                losses = self.criterion(outputs, targets, discriminator_outputs=None, stage=stage)
                val_losses.append({k: v.item() if isinstance(v, torch.Tensor) else v 
                                  for k, v in losses.items()})
                
                # v3.1: Compute metrics on first batch
                if compute_metrics and batch_idx == 0 and self.metric_monitor is not None:
                    generated = outputs['generated_mouths']
                    target = batch['mouth_crops']
                    
                    # Reshape to [N, C, H, W]
                    B, T = generated.shape[:2]
                    generated_flat = generated.reshape(B * T, *generated.shape[2:])
                    target_flat = target.reshape(B * T, *target.shape[2:])
                    
                    # Normalize to [0, 1]
                    generated_norm = (generated_flat + 1.0) / 2.0
                    target_norm = (target_flat + 1.0) / 2.0
                    
                    # Compute all metrics
                    metrics = self.metric_monitor.compute_all_metrics(
                        generated_norm[:50],  # Sample 50 frames
                        target_norm[:50],
                        compute_fid=(self.current_epoch % 5 == 0),  # FID every 5 epochs
                        epoch=self.current_epoch
                    )
                    
                    print(f"\n  v3.1 Metrics:")
                    if 'fid' in metrics:
                        print(f"    FID: {metrics['fid']:.2f}")
                    if 'ms_ssim' in metrics:
                        print(f"    MS-SSIM: {metrics['ms_ssim']:.4f}")
                    if 'combined_score' in metrics:
                        print(f"    Combined Score: {metrics['combined_score']:.4f}")
                    
                    # Add to validation losses
                    val_losses[0].update(metrics)
        
        # Average losses
        avg_losses = {}
        for key in val_losses[0].keys():
            avg_losses[key] = np.mean([loss[key] for loss in val_losses if key in loss])
        
        self.generator.train()
        return avg_losses
    
    def train_epoch(self, stage='full'):
        """Train for one epoch with verbose logging"""

        epoch_losses = []
        import time

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} ({stage})")
        batch_start_time = time.time()

        for batch_idx, batch in enumerate(pbar):
            losses = self.train_step(batch, stage=stage)
            epoch_losses.append(losses)

            # Update progress bar with losses
            pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items() if 'total' in k or 'adv' in k})

            # Verbose logging every 100 batches
            if (batch_idx + 1) % 100 == 0:
                batch_time = time.time() - batch_start_time
                samples_per_sec = 100 * self.config.get('batch_size', 4) * self.num_gpus / batch_time

                # GPU memory usage
                if torch.cuda.is_available():
                    gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                    print(f"\n  [Batch {batch_idx+1}] "
                          f"Speed: {samples_per_sec:.1f} samples/s | "
                          f"GPU Mem: {gpu_memory_allocated:.2f}/{gpu_memory_reserved:.2f} GB | "
                          f"Loss: {losses.get('g_total', 0):.4f}", flush=True)

                batch_start_time = time.time()

        # Average losses
        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[key] = np.mean([loss[key] for loss in epoch_losses])

        return avg_losses
    
    def freeze_audio_encoder(self):
        """Freeze audio encoder for stage 1 (coarse training)"""
        model = self.generator.module if hasattr(self.generator, 'module') else self.generator
        for param in model.audio_encoder.parameters():
            param.requires_grad = False
        print("  ✓ Audio encoder frozen")

    def unfreeze_audio_encoder(self):
        """Unfreeze audio encoder for stage 2+ training"""
        model = self.generator.module if hasattr(self.generator, 'module') else self.generator
        for param in model.audio_encoder.parameters():
            param.requires_grad = True
        print("  ✓ Audio encoder unfrozen")
    
    def train(self):
        """
        Complete progressive training pipeline with v3.1 enhancements
        
        Stage 1 (Coarse): Learn basic mouth shapes without discriminator
        Stage 2 (Sync): Add sync loss, still no discriminator
        Stage 3 (Full): Add adversarial training with v3.1 adaptive WGAN-GP
        """
        
        print("\n" + "="*60)
        print("Starting Progressive Training with v3.1 Optimizations")
        print("="*60)
        
        # Setup real images for metrics
        if self.metric_monitor is not None:
            self._setup_metric_real_images()
        
        # ==================== Stage 1: Coarse Training ====================
        if self.current_stage == 'coarse' and self.current_epoch < 20:
            print("\n" + "="*60)
            print(f"STAGE 1: Coarse Training (L1 + Perceptual) - Resuming from Epoch {self.current_epoch}")
            print("="*60)

            self.freeze_audio_encoder()

            for epoch in range(self.current_epoch, 20):
                self.current_epoch = epoch
                self.update_perceptual_warmup(epoch)

                # Train
                train_losses = self.train_epoch(stage='coarse')

                # Validate
                val_losses = self.validate(stage='coarse', compute_metrics=(epoch % 5 == 0))

                # Log
                print(f"\nEpoch {epoch}:")
                print(f"  Train - L1: {train_losses.get('g_l1', 0):.4f} | "
                      f"Perceptual: {train_losses.get('g_perceptual', 0):.4f}")
                print(f"  Val   - L1: {val_losses.get('l1', 0):.4f} | "
                      f"Total: {val_losses.get('total', 0):.4f}")

                if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
                    wandb.log({
                        'epoch': epoch,
                        'stage': 1,
                        **{f'train/{k}': v for k, v in train_losses.items()},
                        **{f'val/{k}': v for k, v in val_losses.items()}
                    })

                # Save checkpoint
                if val_losses['total'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total']
                    self.save_checkpoint(epoch, is_best=True, stage='coarse')

                # Learning rate scheduling
                self.scheduler_g.step(val_losses['total'])
        
        # ==================== Stage 2: Sync Training ====================
        if self.current_stage in ['coarse', 'sync'] and self.current_epoch < 40:
            print("\n" + "="*60)
            print(f"STAGE 2: Sync Training (+ Audio-Visual Sync) - Resuming from Epoch {max(self.current_epoch, 20)}")
            print("="*60)

            self.current_stage = 'sync'
            self.unfreeze_audio_encoder()

            for epoch in range(max(self.current_epoch, 20), 40):
                self.current_epoch = epoch
                self.update_perceptual_warmup(epoch)

                # Train
                train_losses = self.train_epoch(stage='sync')

                # Validate
                val_losses = self.validate(stage='sync', compute_metrics=(epoch % 5 == 0))

                # Log
                print(f"\nEpoch {epoch}:")
                print(f"  Train - L1: {train_losses.get('g_l1', 0):.4f} | "
                      f"Sync: {train_losses.get('g_sync', 0):.4f}")
                print(f"  Val   - Total: {val_losses.get('total', 0):.4f}")

                if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
                    wandb.log({
                        'epoch': epoch,
                        'stage': 2,
                        **{f'train/{k}': v for k, v in train_losses.items()},
                        **{f'val/{k}': v for k, v in val_losses.items()}
                    })

                # Save checkpoint
                if val_losses['total'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total']
                    self.save_checkpoint(epoch, is_best=True, stage='sync')

                # Learning rate scheduling
                self.scheduler_g.step(val_losses['total'])
        
        # ==================== Stage 3: Full Adversarial Training ====================
        if self.current_stage in ['coarse', 'sync', 'full'] and self.current_epoch < 100:
            print("\n" + "="*60)
            print(f"STAGE 3: Full Adversarial Training (+ v3.1 Adaptive WGAN-GP) - Resuming from Epoch {max(self.current_epoch, 40)}")
            print("="*60)

            self.current_stage = 'full'

            for epoch in range(max(self.current_epoch, 40), 100):
                self.current_epoch = epoch
                self.update_perceptual_warmup(epoch)

                # Train
                train_losses = self.train_epoch(stage='full')

                # Validate
                val_losses = self.validate(stage='full', compute_metrics=(epoch % 5 == 0))

                # Log
                print(f"\nEpoch {epoch}:")
                print(f"  Train - G_total: {train_losses.get('g_total', 0):.4f} | "
                      f"D_loss: {train_losses.get('d_loss', 0):.4f} | "
                      f"Adaptive λ_gp: {train_losses.get('adaptive_lambda_gp', 0):.2f}")
                if 'wasserstein' in train_losses:
                    print(f"         Wasserstein: {train_losses['wasserstein']:.4f} | "
                          f"GP: {train_losses['gradient_penalty']:.4f}")
                print(f"  Val   - Total: {val_losses.get('total', 0):.4f}")

                if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
                    wandb.log({
                        'epoch': epoch,
                        'stage': 3,
                        **{f'train/{k}': v for k, v in train_losses.items()},
                        **{f'val/{k}': v for k, v in val_losses.items()}
                    })

                # Save checkpoint
                if val_losses['total'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total']
                    self.save_checkpoint(epoch, is_best=True, stage='full')

                # Learning rate scheduling
                self.scheduler_g.step(val_losses['total'])
                self.scheduler_d.step(train_losses.get('d_loss', 0))

                # Periodic checkpoint
                if epoch % 10 == 0:
                    self.save_checkpoint(epoch, is_best=False, stage='full')
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
    
    def save_checkpoint(self, epoch, is_best=False, stage='full'):
        """Save model checkpoint and maintain last 3 epochs"""
        checkpoint = {
            'epoch': epoch,
            'stage': stage,
            'generator_state': self.generator.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'opt_g_state': self.opt_g.state_dict(),
            'opt_d_state': self.opt_d.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        checkpoint_dir = Path(self.config['checkpoint_dir'])

        # Save latest
        path = checkpoint_dir / f'checkpoint_epoch{epoch}_{stage}.pt'
        torch.save(checkpoint, path)

        # Save best
        if is_best:
            best_path = checkpoint_dir / f'best_model_{stage}.pt'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best checkpoint: {best_path}")
        else:
            print(f"  ✓ Saved checkpoint: {path}")

        # Clean up old checkpoints - keep only last 3 epochs (plus best)
        # Find all non-best checkpoints for this stage
        all_checkpoints = list(checkpoint_dir.glob(f'checkpoint_epoch*_{stage}.pt'))
        if len(all_checkpoints) > 3:
            # Sort by epoch number
            def get_epoch_num(p):
                try:
                    parts = p.stem.split('_')
                    return int(parts[1].replace('epoch', ''))
                except:
                    return -1

            all_checkpoints.sort(key=get_epoch_num, reverse=True)
            # Keep the 3 most recent, delete the rest
            for old_checkpoint in all_checkpoints[3:]:
                try:
                    old_checkpoint.unlink()
                    print(f"  ✓ Removed old checkpoint: {old_checkpoint.name}")
                except Exception as e:
                    print(f"  ⚠ Failed to remove {old_checkpoint.name}: {e}")


# Entry point for training
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete LipSync Training with v3.1")
    parser.add_argument('--data_dir', default='./processed_data', help='Processed data directory')
    parser.add_argument('--checkpoint_dir', default='./checkpoints_v31', help='Checkpoint directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Data loader workers')
    parser.add_argument('--lr_g', type=float, default=1e-4, help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='Discriminator learning rate')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--use_wgan_gp', action='store_true', default=True, help='Use WGAN-GP loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='Base gradient penalty coefficient')
    parser.add_argument('--n_critic', type=int, default=5, help='Discriminator updates per generator update')
    parser.add_argument('--no_monitor_metrics', action='store_true', help='Disable FID+MS-SSIM monitoring')
    parser.add_argument('--use_sparse_attention', action='store_true', default=True, help='Use sparse transformer attention')
    parser.add_argument('--use_compile', action='store_true', default=True, help='Use torch.compile')
    parser.add_argument('--run_name', type=str, default='v31-training', help='Run name for logging')
    parser.add_argument('--syncnet_checkpoint', type=str, default=None, help='Pretrained SyncNet checkpoint')
    
    args = parser.parse_args()
    
    config = vars(args)
    
    print("="*60)
    print("Complete LipSync GAN Training (v3.1 Enhanced)")
    print("="*60)
    print(f"Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("="*60)
    
    trainer = CompleteTrainer(config)
    trainer.train()
