# LipSync GAN - Audio-Visual Speech Synthesis

A complete audio-visual lip-sync generation system using WGAN-GP and Sparse Attention, optimized for H100 GPU training.

## Overview

This project implements a progressive GAN training pipeline for generating photorealistic lip movements synchronized with audio input. The system uses:

- **Progressive 3-stage training**: Coarse → Sync → Full Adversarial
- **WGAN-GP** with adaptive gradient penalty
- **Sparse Transformer** audio encoder (30-50% compute reduction)
- **Multi-GPU training** with DataParallel (8× H100 80GB HBM3)
- **Complete preprocessing pipeline** for hierarchical video datasets

## Model Architecture

- **Generator**: CompleteLipSyncModel (18.01M params)
  - Sparse Transformer audio encoder
  - Cross-modal attention fusion
  - Identity encoder for person-specific conditioning
  - Temporal smoother with optical flow

- **Discriminator**: AdaptiveWGANGPDiscriminator (24.87M params)
  - Multi-scale discrimination
  - 3D convolutional architecture
  - Adaptive gradient penalty

**Total**: 42.88M parameters (163.58 MB)

## Dataset

- **Source**: 37,402 videos, 625 hours (26 days), 258 GB
- **Processed**: 35,105 videos (93.9%)
- **Train sequences**: 113,826
- **Validation sequences**: 15,604
- **Total sequences**: 129,430
- **Speakers**: 658 persons

## Key Features

✅ **Smart preprocessing** with person-level detection (skips already processed)
✅ **GPU-accelerated** preprocessing with MediaPipe face detection
✅ **Multi-GPU training** with automatic DataParallel
✅ **Progressive training** over 100 epochs (3 stages)
✅ **WandB integration** for experiment tracking
✅ **Automatic checkpoint management** with cleanup
✅ **FID and MS-SSIM** validation metrics

## Quick Start

### 1. Verify Data Structure
```bash
python check_struct.py --data_dir ./transcoded_data
```

### 2. Preprocess Videos
```bash
# Smart batch preprocessing (only processes unprocessed persons)
python preprocess_smart_batch.py
```

### 3. Train Model
```bash
python train.py \
  --data_dir ./processed_data \
  --checkpoint_dir ./checkpoints \
  --batch_size 8 \
  --use_wandb \
  --run_name "lipsync-run1"
```

### 4. Run Inference
```bash
python lipsync_inference.py \
  --checkpoint ./checkpoints/best_model_full.pt \
  --video input.mp4 \
  --audio input.wav \
  --output output.mp4
```

## Documentation

- **[COMPLETE_PROJECT_DOCUMENTATION.md](COMPLETE_PROJECT_DOCUMENTATION.md)**: Full technical documentation (1,550+ lines)
- **[CLAUDE.md](CLAUDE.md)**: Quick reference guide for Claude Code
- **[README_ADAPTIVE_TRAINING.md](README_ADAPTIVE_TRAINING.md)**: Adaptive training setup

## Core Files

### Training & Model
- `train.py` - Main training script with progressive 3-stage training
- `model.py` - Complete model architecture (Generator, Discriminator, SyncNet)
- `optimizations.py` - v3.1 utilities (MultiMetricMonitor, sparse attention)

### Preprocessing
- `pipeline.py` - Complete preprocessing pipeline
- `preprocess_hie.py` - Hierarchical data preprocessing (CPU/GPU)
- `preprocess_smart_batch.py` - GPU-accelerated batch preprocessing
- `check_struct.py` - Data structure verification

### Inference
- `lipsync_inference.py` - Main inference script
- `production_inference.py` - Production-ready inference
- `inference.py` - Core inference utilities

### Utilities
- `calculate_total_duration.py` - Video duration calculator
- `check_person_completion.py` - Person-level completion checker
- `verify_preprocessing.py` - Data integrity checker
- `find_missing_videos.py` - Missing file finder
- `build_index.py` - Dataset index builder

## Performance

- **Training speed**: ~5.9s per batch, 23.3 hours per epoch
- **Total training time**: ~97 days (100 epochs on 8× H100)
- **Throughput**: ~10.8 samples/second (batch_size=8 per GPU)
- **Preprocessing**: ~18s per video (with MFA), ~6s (without MFA)

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA 12.8+
- 8× NVIDIA H100 80GB HBM3 (or similar GPUs)
- 128+ GB RAM
- ~1 TB storage (source + processed data + checkpoints)

## Current Status

- **Completed**: Epoch 0 (validation loss: 23.26)
- **Stage**: 1 (Coarse Training)
- **Checkpoint**: `checkpoint_epoch0_coarse.pt`
- **Next**: Resume training from epoch 1

## Citation

This implementation combines techniques from:
- WGAN-GP (Gulrajani et al., 2017)
- Progressive GAN Training (Karras et al., 2018)
- Sparse Attention (Child et al., 2019)
- FiLM Conditioning (Perez et al., 2018)
- Montreal Forced Aligner (McAuliffe et al., 2017)

## License

[Add your license here]

---

**Generated with Claude Code** 🤖
