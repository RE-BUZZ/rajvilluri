# Complete LipSync GAN Project Documentation

**Audio-Visual Speech Synthesis with WGAN-GP and Sparse Attention**

This document provides complete instructions to recreate this lip-sync generation system from scratch, including all preprocessing, training, model architecture, and WandB integration details.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Environment Setup](#environment-setup)
3. [Data Structure](#data-structure)
4. [Preprocessing Pipeline](#preprocessing-pipeline)
5. [Model Architecture](#model-architecture)
6. [Training Pipeline](#training-pipeline)
7. [WandB Integration](#wandb-integration)
8. [Utility Scripts](#utility-scripts)
9. [Running the Complete Pipeline](#running-the-complete-pipeline)
10. [Configuration Reference](#configuration-reference)
11. [File Reference](#file-reference)
12. [Troubleshooting](#troubleshooting)
13. [Performance Benchmarks](#performance-benchmarks)
14. [Current Training Status](#current-training-status)

---

## Project Overview

This project implements a complete audio-visual lip-sync generation system using:

- **Architecture**: Progressive GAN training (Coarse → Sync → Full Adversarial)
- **Discriminator**: WGAN-GP with adaptive gradient penalty (v3.1)
- **Audio Encoder**: Sparse Transformer with windowed attention (O(n log n) complexity)
- **Generator**: CompleteLipSyncModel with FiLM conditioning and temporal smoothing
- **Optimization**: Multi-GPU training with DataParallel, mixed precision (AMP)
- **Monitoring**: WandB logging with FID and MS-SSIM metrics

### Key Statistics

- **Model Size**: 42.88M trainable parameters
  - Generator: 18.01M params (68.71 MB)
  - Discriminator: 24.87M params (94.88 MB)
  - Total: 42.88M params (163.58 MB)
- **Training Data**:
  - Source: 37,402 videos, 625.17 hours (26.05 days), 258 GB
  - Preprocessed: 35,105 videos (93.9% of source)
  - Train sequences: 113,826
  - Validation sequences: 15,604
  - Total sequences: 129,430
  - Persons: 658 (female1-327, male1-342)
- **Performance**:
  - Training speed: ~5.9s per batch, 23.3 hours per epoch
  - Total training time: ~97 days (100 epochs)
  - Throughput: ~10.8 samples/second (batch_size=8 per GPU)

---

## Environment Setup

### System Requirements

- **GPU**: 8× NVIDIA H100 80GB HBM3 (or similar high-end GPUs)
- **OS**: Linux (tested on Ubuntu with kernel 6.8.0-85)
- **Python**: 3.10+
- **CUDA**: 12.8+ (for PyTorch 2.x)
- **Storage**: ~1 TB total
  - Source data: 258 GB
  - Preprocessed data: ~500 GB
  - Checkpoints: ~500 MB per checkpoint
- **RAM**: 128 GB+ recommended
- **CPUs**: 80+ cores for preprocessing (32+ workers × 8 GPUs)

### Python Dependencies

Create a virtual environment and install dependencies:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support
pip install torch==2.x torchaudio==2.x --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install numpy scipy
pip install opencv-contrib-python
pip install mediapipe
pip install librosa torchaudio
pip install transformers
pip install phonemizer
pip install webrtcvad
pip install tqdm

# Install optional dependencies for v3.1 features
pip install torchmetrics  # For MS-SSIM metric
pip install torch-fid  # For FID metric

# Install WandB for experiment tracking
pip install wandb==0.22.2

# Install Montreal Forced Aligner (MFA) for accurate phoneme alignment
# See: https://montreal-forced-aligner.readthedocs.io/
conda install -c conda-forge montreal-forced-aligner

# Install TextGrid parser for MFA
pip install tgt

# Install VGG for perceptual loss
pip install torchvision
```

### System Dependencies

```bash
# FFmpeg for video/audio processing
sudo apt-get install ffmpeg

# espeak for phonemization
sudo apt-get install espeak espeak-data

# Build tools
sudo apt-get install build-essential
```

### WandB Setup

```bash
# Login to Weights & Biases
wandb login

# Or set API key directly
export WANDB_API_KEY=your_api_key_here
```

---

## Data Structure

### Expected Input Structure

Your raw video data should be organized hierarchically by person (speaker):

```
transcoded_data/
  ├── female1/
  │   ├── female1-a.mp4
  │   ├── female1-b.mp4
  │   ├── female1-c.mp4
  │   └── ...
  ├── female2/
  │   ├── female2-001.mp4
  │   ├── female2-002.mp4
  │   └── ...
  ├── male1/
  │   ├── male1-video1.mp4
  │   └── ...
  └── ...
```

**Key Points:**
- Each person has their own folder
- All videos for a person are in that folder
- Videos should be .mp4 format (or specify `--pattern` for other formats)
- Audio should be embedded in video files
- Average video length: ~60 seconds

### Output Structure

After preprocessing, data is organized into train/val splits:

```
processed_data/
  ├── train/
  │   ├── female1_video1_processed.pkl
  │   ├── female1_video2_processed.pkl
  │   ├── male1_video1_processed.pkl
  │   └── ... (31,168+ files)
  ├── val/
  │   ├── female1_video3_processed.pkl
  │   ├── male2_video1_processed.pkl
  │   └── ... (3,409+ files)
  └── .index_cache_train.pkl  # Auto-generated indices
```

### Processed Data Format

Each `.pkl` file contains:

```python
{
    'sequences': [
        {
            'face_crops': [JPEG bytes] or np.ndarray,  # [T, 512, 512, 3]
            'mouth_crops': [JPEG bytes] or np.ndarray, # [T, 256, 256, 3]
            'visemes': np.ndarray,                     # [T] viseme IDs (0-20)
            'mel_windows': np.ndarray,                 # [T, 80, 7] mel spectrogram
            'frame_indices': np.ndarray,               # [T] frame numbers
            'is_jpeg_compressed': bool                 # Compression flag
        },
        # ... more sequences
    ],
    'num_frames': int,
    'fps': float,
    'audio_sr': int  # 16000
}
```

---

## Preprocessing Pipeline

### Overview

The preprocessing pipeline extracts and aligns:
1. **Face crops** (512×512) - for identity encoding
2. **Mouth crops** (256×256) - generation target with pose normalization
3. **Mel spectrograms** (80 bins) - audio features
4. **Phoneme alignment** - using MFA or energy-based fallback
5. **Viseme labels** (21 classes) - phoneme-to-viseme mapping

### Step 1: Verify Data Structure

Before preprocessing, verify your data structure:

```bash
python check_struct.py --data_dir ./transcoded_data
```

**Output:**
- List of all person folders and video counts
- Total size and estimated processing time
- Recommended preprocessing commands

### Step 2: Calculate Video Duration

Check total hours of video data:

```bash
python calculate_total_duration.py --data_dir ./transcoded_data
```

**Output:**
```
Total videos: 37,402
Total duration: 625.17 hours (26.05 days)
Average duration: 60.17 seconds
```

### Step 3: Run Preprocessing

#### Quick Start (Small Dataset <50 videos)

```bash
python preprocess_hie.py \
  --data_dir ./transcoded_data \
  --output ./processed_data \
  --auto_split \
  --val_ratio 0.1 \
  --augment
```

#### Full Pipeline (Large Dataset)

```bash
# With MFA (slow but accurate)
python preprocess_hie.py \
  --data_dir ./transcoded_data \
  --output ./processed_data \
  --auto_split \
  --val_ratio 0.1 \
  --split_by_person \
  --augment \
  --workers 32 \
  --fps 25

# Without MFA (3x faster)
python preprocess_hie.py \
  --data_dir ./transcoded_data \
  --output ./processed_data \
  --auto_split \
  --val_ratio 0.1 \
  --no_mfa \
  --workers 32
```

#### GPU-Accelerated Preprocessing (8 GPUs)

For large-scale preprocessing with multi-GPU acceleration:

```bash
# Uses all 8 GPUs for parallel processing
python preprocess_smart_batch.py \
  --data_dir ./transcoded_data \
  --output_dir ./processed_data \
  --workers 32
```

**Features:**
- Automatically distributes work across 8 GPUs
- Skips already-preprocessed persons
- GPU acceleration for MediaPipe face detection
- Progress tracking and error handling

### Step 4: Verify Preprocessing Completion

Check if all videos have been preprocessed:

```bash
# Check person-level completion
python check_person_completion.py \
  --source_dir ./transcoded_data \
  --processed_dir ./processed_data

# Find specific missing videos
python find_missing_videos.py \
  --source_dir ./transcoded_data \
  --processed_dir ./processed_data
```

### Step 5: Reprocess Missing Videos (If Needed)

If some videos failed or are missing:

```bash
# Video-level GPU preprocessing for missing files
python reprocess_missing_videos.py \
  --source_dir ./transcoded_data \
  --processed_dir ./processed_data \
  --workers 16  # Reduce if getting CUDA errors
```

**Features:**
- Finds individual missing videos
- GPU-accelerated processing
- Distributes across 8 GPUs
- Handles CUDA memory carefully

### Step 6: Verify Data Integrity

Test preprocessed files for corruption:

```bash
python verify_preprocessing.py \
  --processed_dir ./processed_data/train \
  --workers 32
```

### Preprocessing Configuration

**Key Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fps` | 25 | Target video FPS |
| `mouth_size` | 256×256 | Mouth crop resolution |
| `face_size` | 512×512 | Face crop resolution |
| `audio_sr` | 16000 | Audio sample rate |
| `mel_bins` | 80 | Mel spectrogram bins |
| `use_mfa` | True | Montreal Forced Aligner |
| `apply_augmentation` | False | SpecAugment for training |
| `use_jpeg_compression` | True | Compress crops (75% quality) |
| `max_seq_length` | 75 | Maximum frames per sequence |
| `sequence_overlap` | 40 | Overlap between sequences |
| `vad_aggressiveness` | 2 | Voice Activity Detection (0-3) |

### Viseme Mapping (21 Classes)

The system uses 21 viseme classes for mouth shape representation:

| Viseme ID | Phonemes | Description |
|-----------|----------|-------------|
| 0 | p, b, m | Bilabial closure |
| 1 | f, v | Labiodental |
| 2 | T, D | Interdental (th) |
| 3 | t, d, n | Alveolar stop |
| 4 | l | Alveolar lateral |
| 5 | s, z | Alveolar fricative |
| 6 | S, Z, tS, dZ | Post-alveolar (sh, ch) |
| 7 | r | Approximant r |
| 8 | j | Palatal approximant (y) |
| 9 | w | Labial-velar (w) |
| 10 | k, g, N | Velar |
| 11 | h | Glottal |
| 12 | i, I | Close front (ee, i) |
| 13 | e, E | Mid front (eh) |
| 14 | ae | Near-open front (a) |
| 15 | a, A | Open (ah) |
| 16 | o, O | Mid back (oh) |
| 17 | u, U | Close back (oo) |
| 18 | @ | Schwa (uh) |
| 19 | aI, aU, OI | Diphthongs |
| 20 | sil, sp | Silence/pause |

---

## Model Architecture

### Complete Model: CompleteLipSyncModel

**Total Parameters**: 18.01M (68.71 MB)

```python
CompleteLipSyncModel(
  num_visemes=21,
  use_checkpoint=False,  # Disabled with DataParallel
  use_sparse_attention=True  # v3.1 optimization
)
```

#### Components:

1. **Audio Encoder** (SparseTransformerAudioEncoder)
   - Input: Mel spectrogram windows [B, T, 80, 7]
   - Architecture: CNN → Positional Encoding → Windowed Transformer
   - Sparse attention: O(n log n) complexity vs O(n²) standard
   - Window size: 7 frames (causal masking)
   - 2 transformer layers, 4 attention heads
   - Output: Audio features [B, T, 256]
   - **30-50% compute reduction** vs standard transformer

2. **Viseme Embedding**
   - Embedding layer: 21 visemes → 256 dimensions
   - MLP encoder: 256 → 512 dimensions
   - Dropout: 0.1

3. **Cross-Modal Attention**
   - Multi-head attention (8 heads)
   - Fuses viseme features [B, T, 512] with audio [B, T, 256]
   - Output: Fused features [B, T, 512]
   - Attention weights: [B, 8, T, T]

4. **Identity Encoder**
   - Input: First frame face crop [B, 3, 512, 512]
   - CNN with adaptive pooling → 256-dim identity vector
   - Provides person-specific conditioning
   - Preserves identity across generated frames

5. **Mouth Generator** (MouthGenerator)
   - Input: Concatenated features [512 + 256] + identity [256]
   - Architecture: Linear → ConvTranspose layers with FiLM conditioning
   - 5 upsampling stages: 8×8 → 16×16 → 32×32 → 64×64 → 128×128 → 256×256
   - ResBlocks with GroupNorm (8 groups)
   - Dropout: 0.1
   - Output: Generated mouth [B, T, 3, 256, 256]

6. **Temporal Smoother**
   - Optical flow warping between adjacent frames
   - 3D convolutions for temporal consistency
   - Flow predictor: 2D CNN → optical flow [B, T-1, 2, 256, 256]
   - Reduces flickering artifacts

### Discriminator: AdaptiveWGANGPDiscriminator

**Parameters**: 24.87M (94.88 MB)

```python
AdaptiveWGANGPDiscriminator(num_visemes=21)
```

#### Features:

- **Multi-scale discrimination** (3 scales: original, 2×, 4× downsampled)
- **3D Convolutional architecture** for temporal awareness
- **Adaptive gradient penalty**: λ_gp decays during training
  - Early epochs: λ_gp = 10.0 (strong regularization)
  - Late epochs: λ_gp → 1.0 (allow convergence)
  - Formula: `λ_gp(t) = 10.0 - 9.0 * min(1.0, t / 50)`
- **Auxiliary viseme classification** head for additional supervision
- **PatchGAN architecture**: Outputs validity map instead of single scalar

### Total Model

- **Generator**: 18.01M params (68.71 MB)
- **Discriminator**: 24.87M params (94.88 MB)
- **Total**: 42.88M params (163.58 MB)

---

## Training Pipeline

### Progressive Training (3 Stages)

The model is trained progressively:

1. **Stage 1 (Epochs 0-19): Coarse Training**
   - Loss: L1 + Perceptual (VGG16)
   - Audio encoder **frozen**
   - Focus: Basic mouth shape generation
   - Validation metrics every epoch
   - FID/MS-SSIM every 5 epochs

2. **Stage 2 (Epochs 20-39): Sync Training**
   - Loss: L1 + Perceptual + Sync
   - Audio encoder **unfrozen**
   - Focus: Audio-visual synchronization
   - SyncNet active
   - Validation metrics every epoch

3. **Stage 3 (Epochs 40-99): Full Adversarial**
   - Loss: L1 + Perceptual + Sync + WGAN-GP Adversarial
   - All components active
   - n_critic = 5 (discriminator updates per generator)
   - Focus: Photorealistic texture and temporal consistency
   - Checkpoint every 10 epochs (epochs 50, 60, 70, 80, 90)

### Training Configuration

**Actual Configuration Used:**

```bash
python train.py \
  --data_dir /data/gpunet_admin/processed_data \
  --checkpoint_dir ./gant580_checkpoints \
  --batch_size 8 \
  --num_workers 16 \
  --lr_g 1e-4 \
  --lr_d 1e-4 \
  --use_wandb \
  --use_wgan_gp \
  --lambda_gp 10 \
  --n_critic 5 \
  --use_sparse_attention \
  --run_name gant580-full-dataset
```

**Hardware:**
- 8× NVIDIA H100 80GB HBM3
- DataParallel across all 8 GPUs
- Effective batch size: 64 (8 per GPU × 8 GPUs)
- Mixed precision (AMP): Enabled
- Gradient checkpointing: Disabled (incompatible with DataParallel)

**Hyperparameters:**

```yaml
Model:
  - use_sparse_attention: True
  - num_visemes: 21
  - use_checkpoint: False  # Incompatible with DataParallel

Optimization:
  - optimizer: Adam (β1=0.5, β2=0.999)
  - lr_g: 1e-4
  - lr_d: 1e-4
  - batch_size: 8 per GPU
  - effective_batch_size: 64
  - num_workers: 16 per GPU (128 total)
  - use_amp: True
  - gradient_clipping: None

Training:
  - total_epochs: 100
  - batches_per_epoch: 14,228
  - n_critic: 5
  - lambda_gp: 10 (base), adaptive decay

Regularization:
  - dropout: 0.1-0.2
  - gradient_penalty: Adaptive WGAN-GP
  - perceptual_warmup: Linear over 10 epochs
```

### Starting Training

#### From Scratch

```bash
python train.py \
  --data_dir ./processed_data \
  --checkpoint_dir ./checkpoints_v31 \
  --batch_size 8 \
  --num_workers 16 \
  --lr_g 1e-4 \
  --lr_d 1e-4 \
  --use_wandb \
  --run_name "lipsync-v31-run1"
```

#### Multi-GPU Training (Automatic)

```bash
# Uses DataParallel automatically if multiple GPUs available
# No special configuration needed
python train.py \
  --data_dir ./processed_data \
  --checkpoint_dir ./checkpoints_v31 \
  --batch_size 8 \
  --use_wandb
```

#### Resume from Checkpoint

Training automatically resumes from the latest checkpoint:

```bash
# Will load checkpoint_epoch{N}_{stage}.pt automatically
python train.py \
  --data_dir ./processed_data \
  --checkpoint_dir ./checkpoints_v31 \
  --use_wandb
```

**Output:**
```
✓ Loading checkpoint from: checkpoint_epoch0_coarse.pt
✓ Resumed from Epoch 0 (coarse stage)
✓ Best validation loss: 23.26
✓ Will continue from Epoch 1
```

### Loss Function Components

**Generator Loss:**

```
L_G = λ_L1 * L_L1 + λ_perceptual * L_perceptual + λ_sync * L_sync + λ_adv * L_adv
```

Where:
- `λ_L1 = 1.0`: L1 reconstruction loss
- `λ_perceptual = 1.0`: VGG16 perceptual loss (warmup over 10 epochs)
- `λ_sync = 0.1`: Audio-visual sync loss (Stage 2+)
- `λ_adv = 0.01`: Adversarial loss (Stage 3 only)

**Discriminator Loss (WGAN-GP):**

```
L_D = -(D(real) - D(fake)) + λ_gp(t) * GP
```

Where:
- `λ_gp(t)`: Adaptive gradient penalty (10 → 1 over 50 epochs)
- `GP`: Gradient penalty on interpolated samples

### Checkpoint Management

**Automatic Checkpoint Saving:**
- Saves when validation loss improves: `best_model_{stage}.pt`
- **Keeps last 3 epoch checkpoints** per stage: `checkpoint_epoch{N}_{stage}.pt`
- Stage 3 additional saves: Every 10 epochs (50, 60, 70, 80, 90)
- Auto-cleanup: Deletes older checkpoints automatically

**Checkpoint Contents:**

```python
{
    'epoch': int,
    'stage': 'coarse' | 'sync' | 'full',
    'generator_state': OrderedDict,
    'discriminator_state': OrderedDict,
    'opt_g_state': OrderedDict,
    'opt_d_state': OrderedDict,
    'best_val_loss': float,
    'config': dict
}
```

**Checkpoint Directory Structure:**

```
gant580_checkpoints/
├── best_model_coarse.pt          # Best stage 1 model
├── best_model_sync.pt            # Best stage 2 model (created later)
├── best_model_full.pt            # Best stage 3 model (created later)
├── checkpoint_epoch0_coarse.pt   # Latest coarse checkpoints
├── checkpoint_epoch{N}_coarse.pt # (keeps last 3)
└── fid_real/                     # Real images for FID computation
    └── real_*.png (100 images)
```

### Validation Metrics

Computed every epoch:

- **L1 Loss**: Pixel-wise reconstruction error
- **Perceptual Loss**: VGG16 feature similarity
- **Sync Loss**: Audio-video synchronization score (Stage 2+)
- **FID** (every 5 epochs): Fréchet Inception Distance
- **MS-SSIM** (every 5 epochs): Multi-Scale Structural Similarity

---

## WandB Integration

### Setup

```python
import wandb

wandb.init(
    project="lipsync-v31",
    name="gant580-full-dataset",
    config={
        'batch_size': 8,
        'lr_g': 1e-4,
        'lr_d': 1e-4,
        'n_critic': 5,
        'lambda_gp': 10,
        'use_sparse_attention': True,
        # ... all hyperparameters
    }
)
```

### Logged Metrics

**Per Batch (every 100 batches):**
- Training speed (it/s)
- GPU memory usage
- Gradient norms
- Individual loss components

**Per Epoch:**

Training metrics:
```python
{
    'epoch': int,
    'stage': int,  # 1=coarse, 2=sync, 3=full
    'train/g_total': float,
    'train/g_l1': float,
    'train/g_perceptual': float,
    'train/g_sync': float,  # Stage 2+
    'train/g_adv': float,   # Stage 3
    'train/d_loss': float,  # Stage 3
    'train/wasserstein': float,
    'train/gradient_penalty': float,
    'train/adaptive_lambda_gp': float,
    'train/samples_per_sec': float
}
```

Validation metrics:
```python
{
    'val/total': float,
    'val/l1': float,
    'val/perceptual': float,
    'val/sync': float,
    'val/fid': float,         # Every 5 epochs
    'val/ms_ssim': float      # Every 5 epochs
}
```

### WandB Dashboard

Access your training dashboard at: `https://wandb.ai/<username>/lipsync-v31`

**Current Run:** https://wandb.ai/shreeshman/lipsync-v31

**Visualizations:**
- Loss curves (train/val)
- Learning rate schedules
- GPU utilization and memory
- FID/MS-SSIM trends over time
- Training speed (samples/sec)

---

## Utility Scripts

### Data Verification & Analysis

#### 1. check_struct.py
Validates data directory structure and provides statistics.

```bash
python check_struct.py --data_dir ./transcoded_data
```

**Output:**
- List of person folders
- Video counts per person
- Total size and duration estimates
- Recommended preprocessing commands

#### 2. calculate_total_duration.py
Calculates total video duration using ffprobe.

```bash
python calculate_total_duration.py --data_dir ./transcoded_data
```

**Output:**
- Total videos: 37,402
- Total duration: 625.17 hours (26.05 days)
- Average: 60.17 seconds per video
- Results saved to `video_duration_summary.txt`

#### 3. check_person_completion.py
Checks preprocessing completion at person level.

```bash
python check_person_completion.py \
  --source_dir ./transcoded_data \
  --processed_dir ./processed_data
```

**Output:**
- Fully complete persons
- Partially complete persons (with missing counts)
- Not started persons

#### 4. find_missing_videos.py
Identifies specific missing videos at video level.

```bash
python find_missing_videos.py \
  --source_dir ./transcoded_data \
  --processed_dir ./processed_data
```

**Output:**
- List of missing videos
- Count by person
- Total missing count

#### 5. verify_preprocessing.py
Tests preprocessed files for corruption.

```bash
python verify_preprocessing.py \
  --processed_dir ./processed_data/train \
  --workers 32
```

**Output:**
- Number of valid files
- Number of corrupted files
- List of corrupted files (if any)

### Preprocessing Scripts

#### 6. preprocess_hie.py
Main hierarchical preprocessing script (CPU/single-GPU).

```bash
python preprocess_hie.py \
  --data_dir ./transcoded_data \
  --output ./processed_data \
  --auto_split \
  --augment \
  --workers 32
```

**Features:**
- Multi-process parallel processing
- Train/val auto-splitting
- MFA support (optional)
- SpecAugment (optional)

#### 7. preprocess_smart_batch.py
GPU-accelerated batch preprocessing (8 GPUs).

```bash
python preprocess_smart_batch.py \
  --data_dir ./transcoded_data \
  --output_dir ./processed_data \
  --workers 32
```

**Features:**
- Person-level processing
- Skips already-processed persons
- Distributes across 8 GPUs
- GPU acceleration for face detection

#### 8. reprocess_missing_videos.py
Video-level reprocessing for missing/failed videos.

```bash
python reprocess_missing_videos.py \
  --source_dir ./transcoded_data \
  --processed_dir ./processed_data \
  --workers 16  # Reduce if CUDA errors occur
```

**Features:**
- Finds individual missing videos
- GPU-accelerated processing
- Distributes work across 8 GPUs round-robin
- Handles CUDA memory carefully
- Retries failed videos with reduced workers

**Performance:**
- With 32 workers: ~90% success rate (CUDA busy errors possible)
- With 16 workers: ~99% success rate (stable)
- Speed: ~6-18 seconds per video (depending on MFA)

### Training Scripts

#### 9. train.py
Main training script with progressive 3-stage training.

```bash
python train.py \
  --data_dir ./processed_data \
  --checkpoint_dir ./checkpoints_v31 \
  --batch_size 8 \
  --use_wandb
```

**Features:**
- Automatic checkpoint resuming
- Progressive training (3 stages)
- Multi-GPU DataParallel
- Mixed precision (AMP)
- WandB logging
- FID + MS-SSIM monitoring

---

## Running the Complete Pipeline

### End-to-End Workflow

```bash
# 1. Verify data structure
python check_struct.py --data_dir ./transcoded_data

# 2. Calculate total duration
python calculate_total_duration.py --data_dir ./transcoded_data

# 3. Preprocess data (GPU-accelerated, 8 GPUs)
python preprocess_smart_batch.py \
  --data_dir ./transcoded_data \
  --output_dir ./processed_data \
  --workers 32

# 4. Check preprocessing completion
python check_person_completion.py \
  --source_dir ./transcoded_data \
  --processed_dir ./processed_data

# 5. Reprocess any missing videos
python reprocess_missing_videos.py \
  --source_dir ./transcoded_data \
  --processed_dir ./processed_data \
  --workers 16

# 6. Verify preprocessed data integrity
python verify_preprocessing.py \
  --processed_dir ./processed_data/train \
  --workers 32

# 7. Login to WandB
wandb login

# 8. Start training (8 GPUs)
python train.py \
  --data_dir ./processed_data \
  --checkpoint_dir ./gant580_checkpoints \
  --batch_size 8 \
  --num_workers 16 \
  --use_wandb \
  --run_name "gant580-full-dataset"

# 9. Monitor training
# Open https://wandb.ai/<username>/lipsync-v31
```

### Resuming Interrupted Training

Training automatically resumes from the latest checkpoint:

```bash
# Just run train.py again - it will find the latest checkpoint
python train.py \
  --data_dir ./processed_data \
  --checkpoint_dir ./gant580_checkpoints \
  --use_wandb
```

**Output:**
```
✓ Loading checkpoint from: checkpoint_epoch0_coarse.pt
✓ Resumed from Epoch 0 (coarse stage)
✓ Best validation loss: 23.26
✓ Will continue from Epoch 1
```

---

## Configuration Reference

### PreprocessConfig

```python
@dataclass
class PreprocessConfig:
    # Video settings
    fps: int = 25
    mouth_size: Tuple[int, int] = (256, 256)
    face_size: Tuple[int, int] = (512, 512)

    # Audio settings
    audio_sr: int = 16000
    mel_bins: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    window_frames: int = 7

    # Sequence settings
    max_seq_length: int = 75
    sequence_overlap: int = 40

    # Processing options
    use_mfa: bool = True
    normalize_pose: bool = True
    apply_augmentation: bool = True
    use_gpu: bool = True
    gpu_id: int = 0

    # Compression
    use_jpeg_compression: bool = True
    jpeg_quality: int = 75  # 0-100

    # VAD settings
    vad_aggressiveness: int = 2  # 0-3
```

### Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 8 | Per-GPU batch size |
| `num_workers` | 16 | DataLoader workers per GPU |
| `lr_g` | 1e-4 | Generator learning rate |
| `lr_d` | 1e-4 | Discriminator learning rate |
| `lambda_gp` | 10 | Base gradient penalty coefficient |
| `n_critic` | 5 | Discriminator updates per generator step |
| `use_amp` | True | Mixed precision training |
| `use_sparse_attention` | True | Sparse attention (v3.1) |
| `use_wgan_gp` | True | WGAN-GP loss |

### Stage Boundaries

| Stage | Epochs | Losses | Audio Encoder | Description |
|-------|--------|--------|---------------|-------------|
| Coarse | 0-19 | L1 + Perceptual | Frozen | Basic shape learning |
| Sync | 20-39 | + Sync loss | Unfrozen | Audio-visual alignment |
| Full | 40-99 | + Adversarial | Unfrozen | Realistic generation |

---

## File Reference

### Core Files

| File | LOC | Description |
|------|-----|-------------|
| `train.py` | 1,293 | Main training script with progressive 3-stage training |
| `model.py` | 798 | Complete model architecture (Generator, Discriminator, SyncNet) |
| `pipeline.py` | 1,050 | Preprocessing pipeline (CompletePreprocessor) |
| `optimizations.py` | 343 | v3.1 utilities (MultiMetricMonitor, sparse attention) |

### Preprocessing Scripts

| File | LOC | Description |
|------|-----|-------------|
| `preprocess_hie.py` | 524 | Hierarchical data preprocessing (CPU/single-GPU) |
| `preprocess_smart_batch.py` | 150 | GPU-accelerated batch preprocessing (8 GPUs) |
| `reprocess_missing_videos.py` | 280 | Video-level reprocessing for missing files |

### Utility Scripts

| File | LOC | Description |
|------|-----|-------------|
| `check_struct.py` | 194 | Data structure verification tool |
| `calculate_total_duration.py` | 120 | Video duration calculator (uses ffprobe) |
| `check_person_completion.py` | 145 | Person-level completion checker |
| `find_missing_videos.py` | 110 | Video-level missing file finder |
| `verify_preprocessing.py` | 125 | Preprocessed data integrity checker |

### Model Components (model.py)

```python
# Audio Encoders
class AudioEncoder(nn.Module)                      # GRU-based (fallback)
class SparseTransformerAudioEncoder(nn.Module)     # v3.1 sparse attention (default)

# Cross-Modal Fusion
class CrossModalAttention(nn.Module)               # Multi-head attention (8 heads)
class FiLMLayer(nn.Module)                         # Feature-wise linear modulation

# Generator Components
class MouthGenerator(nn.Module)                    # Mouth synthesis (U-Net style)
class TemporalSmoother(nn.Module)                  # Optical flow + 3D conv
class OpticalFlowWarping(nn.Module)                # Flow prediction network

# Complete Models
class CompleteLipSyncModel(nn.Module)              # Main generator (18.01M params)
class AdaptiveWGANGPDiscriminator(nn.Module)       # v3.1 discriminator (24.87M params)
class SyncDiscriminator(nn.Module)                 # Sync network (used in Stage 2+)

# Loss Functions
class CompleteLoss(nn.Module)                      # Unified loss with stage awareness
class FixedSyncLoss(nn.Module)                     # Audio-visual sync loss component
```

### Data Flow

```
Raw Video (person/video.mp4)
  ↓
[CompletePreprocessor]
  ├─ HeadPoseNormalizer → Face/Mouth crops (256×256, 512×512)
  ├─ GPUAudioProcessor → Mel spectrograms (80 bins, 25 fps)
  └─ ComprehensivePhonemeAligner + CompleteVisemeMapper → Viseme labels (21 classes)
  ↓
Processed PKL (sequences with face, mouth, mel, visemes)
  ↓
[LazyLipSyncDataset + DataLoader]
  ↓
Training Batch {face_crops, mouth_crops, mel_windows, visemes}
  ↓
[CompleteLipSyncModel]
  ├─ SparseTransformerAudioEncoder(mel_windows) → audio_features [B, T, 256]
  ├─ VisemeEmbedding(visemes) → viseme_emb [B, T, 256]
  ├─ VisemeFeatureEncoder(viseme_emb) → viseme_features [B, T, 512]
  ├─ CrossModalAttention(viseme_features, audio_features) → fused_features [B, T, 512]
  ├─ IdentityEncoder(face_crops[0]) → identity_features [B, 256]
  ├─ Concatenate(fused_features, audio_features) → combined [B, T, 768]
  ├─ MouthGenerator(combined, identity_features) → generated_mouths [B, T, 3, 256, 256]
  └─ TemporalSmoother(generated_mouths) → smooth_mouths [B, T, 3, 256, 256]
  ↓
[CompleteLoss]
  ├─ L1Loss(smooth_mouths, target_mouths)
  ├─ VGG16PerceptualLoss(smooth_mouths, target_mouths)
  ├─ SyncLoss(smooth_mouths, mel_windows) [Stage 2+]
  └─ WGANGPLoss(Discriminator(smooth_mouths)) [Stage 3]
  ↓
Optimizer Updates (Adam)
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**
- Reduce `batch_size`: Try 4, 2, or 1 per GPU
- Reduce `num_workers` in DataLoader
- Use smaller sequences: Reduce `max_seq_length` in preprocessing
- Note: Gradient checkpointing disabled with DataParallel

**Batch Size Limits (H100 80GB):**
- batch_size=8: **Stable** (10-12 GB per GPU)
- batch_size=12: **OOM** during perceptual loss computation
- batch_size=16: **OOM** during perceptual loss computation
- batch_size=24: **OOM** early in training

#### 2. Preprocessing CUDA Busy Errors

**Symptoms:**
```
CUDA error: CUDA-capable device(s) is/are busy or unavailable
```

**Cause:** Too many parallel workers trying to access GPUs

**Solutions:**
- Reduce `--workers` from 32 to 16 or 8
- Use `reprocess_missing_videos.py` with `--workers 16` for retries
- 16 workers typically achieves ~99% success rate

#### 3. MFA Installation Issues

**Symptoms:**
```
FileNotFoundError: mfa not found
```

**Solutions:**
```bash
# Install via conda
conda install -c conda-forge montreal-forced-aligner

# Or use --no_mfa flag for faster (but less accurate) preprocessing
python preprocess_hie.py --no_mfa ...
```

#### 4. Preprocessing Multiprocessing Error

**Symptoms:**
```
RuntimeError: Cannot re-initialize CUDA in forked subprocess
```

**Solution:** Already fixed in code with:
```python
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
```

#### 5. WandB Authentication Failed

**Symptoms:**
```
wandb: ERROR Error uploading
```

**Solutions:**
```bash
# Re-login
wandb login

# Or set API key
export WANDB_API_KEY=your_key
```

#### 6. Corrupted Preprocessed Files

**Symptoms:**
- EOFError when loading pickle files
- Training crashes with "invalid file" errors

**Solutions:**
```bash
# Verify all files
python verify_preprocessing.py \
  --processed_dir ./processed_data/train \
  --workers 32

# Delete corrupted source videos (if identified)
rm /path/to/corrupted_video.mp4

# Reprocess affected videos
python reprocess_missing_videos.py --workers 16
```

#### 7. Training Speed Issues

**Symptoms:**
- Training very slow (>10s per batch)
- GPU utilization at 0%

**Causes & Solutions:**
- **DataLoader bottleneck**: Increase `num_workers` to 16
- **Data loading from disk**: Ensure processed_data on fast SSD
- **GPU 0 bottleneck**: DataParallel limitation (GPU 0 handles scatter/gather)
- **Batch size too small**: Increase if memory allows

#### 8. Validation Loss Not Improving

**Symptoms:**
- Loss plateaus early
- FID not decreasing

**Solutions:**
- Verify perceptual loss warmup completed (epoch 10+)
- Check data quality: `ls processed_data/train/*.pkl | wc -l`
- Try lower learning rate: `--lr_g 5e-5 --lr_d 5e-5`
- Increase discriminator training: `--n_critic 10`
- Ensure all data is being used (check wandb logs for sequence counts)

---

## Performance Benchmarks

### Preprocessing Speed

| Setup | Videos/Hour | Time per Video | Notes |
|-------|-------------|----------------|-------|
| 32 workers, MFA, 8 GPUs | ~200 | ~18s | H100, full accuracy |
| 32 workers, no MFA, 8 GPUs | ~600 | ~6s | 3× faster |
| 16 workers, MFA, 1 GPU | ~100 | ~18s | Single GPU |

**Total Preprocessing Time (37,402 videos):**
- With MFA, 32 workers: ~187 hours (~8 days)
- Without MFA, 32 workers: ~62 hours (~2.6 days)

### Training Speed

| Setup | Samples/s | Time/Batch | Time/Epoch | Notes |
|-------|-----------|------------|------------|-------|
| 8× H100, batch=8 | 10.8 | 5.9s | 23.3h | DataParallel (actual) |
| 8× H100, batch=4 | 5.4 | 11.8s | 46.6h | Half batch size |

**Actual Performance (Current Setup):**
- Hardware: 8× H100 80GB HBM3
- Batch size: 8 per GPU (64 effective)
- Time per batch: ~5.9 seconds
- Batches per epoch: 14,228
- Time per epoch: ~23.3 hours
- **Total training time (100 epochs): ~97 days**

**GPU Utilization:**
- GPU 0: 58.5 GB / 81.6 GB (72%), 100% compute
- GPU 1-7: ~10.1 GB / 81.6 GB (12% each)
- Note: GPU 0 bottleneck due to DataParallel scatter/gather

**Epoch Breakdown by Stage:**
- Stage 1 (20 epochs): ~19.4 days
- Stage 2 (20 epochs): ~19.4 days
- Stage 3 (60 epochs): ~58.3 days

### Memory Usage

| Component | Memory (per GPU) |
|-----------|------------------|
| Model (G + D) | ~2.5 GB |
| Batch data (batch=8) | ~3 GB |
| Gradients & optimizer states | ~4 GB |
| CUDA overhead | ~1 GB |
| **Total (GPU 0)** | **~58 GB** |
| **Total (GPU 1-7)** | **~10 GB** |

---

## Current Training Status

### Overview

**Date**: 2025-10-27
**Status**: Training stopped manually after epoch 0
**Hardware**: 8× NVIDIA H100 80GB HBM3
**Checkpoint Directory**: `./gant580_checkpoints`

### Progress

```yaml
Completed:
  - Epoch 0: ✓ Completed (Stage 1: Coarse Training)
  - Validation loss: 23.26 (best so far)
  - Checkpoints saved:
      - best_model_coarse.pt
      - checkpoint_epoch0_coarse.pt
      - fid_real/ (100 real images for FID computation)

Current:
  - Epoch 1: Stopped at batch ~168/14,228 (~1.2%)
  - Current stage: Stage 1 (Coarse Training)
  - Generator loss: ~28-32 (fluctuating normally)

Remaining:
  - Stage 1: 18 more epochs (epochs 1-19)
  - Stage 2: 20 epochs (epochs 20-39)
  - Stage 3: 60 epochs (epochs 40-99)
  - Estimated time: ~95 days
```

### Dataset Statistics

```yaml
Source Data:
  - Location: /data/gpunet_admin/ffmpeg/transcoded_data/
  - Total videos: 37,402 MP4 files
  - Total duration: 625.17 hours (26.05 days)
  - Average duration: 60.17 seconds per video
  - Size: 258 GB
  - Persons: 658 folders (female1-327, male1-342)

Preprocessed Data:
  - Location: /data/gpunet_admin/processed_data/
  - Videos preprocessed: 35,105 (93.9% of source)
  - Missing/corrupted: 2,297 videos (6.1%)
  - Train sequences: 113,826
  - Val sequences: 15,604
  - Total sequences: 129,430
  - Split ratio: ~88% train / 12% val
  - Processed size: ~500 GB
```

### Model Configuration

```yaml
Model:
  - Generator: CompleteLipSyncModel (18.01M params)
  - Discriminator: AdaptiveWGANGPDiscriminator (24.87M params)
  - Total: 42.88M parameters (163.58 MB)
  - Sparse attention: Enabled (v3.1)
  - Gradient checkpointing: Disabled (DataParallel incompatible)

Training:
  - Batch size: 8 per GPU
  - Effective batch size: 64 (8 GPUs)
  - Num workers: 16 per GPU (128 total)
  - Learning rate (G): 1e-4
  - Learning rate (D): 1e-4
  - n_critic: 5
  - λ_gp (base): 10
  - Mixed precision: Enabled (AMP)

Hardware:
  - GPUs: 8× NVIDIA H100 80GB HBM3
  - Parallelization: DataParallel
  - GPU 0 memory: 58.5 GB / 81.6 GB (72%)
  - GPU 1-7 memory: ~10.1 GB / 81.6 GB (12% each)
```

### Performance Metrics

```yaml
Training Speed:
  - Time per batch: 5.9 seconds
  - Batches per epoch: 14,228
  - Time per epoch: 23.3 hours
  - Samples per second: 10.8
  - Estimated total time: 97 days

Current Metrics (Epoch 0):
  - Best validation loss: 23.26
  - Generator loss: ~28-32
  - Training proceeding normally

Wandb:
  - Project: lipsync-v31
  - Run: gant580-full-dataset
  - URL: https://wandb.ai/shreeshman/lipsync-v31
```

### Next Steps

To resume training:

```bash
python train.py \
  --data_dir /data/gpunet_admin/processed_data \
  --checkpoint_dir ./gant580_checkpoints \
  --batch_size 8 \
  --num_workers 16 \
  --use_wandb
```

The training will automatically resume from `checkpoint_epoch0_coarse.pt` and continue from Epoch 1.

---

## Advanced Topics

### Custom Data

To use your own video data:

1. **Organize data hierarchically** (person folders)
2. **Run check_struct.py** to verify structure
3. **Calculate duration** with `calculate_total_duration.py`
4. **Preprocess** with appropriate flags
5. **Verify completion** with `check_person_completion.py`
6. **Reprocess missing** with `reprocess_missing_videos.py` if needed

### Model Customization

**Change model size:**

```python
# In model.py
class CompleteLipSyncModel:
    def __init__(self, ...):
        # Increase capacity
        self.audio_encoder = SparseTransformerAudioEncoder(
            output_dim=512,  # Was 256
            ...
        )
```

**Add custom loss:**

```python
# In train.py CompleteLoss
def forward(self, ...):
    # Add your loss
    losses['custom'] = your_loss_function(...) * lambda_custom
    return losses
```

### Inference

Inference code is in `inference.py`. Basic usage:

```python
from model import CompleteLipSyncModel
import torch

# Load model
model = CompleteLipSyncModel(num_visemes=21, use_sparse_attention=True)
checkpoint = torch.load('gant580_checkpoints/best_model_full.pt')
model.load_state_dict(checkpoint['generator_state'])
model.eval()

# Prepare input batch
# ... (see inference.py for full implementation)

# Generate
with torch.no_grad():
    outputs = model(batch)
    generated_mouths = outputs['generated_mouths']
```

---

## Citation & References

### Papers & Techniques

This implementation combines ideas from:

- **WGAN-GP**: Gulrajani et al., "Improved Training of Wasserstein GANs" (2017)
- **Progressive GAN Training**: Karras et al., "Progressive Growing of GANs" (2018)
- **Sparse Attention**: Child et al., "Generating Long Sequences with Sparse Transformers" (2019)
- **FiLM Conditioning**: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer" (2018)
- **Wav2Lip**: Prajwal et al., "A Lip Sync Expert Is All You Need" (2020)
- **Montreal Forced Aligner**: McAuliffe et al. (2017)

### Architecture Inspiration

- Audio-visual sync: Inspired by SyncNet (Chung & Zisserman, 2016)
- Temporal smoothing: Optical flow warping from vid2vid (Wang et al., 2018)
- Multi-scale discrimination: From pix2pixHD (Wang et al., 2018)

---

## Quick Reference Commands

```bash
# Complete pipeline from scratch
python check_struct.py --data_dir ./transcoded_data
python calculate_total_duration.py --data_dir ./transcoded_data
python preprocess_smart_batch.py --data_dir ./transcoded_data --output_dir ./processed_data --workers 32
python check_person_completion.py --source_dir ./transcoded_data --processed_dir ./processed_data
python reprocess_missing_videos.py --source_dir ./transcoded_data --processed_dir ./processed_data --workers 16
python verify_preprocessing.py --processed_dir ./processed_data/train --workers 32
wandb login
python train.py --data_dir ./processed_data --checkpoint_dir ./checkpoints_v31 --batch_size 8 --use_wandb

# Resume training
python train.py --data_dir ./processed_data --checkpoint_dir ./checkpoints_v31 --use_wandb

# Check progress
ls -lht checkpoints_v31/*.pt | head  # Latest checkpoints
tail -f training.log                  # Training logs (if exists)
nvidia-smi                            # GPU usage
```

---

**Last Updated**: 2025-10-27
**Version**: 3.1
**Status**: Epoch 0 completed, training paused at epoch 1
**Total Documentation**: ~1,400 lines

For questions or issues, refer to the troubleshooting section or check WandB run logs at https://wandb.ai/shreeshman/lipsync-v31.
