# Adaptive Training Scripts

## Overview
These scripts automatically try optimal configurations and fall back gracefully on errors.

## Quick Start

### 1. Start Adaptive Training
```bash
cd /data/gpunet_admin/gantcode
./adaptive_train.sh
```

**What it does:**
1. Stops any existing training
2. Tries **AGGRESSIVE** config (batch_size=8, workers=16)
3. If OOM → tries **MEDIUM** config (batch_size=6, workers=12)
4. If OOM → falls back to **CURRENT** config (batch_size=4, workers=8)
5. Runs whichever config works for 120s to verify stability
6. Leaves training running in background

### 2. Monitor Training
```bash
./monitor_training.sh
```

Shows:
- Current PID and config
- GPU memory usage
- Latest training progress
- Which log file is active

### 3. View Live Logs
```bash
# Auto-detect active log
tail -f training_*.log

# Or specific config logs
tail -f training_bs8_w16.log      # Aggressive config
tail -f training_bs6_w12.log      # Medium config
tail -f training_bs4_w8_fallback.log  # Current config
```

## Expected Results

| Config | Batch Size | Workers | Speedup | Risk |
|--------|------------|---------|---------|------|
| Aggressive | 8 (64 total) | 16 | ~2.0x | May OOM |
| Medium | 6 (48 total) | 12 | ~1.95x | Low OOM risk |
| Current | 4 (32 total) | 8 | ~1.81x | Safe |

## Log Files

- `training_bs8_w16.log` - Aggressive attempt
- `training_bs6_w12.log` - Medium attempt  
- `training_bs4_w8_fallback.log` - Fallback attempt
- `training_113k_resumed.log` - Previous manual run

## Manual Control

### Stop training
```bash
pkill -SIGTERM -f "train.py"
```

### Start specific config manually
```bash
# Aggressive
python train.py --data_dir /data/gpunet_admin/processed_data \
  --checkpoint_dir /data/gpunet_admin/checkpoints_v31 \
  --batch_size 8 --num_workers 16 --use_wandb

# Medium  
python train.py --data_dir /data/gpunet_admin/processed_data \
  --checkpoint_dir /data/gpunet_admin/checkpoints_v31 \
  --batch_size 6 --num_workers 12 --use_wandb

# Current
python train.py --data_dir /data/gpunet_admin/processed_data \
  --checkpoint_dir /data/gpunet_admin/checkpoints_v31 \
  --batch_size 4 --num_workers 8 --use_wandb
```

## Troubleshooting

### All configs fail
1. Check logs: `cat training_bs*.log | grep -i error`
2. Check GPU memory: `nvidia-smi`
3. Check disk space: `df -h /data`
4. Try reducing batch size to 2: 
   ```bash
   python train.py --batch_size 2 --num_workers 4 ...
   ```

### Training speed not improving
- Check if NVLink is active: `nvidia-smi nvlink --status`
- Check CPU bottleneck: `htop`
- Check disk I/O: `iostat -x 1`

### Want to restore 303K dataset
```bash
cp /data/gpunet_admin/processed_data/.index_cache_train_303k_backup.pkl \
   /data/gpunet_admin/processed_data/.index_cache_train.pkl
```

## Time Estimates

With **113,826 sequences** and **28,456 steps/epoch**:

| Config | Speed | Time/Epoch |
|--------|-------|------------|
| Aggressive (bs=8) | ~60 samples/s | ~3.2 hours |
| Medium (bs=6) | ~55 samples/s | ~3.5 hours |
| Current (bs=4) | ~50 samples/s | ~3.8 hours |

You save ~0.6 hours per epoch with aggressive config!
