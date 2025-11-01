# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains scripts for preprocessing hierarchical video datasets for audio-visual speech processing. The main use case is preparing video data (organized by person/speaker) for training audio-visual models, optimized for H100 GPU environments.

## Directory Structure

Expected data structure for processing:
```
transcoded_data/
  ├── female1/
  │   ├── female1-a.mp4
  │   ├── female1-b.mp4
  │   └── ...
  ├── female2/
  │   └── ...
  ├── male1/
  │   └── ...
  └── ...
```

Output structure:
```
processed_data/
  ├── train/
  │   └── {person}_{video}_processed.pkl
  └── val/
      └── {person}_{video}_processed.pkl
```

## Key Scripts

### 1. check_struct.py
Data structure verification script - run this FIRST before preprocessing.

**Purpose**: Validates the data directory structure, counts videos, estimates processing time, and provides recommended preprocessing commands.

**Usage**:
```bash
python check_struct.py --data_dir ./transcoded_data
```

**What it does**:
- Scans person folders and counts videos by format
- Reports total size and video counts
- Estimates preprocessing time based on worker count
- Provides recommendations for preprocessing commands based on dataset size

### 2. preprocess_hie.py
Main hierarchical video preprocessing script optimized for parallel processing.

**Dependencies**: Imports from a `preprocessing` module (CompletePreprocessor, PreprocessConfig) which is expected to be in the parent directory of the script.

**Key features**:
- Multi-process parallel video processing
- Automatic train/validation splitting (by person or random)
- Support for Montreal Forced Aligner (MFA) for audio alignment
- SpecAugment for data augmentation
- VAD (Voice Activity Detection) configuration

**Common commands**:

Basic preprocessing with augmentation:
```bash
python preprocess_hie.py \
  --data_dir ./transcoded_data \
  --output ./processed_data/train \
  --augment
```

Automatic train/val split:
```bash
python preprocess_hie.py \
  --data_dir ./transcoded_data \
  --output ./processed_data \
  --auto_split \
  --val_ratio 0.1 \
  --augment
```

Fast mode without MFA:
```bash
python preprocess_hie.py \
  --data_dir ./transcoded_data \
  --output ./processed_data/train \
  --no_mfa \
  --workers 32
```

Process specific persons only:
```bash
python preprocess_hie.py \
  --data_dir ./transcoded_data \
  --output ./processed_data/train \
  --persons female1 female2 male1
```

Dry run (preview what will be processed):
```bash
python preprocess_hie.py \
  --data_dir ./transcoded_data \
  --output ./processed_data \
  --dry_run
```

## Architecture Notes

### Processing Pipeline (preprocess_hie.py)

1. **Data Discovery** (`scan_data_directory`): Scans hierarchical folder structure to find all videos matching a pattern
2. **Train/Val Split** (`create_train_val_split`): Creates splits either by person (more realistic) or randomly
3. **Parallel Processing** (`process_batch`): Uses ProcessPoolExecutor for parallel video processing
4. **Single Video Processing** (`process_single_video`):
   - Creates a CompletePreprocessor instance per process
   - Processes video with specified config (FPS, MFA, augmentation, VAD)
   - Saves output as pickle file with naming: `{person_name}_{video_name}_processed.pkl`

### Key Parameters

- `--fps`: Target frames per second (default: 25)
- `--use_mfa` / `--no_mfa`: Enable/disable Montreal Forced Aligner (MFA disabled = faster but less accurate)
- `--augment`: Enable SpecAugment for training data augmentation
- `--vad_aggressiveness`: Voice Activity Detection sensitivity (0-3, default: 2)
- `--workers`: Number of parallel workers (default: 80% of available CPUs)
- `--auto_split`: Automatically create train/val splits
- `--split_by_person`: Split by person for more realistic evaluation (default: true)
- `--pattern`: Video file pattern (default: "*.mp4")

### Output Files

Each processed video generates:
- `{person}_{video}_processed.pkl`: Pickle file containing processed sequences
- `preprocessing.log`: Processing log
- `preprocessing_summary.txt`: Summary of processing results

### Missing Dependencies

The `preprocessing` module (imported in preprocess_hie.py:20) is expected to provide:
- `CompletePreprocessor`: Main preprocessing class
- `PreprocessConfig`: Configuration dataclass/class

This module should be in the parent directory but is not present in the current repository.

## Workflow

1. **Verify data structure**: Run `check_struct.py` to validate your data
2. **Review recommendations**: The script provides optimized commands based on dataset size
3. **Run preprocessing**: Use `preprocess_hie.py` with recommended settings
4. **Check outputs**: Review logs and summary files for any failures

## Performance Considerations

- Default worker count is 80% of available CPUs
- MFA alignment is the slowest part (~18s per video with MFA, ~6s without)
- For H100 environments, high parallelization (32+ workers) is recommended for large datasets
- SpecAugment should only be enabled for training data, not validation
