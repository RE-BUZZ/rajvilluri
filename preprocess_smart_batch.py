"""
Smart batched preprocessing - only processes unprocessed persons
Strategy: For each batch of persons, process train files then val files
Uses video-based split (90% train, 10% val from each person)
"""

import subprocess
import sys
from pathlib import Path
import time
import re

# Configuration
TRANSCODED_DIR = "/data/gpunet_admin/ffmpeg/transcoded_data"
OUTPUT_DIR = "/data/gpunet_admin/processed_data"
BATCH_SIZE = 100  # Process 100 persons per batch
WORKERS = 32  # Number of parallel workers (optimized for 8x H100)
GPUS = "0,1,2,3,4,5,6,7"  # Use all 8 H100 GPUs
VAL_RATIO = 0.1  # 10% validation

def get_processed_persons():
    """Get set of already processed persons"""
    train_dir = Path(OUTPUT_DIR) / "train"
    val_dir = Path(OUTPUT_DIR) / "val"

    processed = set()

    # Check train
    for f in train_dir.glob('*.pkl'):
        match = re.match(r'([^_]+)_', f.name)
        if match:
            processed.add(match.group(1))

    # Check val (should be same as train for video-based split)
    for f in val_dir.glob('*.pkl'):
        match = re.match(r'([^_]+)_', f.name)
        if match:
            processed.add(match.group(1))

    return processed

def get_all_persons():
    """Get all persons from transcoded data"""
    transcoded = Path(TRANSCODED_DIR)
    return set(d.name for d in transcoded.iterdir() if d.is_dir())

def main():
    # Get persons to process
    all_persons = get_all_persons()
    processed_persons = get_processed_persons()
    remaining_persons = sorted(all_persons - processed_persons)

    print("="*80)
    print("SMART BATCHED PREPROCESSING")
    print("="*80)
    print(f"Total persons: {len(all_persons)}")
    print(f"Already processed: {len(processed_persons)}")
    print(f"Remaining to process: {len(remaining_persons)}")
    print(f"Batch size: {BATCH_SIZE} persons")
    print(f"Number of batches: {(len(remaining_persons) + BATCH_SIZE - 1) // BATCH_SIZE}")
    print(f"Workers: {WORKERS}")
    print(f"GPUs: {GPUS}")
    print(f"Split: {int((1-VAL_RATIO)*100)}% train, {int(VAL_RATIO*100)}% val (video-based)")
    print("="*80)

    if len(remaining_persons) == 0:
        print("\n✓ All persons already processed!")
        return

    # Create output directories
    train_dir = Path(OUTPUT_DIR) / "train"
    val_dir = Path(OUTPUT_DIR) / "val"
    train_dir.mkdir(exist_ok=True, parents=True)
    val_dir.mkdir(exist_ok=True, parents=True)

    # Process in batches
    batch_num = 1
    for i in range(0, len(remaining_persons), BATCH_SIZE):
        batch_persons = remaining_persons[i:i + BATCH_SIZE]

        print(f"\n{'='*80}")
        print(f"BATCH {batch_num}: {len(batch_persons)} persons")
        print(f"{'='*80}")

        # STEP 1: Process TRAINING files for this batch (90% of videos)
        print(f"\n[BATCH {batch_num}] Processing TRAINING files (90% of videos)...")
        print("-"*80)

        train_cmd = [
            "env", f"CUDA_VISIBLE_DEVICES={GPUS}",
            "python", "preprocess_hie.py",
            "--data_dir", TRANSCODED_DIR,
            "--output", OUTPUT_DIR,
            "--persons", *batch_persons,
            "--auto_split",  # Enable automatic splitting
            "--split", "train",  # Process only training videos
            "--val_ratio", str(VAL_RATIO),  # 10% val ratio
            "--augment",  # Enable augmentation for training
            "--workers", str(WORKERS),
            "--fps", "25"
        ]

        start_time = time.time()
        result = subprocess.run(train_cmd, cwd="/data/gpunet_admin/gantcode")
        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"✓ Training files completed in {elapsed/60:.1f} min")
        else:
            print(f"✗ Training files FAILED! Continuing anyway...")

        # STEP 2: Process VALIDATION files for this batch (10% of videos)
        print(f"\n[BATCH {batch_num}] Processing VALIDATION files (10% of videos)...")
        print("-"*80)

        val_cmd = [
            "env", f"CUDA_VISIBLE_DEVICES={GPUS}",
            "python", "preprocess_hie.py",
            "--data_dir", TRANSCODED_DIR,
            "--output", OUTPUT_DIR,
            "--persons", *batch_persons,
            "--auto_split",  # Enable automatic splitting
            "--split", "val",  # Process only validation videos
            "--val_ratio", str(VAL_RATIO),  # Same 10% val ratio
            # NO --augment for validation
            "--workers", str(WORKERS),
            "--fps", "25"
        ]

        start_time = time.time()
        result = subprocess.run(val_cmd, cwd="/data/gpunet_admin/gantcode")
        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"✓ Validation files completed in {elapsed/60:.1f} min")
        else:
            print(f"✗ Validation files FAILED!")

        # Checkpoint summary
        print(f"\n{'='*80}")
        print(f"✓ BATCH {batch_num} CHECKPOINT COMPLETE")
        print(f"  Persons processed: {len(batch_persons)}")
        print(f"  Total progress: {min(i + BATCH_SIZE, len(remaining_persons))}/{len(remaining_persons)}")
        print(f"{'='*80}\n")

        batch_num += 1

    print("\n" + "="*80)
    print("ALL BATCHES COMPLETE!")
    print("="*80)
    print(f"Total new persons processed: {len(remaining_persons)}")
    print(f"Total persons now: {len(all_persons)}")
    print("="*80)

if __name__ == "__main__":
    main()
