"""
Hierarchical Data Preprocessing Script
For data structure: transcoded_data/person_folders/videos

Optimized for H100 GPU with parallel processing
"""

import sys
from pathlib import Path
import argparse
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import logging

# CRITICAL: Set multiprocessing start method to 'spawn' for CUDA compatibility
# Must be done before any CUDA operations
mp.set_start_method('spawn', force=True)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing import CompletePreprocessor, PreprocessConfig


def setup_logging(output_dir: Path):
    """Setup logging to file and console"""
    log_file = output_dir / 'preprocessing.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def process_single_video(args):
    """
    Process a single video (designed for multiprocessing)

    Args:
        args: Tuple of (video_path, output_dir, config_dict, person_name, apply_augmentation, gpu_id)

    Returns:
        Tuple of (success, video_path, message, num_sequences)
    """
    video_path, output_dir, config_dict, person_name, apply_augmentation, gpu_id = args

    try:
        # Create preprocessor (each process gets its own)
        # Use the per-video apply_augmentation flag and assigned GPU
        config = PreprocessConfig(
            fps=config_dict['fps'],
            use_mfa=config_dict['use_mfa'],
            apply_augmentation=apply_augmentation,  # Use the per-split flag
            vad_aggressiveness=config_dict['vad_aggressiveness'],
            use_gpu=config_dict.get('use_gpu', True),  # Enable GPU by default
            gpu_id=gpu_id  # Assign specific GPU
        )
        preprocessor = CompletePreprocessor(config)
        
        # Process video (augmentation is configured in PreprocessConfig)
        result = preprocessor.process_video(
            video_path=str(video_path)
        )
        
        # Save processed data with person name in filename
        video_name = video_path.stem
        output_file = output_dir / f"{person_name}_{video_name}_processed.pkl"
        
        with open(output_file, 'wb') as f:
            pickle.dump(result, f)
        
        num_sequences = len(result['sequences'])
        
        return (True, str(video_path), f"Success: {num_sequences} sequences", num_sequences)
        
    except Exception as e:
        return (False, str(video_path), f"Error: {str(e)}", 0)


def scan_data_directory(data_dir: Path, pattern: str = "*.mp4"):
    """
    Scan hierarchical directory structure
    
    Expected structure:
        transcoded_data/
            female1/
                female1-a.mp4
                female1-b.mp4
            male1/
                male1-a.mp4
                male1-b.mp4
    
    Returns:
        List of tuples: (video_path, person_name)
    """
    video_list = []
    
    # Find all person folders
    person_folders = [d for d in data_dir.iterdir() if d.is_dir()]
    
    print(f"\nScanning directory structure...")
    print(f"Base directory: {data_dir}")
    print(f"Found {len(person_folders)} person folders:")
    
    for person_folder in sorted(person_folders):
        # Find all videos in this person folder
        videos = list(person_folder.glob(pattern))
        
        if videos:
            print(f"  - {person_folder.name}: {len(videos)} videos")
            for video in videos:
                video_list.append((video, person_folder.name))
        else:
            print(f"  - {person_folder.name}: No videos found")
    
    return video_list


def create_train_val_split(video_list, val_ratio=0.1, by_person=True):
    """
    Create train/validation split
    
    Args:
        video_list: List of (video_path, person_name) tuples
        val_ratio: Ratio of validation data (default: 0.1 = 10%)
        by_person: If True, split by person (more realistic), else split randomly
    
    Returns:
        train_list, val_list
    """
    from collections import defaultdict
    import random
    
    if by_person:
        # Group by person
        person_videos = defaultdict(list)
        for video_path, person_name in video_list:
            person_videos[person_name].append((video_path, person_name))
        
        # For each person, split their videos
        train_list = []
        val_list = []
        
        for person_name, videos in person_videos.items():
            random.shuffle(videos)
            split_idx = int(len(videos) * (1 - val_ratio))
            
            train_list.extend(videos[:split_idx])
            val_list.extend(videos[split_idx:])
        
        print(f"\nTrain/Val split by person:")
        print(f"  Train: {len(train_list)} videos from {len(person_videos)} persons")
        print(f"  Val:   {len(val_list)} videos from {len(person_videos)} persons")
    
    else:
        # Random split
        random.shuffle(video_list)
        split_idx = int(len(video_list) * (1 - val_ratio))
        train_list = video_list[:split_idx]
        val_list = video_list[split_idx:]
        
        print(f"\nRandom train/val split:")
        print(f"  Train: {len(train_list)} videos")
        print(f"  Val:   {len(val_list)} videos")
    
    return train_list, val_list


def process_batch(video_batch, output_dir, config_dict, apply_augmentation, max_workers=None, num_gpus=None):
    """
    Process a batch of videos using multiprocessing with multi-GPU distribution

    Args:
        video_batch: List of (video_path, person_name) tuples
        output_dir: Output directory
        config_dict: Config dictionary
        apply_augmentation: Whether to apply augmentation
        max_workers: Number of parallel workers (None = auto)
        num_gpus: Number of GPUs to distribute across (None = auto-detect)
    """
    # Auto-detect number of GPUs if not specified
    if num_gpus is None:
        import torch
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Prepare arguments for each video with GPU assignment
    # Distribute workers evenly across all GPUs
    args_list = []
    for idx, (video_path, person_name) in enumerate(video_batch):
        gpu_id = idx % num_gpus  # Round-robin GPU assignment
        args_list.append((video_path, output_dir, config_dict, person_name, apply_augmentation, gpu_id))
    
    # Determine number of workers
    if max_workers is None:
        # Use 80% of available CPUs (leave some for system)
        max_workers = max(1, int(mp.cpu_count() * 0.8))
    
    print(f"\nProcessing {len(video_batch)} videos with {max_workers} workers...")
    
    results = []
    success_count = 0
    fail_count = 0
    total_sequences = 0
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_video, args): args[0] for args in args_list}
        
        # Process results as they complete
        with tqdm(total=len(futures), desc="Processing videos") as pbar:
            for future in as_completed(futures):
                video_path = futures[future]
                try:
                    success, path, message, num_seqs = future.result()
                    
                    if success:
                        success_count += 1
                        total_sequences += num_seqs
                    else:
                        fail_count += 1
                        logging.error(f"Failed: {path} - {message}")
                    
                    results.append((success, path, message, num_seqs))
                    
                except Exception as e:
                    fail_count += 1
                    logging.error(f"Exception processing {video_path}: {e}")
                
                pbar.update(1)
                pbar.set_postfix({
                    'success': success_count,
                    'failed': fail_count,
                    'sequences': total_sequences
                })
    
    return results, success_count, fail_count, total_sequences


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess hierarchical video dataset (optimized for H100)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Directory Structure Expected:
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

Examples:
  # Basic preprocessing (train data with augmentation)
  python scripts/preprocess_hierarchical.py \\
    --data_dir ./transcoded_data \\
    --output ./processed_data/train \\
    --augment

  # Validation data (no augmentation)
  python scripts/preprocess_hierarchical.py \\
    --data_dir ./transcoded_data \\
    --output ./processed_data/val \\
    --split val

  # Automatic train/val split
  python scripts/preprocess_hierarchical.py \\
    --data_dir ./transcoded_data \\
    --output ./processed_data \\
    --auto_split \\
    --val_ratio 0.1

  # Fast mode (disable MFA for speed testing)
  python scripts/preprocess_hierarchical.py \\
    --data_dir ./transcoded_data \\
    --output ./processed_data/train \\
    --no_mfa \\
    --workers 32

  # Process specific persons only
  python scripts/preprocess_hierarchical.py \\
    --data_dir ./transcoded_data \\
    --output ./processed_data/train \\
    --persons female1 female2 male1
        """
    )
    
    # Input/Output
    parser.add_argument('--data_dir', required=True, 
                       help='Root directory with person folders')
    parser.add_argument('--output', required=True,
                       help='Output directory for processed data')
    
    # Data selection
    parser.add_argument('--pattern', default='*.mp4',
                       help='File pattern for videos (default: *.mp4)')
    parser.add_argument('--persons', nargs='+', default=None,
                       help='Process only specific persons (folder names)')
    parser.add_argument('--max_videos', type=int, default=None,
                       help='Maximum number of videos to process')
    
    # Train/Val split
    parser.add_argument('--auto_split', action='store_true',
                       help='Automatically split into train/val')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation ratio if auto_split (default: 0.1)')
    parser.add_argument('--split', choices=['train', 'val', 'all'], default='all',
                       help='Which split to process (requires auto_split)')
    parser.add_argument('--split_by_person', action='store_true', default=True,
                       help='Split by person (more realistic eval)')
    
    # Processing options
    parser.add_argument('--fps', type=int, default=25,
                       help='Target FPS (default: 25)')
    parser.add_argument('--no_mfa', action='store_true',
                       help='Disable Montreal Forced Aligner (faster but less accurate)')
    parser.add_argument('--augment', action='store_true',
                       help='Enable SpecAugment (for training data only)')
    parser.add_argument('--vad_aggressiveness', type=int, default=2, choices=[0,1,2,3],
                       help='VAD aggressiveness (0=lenient, 3=aggressive, default: 2)')
    
    # Performance options
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: auto = 80%% of CPUs)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Process videos in batches (for memory management)')
    
    # Debugging
    parser.add_argument('--dry_run', action='store_true',
                       help='Scan and show what would be processed without actually processing')
    
    args = parser.parse_args()
    
    print("="*80)
    print("HIERARCHICAL VIDEO PREPROCESSING (H100 Optimized)")
    print("="*80)
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return 1
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting preprocessing: {data_dir} -> {output_dir}")
    
    # Scan directory structure
    video_list = scan_data_directory(data_dir, args.pattern)
    
    # Filter by specific persons if requested
    if args.persons:
        video_list = [(v, p) for v, p in video_list if p in args.persons]
        print(f"\nFiltered to {len(video_list)} videos from persons: {args.persons}")
    
    # Limit number of videos if requested
    if args.max_videos and len(video_list) > args.max_videos:
        import random
        random.shuffle(video_list)
        video_list = video_list[:args.max_videos]
        print(f"\nLimited to {args.max_videos} random videos")
    
    if not video_list:
        print("ERROR: No videos found!")
        return 1
    
    print(f"\nTotal videos to process: {len(video_list)}")
    
    # Auto train/val split
    if args.auto_split:
        train_list, val_list = create_train_val_split(
            video_list, 
            val_ratio=args.val_ratio,
            by_person=args.split_by_person
        )
        
        # Determine which split to process
        if args.split == 'train':
            video_list = train_list
            output_dir = output_dir / 'train'
            apply_augmentation = args.augment
        elif args.split == 'val':
            video_list = val_list
            output_dir = output_dir / 'val'
            apply_augmentation = False  # Never augment validation
        else:  # 'all'
            # Process both splits
            output_dir.mkdir(exist_ok=True)
            train_dir = output_dir / 'train'
            val_dir = output_dir / 'val'
            train_dir.mkdir(exist_ok=True)
            val_dir.mkdir(exist_ok=True)
    else:
        apply_augmentation = args.augment
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Dry run - just show what would be processed
    if args.dry_run:
        print("\n" + "="*80)
        print("DRY RUN - Videos that would be processed:")
        print("="*80)
        for video_path, person_name in video_list:
            print(f"  {person_name:15s} | {video_path.name}")
        print("="*80)
        print(f"Total: {len(video_list)} videos")
        return 0
    
    # Create config dict for multiprocessing
    config_dict = {
        'fps': args.fps,
        'use_mfa': not args.no_mfa,
        'spec_augment': args.augment,
        'vad_aggressiveness': args.vad_aggressiveness,
        'use_gpu': True  # Enable GPU acceleration
    }
    
    print("\nProcessing Configuration:")
    print(f"  Output directory: {output_dir}")
    print(f"  FPS: {args.fps}")
    print(f"  MFA: {'enabled' if not args.no_mfa else 'disabled'}")
    print(f"  SpecAugment: {'enabled (train only)' if args.augment else 'disabled'}")
    print(f"  VAD aggressiveness: {args.vad_aggressiveness}")
    print(f"  Workers: {args.workers if args.workers else 'auto'}")
    
    # Process all videos or in batches
    if args.auto_split and args.split == 'all':
        # Process train and val separately
        print("\n" + "="*80)
        print("PROCESSING TRAINING DATA")
        print("="*80)
        train_results, train_success, train_fail, train_seqs = process_batch(
            train_list, train_dir, config_dict, 
            apply_augmentation=args.augment, 
            max_workers=args.workers
        )
        
        print("\n" + "="*80)
        print("PROCESSING VALIDATION DATA")
        print("="*80)
        val_results, val_success, val_fail, val_seqs = process_batch(
            val_list, val_dir, config_dict,
            apply_augmentation=False,  # No augmentation for val
            max_workers=args.workers
        )
        
        # Combined statistics
        total_success = train_success + val_success
        total_fail = train_fail + val_fail
        total_seqs = train_seqs + val_seqs
        
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        print(f"Training:")
        print(f"  Successful: {train_success} videos")
        print(f"  Failed: {train_fail} videos")
        print(f"  Sequences: {train_seqs}")
        print(f"  Output: {train_dir}")
        print(f"\nValidation:")
        print(f"  Successful: {val_success} videos")
        print(f"  Failed: {val_fail} videos")
        print(f"  Sequences: {val_seqs}")
        print(f"  Output: {val_dir}")
        print(f"\nTotal:")
        print(f"  Successful: {total_success}/{len(video_list)} videos ({total_success/len(video_list)*100:.1f}%)")
        print(f"  Total sequences: {total_seqs}")
    
    else:
        # Process single batch
        results, success_count, fail_count, total_sequences = process_batch(
            video_list, output_dir, config_dict,
            apply_augmentation=apply_augmentation,
            max_workers=args.workers
        )
        
        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print("="*80)
        print(f"Successful: {success_count}/{len(video_list)} videos ({success_count/len(video_list)*100:.1f}%)")
        print(f"Failed: {fail_count} videos")
        print(f"Total sequences: {total_sequences}")
        print(f"Output directory: {output_dir}")
    
    # Save processing summary
    summary_file = output_dir / 'preprocessing_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("Preprocessing Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data directory: {data_dir}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Total videos processed: {len(video_list)}\n")
        f.write(f"Successful: {success_count}\n")
        f.write(f"Failed: {fail_count}\n")
        f.write(f"Total sequences: {total_sequences}\n\n")
        f.write("Configuration:\n")
        f.write(f"  FPS: {args.fps}\n")
        f.write(f"  MFA: {'enabled' if not args.no_mfa else 'disabled'}\n")
        f.write(f"  SpecAugment: {'enabled' if apply_augmentation else 'disabled'}\n")
        f.write(f"  VAD aggressiveness: {args.vad_aggressiveness}\n")
    
    print(f"\nSummary saved to: {summary_file}")
    print(f"Log file: {output_dir / 'preprocessing.log'}")
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
