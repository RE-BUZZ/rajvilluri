"""
Fast index builder for training data
"""
import pickle
from pathlib import Path
from tqdm import tqdm
import time

def build_index(data_dir, split='train'):
    """Build index quickly and save it"""
    data_dir = Path(data_dir)

    # Get all files
    all_files = list(data_dir.glob('**/*.pkl'))

    # Filter by split
    if split == 'train':
        files = [f for f in all_files if '/train/' in str(f)]
    else:
        files = [f for f in all_files if '/val/' in str(f)]

    print(f"\nBuilding {split} index from {len(files)} files...")
    print(f"Directory: {data_dir}")

    index = []
    failed_files = []

    start_time = time.time()

    for i, file_path in enumerate(tqdm(files, desc=f"Indexing {split}")):
        try:
            # Progress every 100 files
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(files) - i - 1) / rate if rate > 0 else 0
                print(f"\n  [{i+1}/{len(files)}] Rate: {rate:.1f} files/s | ETA: {remaining:.0f}s")

            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                num_sequences = len(data.get('sequences', []))

                for seq_idx in range(num_sequences):
                    index.append((file_path, seq_idx))

        except Exception as e:
            print(f"\n  ⚠ Failed to load {file_path.name}: {e}")
            failed_files.append((str(file_path), str(e)))
            continue

    elapsed = time.time() - start_time

    print(f"\n✓ Indexed {len(files)} files in {elapsed:.1f}s ({len(files)/elapsed:.1f} files/s)")
    print(f"  Total sequences: {len(index)}")
    print(f"  Failed files: {len(failed_files)}")

    if failed_files:
        print(f"\n⚠ Failed files:")
        for fpath, error in failed_files[:10]:  # Show first 10
            print(f"  - {Path(fpath).name}: {error}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")

    # Save index
    cache_file = data_dir / f'.index_cache_{split}.pkl'
    cache_data = {'files': files, 'index': index}

    print(f"\nSaving index to {cache_file}...")
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

    print(f"✓ Saved index cache ({len(index)} sequences)")

    return index, failed_files


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/data/gpunet_admin/processed_data', help='Data directory')
    parser.add_argument('--split', choices=['train', 'val', 'both'], default='both', help='Which split to index')
    args = parser.parse_args()

    print("="*60)
    print("Fast Index Builder")
    print("="*60)

    if args.split in ['train', 'both']:
        train_index, train_failed = build_index(args.data_dir, split='train')

    if args.split in ['val', 'both']:
        val_index, val_failed = build_index(args.data_dir, split='val')

    print("\n" + "="*60)
    print("Index building complete!")
    print("="*60)
