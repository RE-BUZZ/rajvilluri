#!/usr/bin/env python3
"""
Verification script to check preprocessed data integrity
"""
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import sys

def test_pickle_file(pkl_path):
    """Test if a pickle file can be loaded and has valid data"""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # Basic validation
        if not isinstance(data, dict):
            return (str(pkl_path), False, "Not a dict")

        if 'sequences' not in data:
            return (str(pkl_path), False, "Missing 'sequences' key")

        if not isinstance(data['sequences'], list):
            return (str(pkl_path), False, "'sequences' not a list")

        if len(data['sequences']) == 0:
            return (str(pkl_path), False, "Empty sequences")

        return (str(pkl_path), True, f"{len(data['sequences'])} sequences")

    except Exception as e:
        return (str(pkl_path), False, f"Error: {str(e)[:100]}")

def main():
    train_dir = Path("/data/gpunet_admin/processed_data/train")
    val_dir = Path("/data/gpunet_admin/processed_data/val")

    # Collect all pickle files
    print("Collecting pickle files...")
    train_files = list(train_dir.glob("*.pkl"))
    val_files = list(val_dir.glob("*.pkl"))
    all_files = train_files + val_files

    print(f"Found {len(train_files)} train files")
    print(f"Found {len(val_files)} val files")
    print(f"Total: {len(all_files)} files")
    print(f"\nTesting files for corruption...")

    # Test files in parallel
    corrupt_files = []
    valid_files = 0
    empty_files = []

    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(test_pickle_file, f): f for f in all_files}

        for future in tqdm(as_completed(futures), total=len(all_files)):
            filepath, is_valid, message = future.result()

            if not is_valid:
                corrupt_files.append((filepath, message))
                if "Empty sequences" in message:
                    empty_files.append(filepath)
            else:
                valid_files += 1

    # Print results
    print("\n" + "="*80)
    print("VERIFICATION RESULTS")
    print("="*80)
    print(f"Total files tested: {len(all_files)}")
    print(f"Valid files: {valid_files}")
    print(f"Corrupt/Invalid files: {len(corrupt_files)}")
    print(f"Empty sequence files: {len(empty_files)}")

    if corrupt_files:
        print("\n" + "="*80)
        print("CORRUPT/INVALID FILES:")
        print("="*80)
        for filepath, message in corrupt_files[:50]:  # Show first 50
            print(f"{Path(filepath).name}: {message}")

        if len(corrupt_files) > 50:
            print(f"\n... and {len(corrupt_files) - 50} more")

    if empty_files:
        print("\n" + "="*80)
        print(f"FILES WITH EMPTY SEQUENCES ({len(empty_files)}):")
        print("="*80)
        for filepath in empty_files[:20]:
            print(Path(filepath).name)
        if len(empty_files) > 20:
            print(f"... and {len(empty_files) - 20} more")

    print("\n" + "="*80)
    if corrupt_files:
        print("⚠ ISSUES FOUND - Some files need reprocessing")
        sys.exit(1)
    else:
        print("✓ ALL FILES VALID")
        sys.exit(0)

if __name__ == "__main__":
    main()
