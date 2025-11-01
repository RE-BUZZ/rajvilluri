#!/usr/bin/env python3
"""
Find missing/failed videos that need reprocessing
"""
from pathlib import Path
import re

def main():
    transcoded_dir = Path("/data/gpunet_admin/ffmpeg/transcoded_data")
    train_dir = Path("/data/gpunet_admin/processed_data/train")
    val_dir = Path("/data/gpunet_admin/processed_data/val")

    # Get all source videos
    print("Scanning source videos...")
    source_videos = {}
    for person_dir in transcoded_dir.iterdir():
        if person_dir.is_dir():
            for video in person_dir.glob("*.mp4"):
                key = f"{person_dir.name}_{video.stem}"
                source_videos[key] = str(video)

    print(f"Found {len(source_videos)} source videos")

    # Get all processed files
    print("Scanning processed files...")
    processed_keys = set()
    for pkl_file in train_dir.glob("*.pkl"):
        match = re.match(r'(.+)_processed\.pkl$', pkl_file.name)
        if match:
            processed_keys.add(match.group(1))

    for pkl_file in val_dir.glob("*.pkl"):
        match = re.match(r'(.+)_processed\.pkl$', pkl_file.name)
        if match:
            processed_keys.add(match.group(1))

    print(f"Found {len(processed_keys)} processed files")

    # Find missing
    missing_keys = set(source_videos.keys()) - processed_keys

    # Load empty sequence files
    empty_files = []
    with open("empty_sequences.txt", "w") as f:
        # Collect empty files from verification
        for pkl_file in list(train_dir.glob("*.pkl")) + list(val_dir.glob("*.pkl")):
            import pickle
            try:
                with open(pkl_file, 'rb') as pf:
                    data = pickle.load(pf)
                    if 'sequences' in data and len(data['sequences']) == 0:
                        match = re.match(r'(.+)_processed\.pkl$', pkl_file.name)
                        if match:
                            key = match.group(1)
                            empty_files.append(key)
                            f.write(f"{key}\n")
            except:
                pass

    print(f"Found {len(empty_files)} empty sequence files")

    # Combine missing and empty
    all_need_reprocessing = missing_keys.union(set(empty_files))

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total source videos: {len(source_videos)}")
    print(f"Successfully processed: {len(processed_keys) - len(empty_files)}")
    print(f"Missing processed files: {len(missing_keys)}")
    print(f"Empty sequences (failed): {len(empty_files)}")
    print(f"Total needing reprocessing: {len(all_need_reprocessing)}")
    print("="*80)

    # Write list of videos to reprocess
    with open("videos_to_reprocess.txt", "w") as f:
        for key in sorted(all_need_reprocessing):
            if key in source_videos:
                f.write(f"{source_videos[key]}\n")

    print(f"\nWritten list to: videos_to_reprocess.txt")
    print(f"Total videos to reprocess: {len(all_need_reprocessing)}")

if __name__ == "__main__":
    main()
