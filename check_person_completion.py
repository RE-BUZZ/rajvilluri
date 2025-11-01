#!/usr/bin/env python3
"""
Check completion status per person
"""
from pathlib import Path
import re
from collections import defaultdict

def main():
    transcoded_dir = Path("/data/gpunet_admin/ffmpeg/transcoded_data")
    train_dir = Path("/data/gpunet_admin/processed_data/train")
    val_dir = Path("/data/gpunet_admin/processed_data/val")

    print("Scanning source videos by person...")
    source_by_person = defaultdict(list)
    for person_dir in transcoded_dir.iterdir():
        if person_dir.is_dir():
            for video in person_dir.glob("*.mp4"):
                key = f"{person_dir.name}_{video.stem}"
                source_by_person[person_dir.name].append(key)

    print(f"Found {len(source_by_person)} persons")

    # Get processed videos by person
    print("Scanning processed files by person...")
    processed_by_person = defaultdict(set)

    for pkl_file in train_dir.glob("*.pkl"):
        match = re.match(r'([^_]+)_(.+)_processed\.pkl$', pkl_file.name)
        if match:
            person = match.group(1)
            video_key = f"{person}_{match.group(2)}"
            processed_by_person[person].add(video_key)

    for pkl_file in val_dir.glob("*.pkl"):
        match = re.match(r'([^_]+)_(.+)_processed\.pkl$', pkl_file.name)
        if match:
            person = match.group(1)
            video_key = f"{person}_{match.group(2)}"
            processed_by_person[person].add(video_key)

    # Analyze completion
    fully_complete = []
    partially_complete = []
    not_started = []

    for person, source_videos in sorted(source_by_person.items()):
        total = len(source_videos)
        processed = len(processed_by_person[person])
        missing = total - processed

        if missing == 0:
            fully_complete.append((person, total, processed))
        elif processed == 0:
            not_started.append((person, total))
        else:
            partially_complete.append((person, total, processed, missing))

    # Print results
    print("\n" + "="*80)
    print("PERSON-LEVEL COMPLETION ANALYSIS")
    print("="*80)
    print(f"Total persons: {len(source_by_person)}")
    print(f"Fully complete: {len(fully_complete)}")
    print(f"Partially complete: {len(partially_complete)}")
    print(f"Not started: {len(not_started)}")
    print("="*80)

    if partially_complete:
        print(f"\nPARTIALLY COMPLETE PERSONS ({len(partially_complete)}):")
        print("-"*80)
        print(f"{'Person':<15} {'Total':<8} {'Done':<8} {'Missing':<8} {'%Done':<8}")
        print("-"*80)
        for person, total, processed, missing in sorted(partially_complete, key=lambda x: x[3], reverse=True)[:30]:
            pct = (processed / total * 100)
            print(f"{person:<15} {total:<8} {processed:<8} {missing:<8} {pct:>6.1f}%")

        if len(partially_complete) > 30:
            print(f"... and {len(partially_complete) - 30} more")

    if not_started:
        print(f"\nNOT STARTED PERSONS ({len(not_started)}):")
        print("-"*80)
        for person, total in not_started[:20]:
            print(f"{person}: {total} videos")
        if len(not_started) > 20:
            print(f"... and {len(not_started) - 20} more")

    # Summary stats
    total_source = sum(len(v) for v in source_by_person.values())
    total_processed = sum(len(v) for v in processed_by_person.values())
    total_missing = total_source - total_processed

    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print(f"Total videos in source: {total_source}")
    print(f"Total videos processed: {total_processed}")
    print(f"Total videos missing: {total_missing}")
    print(f"Overall completion: {total_processed/total_source*100:.1f}%")
    print("="*80)

    # Write partially complete persons to file
    with open("partially_complete_persons.txt", "w") as f:
        for person, total, processed, missing in partially_complete:
            f.write(f"{person}\n")

    print(f"\nWritten {len(partially_complete)} partially complete persons to: partially_complete_persons.txt")

if __name__ == "__main__":
    main()
