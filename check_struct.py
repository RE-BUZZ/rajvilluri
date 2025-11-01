"""
Quick Data Structure Verification Script
Run this first to check your transcoded_data folder
"""

import sys
from pathlib import Path
from collections import defaultdict


def analyze_data_structure(data_dir):
    """Analyze the hierarchical data structure"""
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ ERROR: Directory not found: {data_dir}")
        return False
    
    print("="*80)
    print("DATA STRUCTURE ANALYSIS")
    print("="*80)
    print(f"Directory: {data_path.absolute()}\n")
    
    # Find person folders
    person_folders = [d for d in data_path.iterdir() if d.is_dir()]
    
    if not person_folders:
        print("❌ ERROR: No subdirectories found!")
        print("Expected structure:")
        print("  transcoded_data/")
        print("    ├── female1/")
        print("    ├── female2/")
        print("    └── male1/")
        return False
    
    print(f"✅ Found {len(person_folders)} person folders:\n")
    
    # Analyze each person folder
    video_stats = defaultdict(lambda: {'mp4': 0, 'avi': 0, 'mkv': 0, 'other': 0, 'total': 0})
    total_videos = 0
    total_size_gb = 0
    
    for person_folder in sorted(person_folders):
        print(f"📁 {person_folder.name}/")
        
        # Find all video files
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.flv'}
        videos = [f for f in person_folder.iterdir() 
                 if f.is_file() and f.suffix.lower() in video_extensions]
        
        if not videos:
            print(f"   ⚠️  No videos found")
            continue
        
        # Count by extension
        ext_counts = defaultdict(int)
        folder_size = 0
        for video in videos:
            ext = video.suffix.lower().replace('.', '')
            ext_counts[ext] += 1
            folder_size += video.stat().st_size
        
        total_videos += len(videos)
        total_size_gb += folder_size / (1024**3)
        
        print(f"   Videos: {len(videos)}")
        for ext, count in sorted(ext_counts.items()):
            print(f"     - {ext.upper()}: {count} files")
            video_stats[person_folder.name][ext] = count
            video_stats[person_folder.name]['total'] += count
        
        print(f"   Size: {folder_size / (1024**3):.2f} GB")
        
        # Show first few filenames as examples
        print(f"   Examples:")
        for video in sorted(videos)[:3]:
            print(f"     - {video.name}")
        if len(videos) > 3:
            print(f"     ... and {len(videos) - 3} more")
        print()
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total person folders: {len(person_folders)}")
    print(f"Total videos: {total_videos}")
    print(f"Total size: {total_size_gb:.2f} GB")
    
    # Check video formats
    all_extensions = set()
    for stats in video_stats.values():
        all_extensions.update(k for k in stats.keys() if k != 'total')
    
    if all_extensions:
        print(f"\nVideo formats found: {', '.join(sorted(all_extensions))}")
        
        if 'mp4' not in all_extensions:
            print("\n⚠️  WARNING: No .mp4 files found!")
            print("You'll need to specify the pattern:")
            if 'avi' in all_extensions:
                print('  --pattern "*.avi"')
            elif 'mkv' in all_extensions:
                print('  --pattern "*.mkv"')
    
    # Estimate processing time
    print(f"\n📊 ESTIMATED PREPROCESSING TIME (with MFA):")
    print(f"  With 32 workers: {total_videos * 18 / 3600:.1f} hours")
    print(f"  With 16 workers: {total_videos * 18 / 1800:.1f} hours")
    print(f"  With 8 workers:  {total_videos * 18 / 900:.1f} hours")
    
    print(f"\n📊 ESTIMATED PREPROCESSING TIME (without MFA):")
    print(f"  With 32 workers: {total_videos * 6 / 3600:.1f} hours")
    print(f"  With 16 workers: {total_videos * 6 / 1800:.1f} hours")
    
    # Estimate output size
    estimated_output_gb = total_videos * 3 / 1024  # ~3 MB per video
    print(f"\n💾 ESTIMATED OUTPUT SIZE: {estimated_output_gb:.2f} GB")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if total_videos < 50:
        print("✅ Small dataset - process all at once")
        print("   Recommended command:")
        pattern = "*.mp4" if 'mp4' in all_extensions else f"*.{list(all_extensions)[0]}"
        print(f"""
   python scripts/preprocess_hierarchical.py \\
     --data_dir {data_dir} \\
     --output ./processed_data \\
     --auto_split \\
     --pattern "{pattern}" \\
     --augment
        """)
    
    elif total_videos < 200:
        print("✅ Medium dataset - use parallel processing")
        pattern = "*.mp4" if 'mp4' in all_extensions else f"*.{list(all_extensions)[0]}"
        print(f"""
   python scripts/preprocess_hierarchical.py \\
     --data_dir {data_dir} \\
     --output ./processed_data \\
     --auto_split \\
     --pattern "{pattern}" \\
     --workers 16 \\
     --augment
        """)
    
    else:
        print("✅ Large dataset - maximum parallelization")
        pattern = "*.mp4" if 'mp4' in all_extensions else f"*.{list(all_extensions)[0]}"
        print(f"""
   # Recommended: Start with a test run
   python scripts/preprocess_hierarchical.py \\
     --data_dir {data_dir} \\
     --output ./processed_data_test \\
     --max_videos 10 \\
     --pattern "{pattern}"
   
   # Then process everything
   python scripts/preprocess_hierarchical.py \\
     --data_dir {data_dir} \\
     --output ./processed_data \\
     --auto_split \\
     --pattern "{pattern}" \\
     --workers 32 \\
     --augment
        """)
    
    print("\n" + "="*80)
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify data structure before preprocessing")
    parser.add_argument('--data_dir', default='./transcoded_data',
                       help='Path to transcoded_data folder')
    
    args = parser.parse_args()
    
    success = analyze_data_structure(args.data_dir)
    
    if success:
        print("\n✅ Data structure looks good! Ready for preprocessing.")
        print("\nNext step: Run the recommended command above")
    else:
        print("\n❌ Please fix the data structure issues first")
        sys.exit(1)
