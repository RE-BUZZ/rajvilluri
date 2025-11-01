#!/usr/bin/env python3
"""
Calculate total duration of all videos in transcoded_data
"""
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import json

def get_video_duration(video_path):
    """Get duration of a single video in seconds using ffprobe"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            data = json.loads(result.stdout)
            duration = float(data['format']['duration'])
            return (str(video_path), duration, True, None)
        else:
            return (str(video_path), 0, False, "ffprobe failed")

    except subprocess.TimeoutExpired:
        return (str(video_path), 0, False, "Timeout")
    except Exception as e:
        return (str(video_path), 0, False, str(e))

def main():
    transcoded_dir = Path("/data/gpunet_admin/ffmpeg/transcoded_data")

    print("Scanning for video files...")
    video_files = list(transcoded_dir.rglob("*.mp4"))

    print(f"Found {len(video_files)} MP4 files")
    print("Calculating durations (this may take a few minutes)...\n")

    total_duration = 0
    failed_videos = []
    successful_count = 0

    # Process videos in parallel
    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(get_video_duration, video): video for video in video_files}

        for future in tqdm(as_completed(futures), total=len(video_files), desc="Processing videos"):
            video_path, duration, success, error = future.result()

            if success:
                total_duration += duration
                successful_count += 1
            else:
                failed_videos.append((video_path, error))

    # Convert to various time units
    total_seconds = total_duration
    total_minutes = total_duration / 60
    total_hours = total_duration / 3600
    total_days = total_duration / 86400

    # Calculate averages
    avg_duration_seconds = total_duration / successful_count if successful_count > 0 else 0
    avg_duration_minutes = avg_duration_seconds / 60

    # Print results
    print("\n" + "="*80)
    print("VIDEO DURATION ANALYSIS")
    print("="*80)
    print(f"Total videos processed: {len(video_files)}")
    print(f"Successfully analyzed: {successful_count}")
    print(f"Failed: {len(failed_videos)}")
    print()
    print("TOTAL DURATION:")
    print(f"  {total_seconds:,.2f} seconds")
    print(f"  {total_minutes:,.2f} minutes")
    print(f"  {total_hours:,.2f} hours")
    print(f"  {total_days:,.2f} days")
    print()
    print("AVERAGE VIDEO DURATION:")
    print(f"  {avg_duration_seconds:.2f} seconds ({avg_duration_minutes:.2f} minutes)")
    print()

    if failed_videos:
        print("="*80)
        print(f"FAILED VIDEOS ({len(failed_videos)}):")
        print("="*80)
        for video, error in failed_videos[:20]:
            print(f"{Path(video).name}: {error}")
        if len(failed_videos) > 20:
            print(f"... and {len(failed_videos) - 20} more")

    print("="*80)

    # Save detailed results
    with open("video_duration_summary.txt", "w") as f:
        f.write(f"Total Videos: {len(video_files)}\n")
        f.write(f"Successfully Analyzed: {successful_count}\n")
        f.write(f"Failed: {len(failed_videos)}\n")
        f.write(f"\nTotal Duration:\n")
        f.write(f"  Seconds: {total_seconds:,.2f}\n")
        f.write(f"  Minutes: {total_minutes:,.2f}\n")
        f.write(f"  Hours: {total_hours:,.2f}\n")
        f.write(f"  Days: {total_days:,.2f}\n")
        f.write(f"\nAverage Video Duration:\n")
        f.write(f"  {avg_duration_seconds:.2f} seconds ({avg_duration_minutes:.2f} minutes)\n")

    print("\nResults saved to: video_duration_summary.txt")

if __name__ == "__main__":
    main()
