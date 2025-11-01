"""
VITS Data Preprocessor - Resume-capable Version
Saves in real-time, handles existing splits, prevents data loss
"""

import whisper
import librosa
import soundfile as sf
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
import hashlib
from datetime import datetime
warnings.filterwarnings('ignore')

class VITSPreprocessor:
    def __init__(self, use_faster_model=False):
        """Initialize with resume capability"""
        # Your paths
        self.base_dir = Path(r"G:\VOICE")
        self.input_dir = Path(r"G:\VOICE\DATA\Zaddy")
        self.output_dir = Path(r"G:\VOICE\DATA\preprocessed_data")
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "wavs").mkdir(exist_ok=True)
        
        # Checkpoint files for resuming
        self.checkpoint_file = self.output_dir / "metadata_checkpoint.jsonl"
        self.processed_files_log = self.output_dir / "processed_files.txt"
        self.existing_splits_log = self.output_dir / "existing_splits_processed.txt"
        
        # Load Whisper model
        model_name = "medium" if use_faster_model else "large-v3"
        print(f"Loading Whisper {model_name} model...")
        print("This may take a moment...")
        self.whisper_model = whisper.load_model(model_name)
        print(f"✅ Model loaded successfully!")
        
        # VITS requirements
        self.target_sr = 22050  # VITS standard
        self.min_duration = 3.0  # Minimum segment length
        self.max_duration = 12.0  # Maximum segment length
        self.optimal_duration = 7.0  # Target duration
        
        print(f"📁 Input Directory: {self.input_dir}")
        print(f"📁 Output Directory: {self.output_dir}")
        print(f"💾 Checkpoint saving: ENABLED")
        
    def load_existing_metadata(self):
        """Load any existing metadata from checkpoint"""
        existing_metadata = []
        if self.checkpoint_file.exists():
            print("📂 Found existing checkpoint, loading metadata...")
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        existing_metadata.append(json.loads(line))
                    except:
                        continue
            print(f"   Loaded {len(existing_metadata)} existing segments")
        return existing_metadata
    
    def get_processed_files(self):
        """Get list of already processed original files"""
        processed = set()
        if self.processed_files_log.exists():
            with open(self.processed_files_log, 'r') as f:
                processed = set(line.strip() for line in f if line.strip())
        return processed
    
    def get_processed_splits(self):
        """Get list of already processed split files"""
        processed = set()
        if self.existing_splits_log.exists():
            with open(self.existing_splits_log, 'r') as f:
                processed = set(line.strip() for line in f if line.strip())
        return processed
    
    def process_existing_splits_first(self):
        """Process any existing split WAV files that don't have transcriptions"""
        print("\n" + "="*70)
        print("🔍 Checking for existing split files without transcriptions...")
        print("="*70)
        
        wavs_dir = self.output_dir / "wavs"
        existing_splits = list(wavs_dir.glob("*.wav"))
        
        if not existing_splits:
            print("   No existing split files found")
            return []
        
        print(f"   Found {len(existing_splits)} split WAV files")
        
        # Load existing metadata to check what's already transcribed
        existing_metadata = self.load_existing_metadata()
        transcribed_files = {item['path'].replace('wavs/', '') for item in existing_metadata}
        processed_splits = self.get_processed_splits()
        
        # Find splits that need transcription
        untranscribed = []
        for wav_file in existing_splits:
            if wav_file.name not in transcribed_files and wav_file.name not in processed_splits:
                untranscribed.append(wav_file)
        
        if not untranscribed:
            print(f"   ✅ All {len(existing_splits)} splits already have transcriptions")
            return existing_metadata
        
        print(f"   ⚠️ Found {len(untranscribed)} splits without transcriptions")
        print("   Starting transcription of existing splits...")
        
        new_metadata = []
        
        for idx, wav_file in enumerate(untranscribed, 1):
            print(f"\n   [{idx}/{len(untranscribed)}] Transcribing: {wav_file.name}")
            
            # Parse speaker from filename (assumes format: speakername_xxx_xxx.wav)
            parts = wav_file.stem.split('_')
            speaker_name = parts[0] if parts else "unknown"
            
            try:
                # Get audio duration
                audio, sr = librosa.load(wav_file, sr=self.target_sr)
                duration = len(audio) / sr
                
                # Skip if too short or too long
                if duration < self.min_duration or duration > self.max_duration:
                    print(f"       ⚠️ Skipping: duration {duration:.1f}s out of range")
                    # Mark as processed anyway
                    with open(self.existing_splits_log, 'a') as f:
                        f.write(wav_file.name + '\n')
                    continue
                
                # Transcribe
                result = self.whisper_model.transcribe(
                    str(wav_file),
                    language="en",
                    task="transcribe",
                    verbose=False,
                    temperature=0,
                    fp16=True
                )
                
                text = result.get("text", "").strip()
                
                if text:
                    metadata_item = {
                        "path": f"wavs/{wav_file.name}",
                        "text": text,
                        "speaker_id": 0,  # Will be updated later
                        "speaker_name": speaker_name,
                        "duration": duration,
                        "original_file": "recovered_split"
                    }
                    
                    # Save immediately to checkpoint
                    with open(self.checkpoint_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(metadata_item, ensure_ascii=False) + '\n')
                    
                    new_metadata.append(metadata_item)
                    print(f"       ✅ Transcribed: {text[:50]}...")
                else:
                    print(f"       ⚠️ No text detected")
                
                # Mark as processed
                with open(self.existing_splits_log, 'a') as f:
                    f.write(wav_file.name + '\n')
                    
            except Exception as e:
                print(f"       ❌ Error: {str(e)}")
                # Mark as processed anyway to avoid retrying bad files
                with open(self.existing_splits_log, 'a') as f:
                    f.write(wav_file.name + '\n')
                continue
            
            # Save progress every 10 files
            if idx % 10 == 0:
                print(f"       💾 Progress saved: {idx}/{len(untranscribed)} files done")
        
        print(f"\n   ✅ Transcribed {len(new_metadata)} existing splits")
        return existing_metadata + new_metadata
    
    def process_all_data(self):
        """Main processing function with resume capability"""
        print("\n" + "="*70)
        print("Starting VITS Data Preprocessing (Resume-capable)")
        print("="*70 + "\n")
        
        # First, handle any existing splits
        all_metadata = self.process_existing_splits_first()
        
        # Get already processed original files
        processed_files = self.get_processed_files()
        
        if processed_files:
            print(f"\n📊 Resuming from previous session:")
            print(f"   - {len(processed_files)} original files already processed")
            print(f"   - {len(all_metadata)} total segments in checkpoint")
        
        # Rebuild speaker info from existing metadata
        speaker_info = {}
        for item in all_metadata:
            speaker_name = item['speaker_name']
            if speaker_name not in speaker_info:
                speaker_info[speaker_name] = {
                    "id": item.get('speaker_id', 0),
                    "original_files": 0,
                    "created_segments": 0,
                    "total_duration": 0
                }
            speaker_info[speaker_name]["created_segments"] += 1
            speaker_info[speaker_name]["total_duration"] += item['duration']
        
        # Get all speaker folders
        speaker_dirs = [d for d in self.input_dir.iterdir() if d.is_dir()]
        
        if not speaker_dirs:
            print(f"❌ No speaker folders found in {self.input_dir}")
            if all_metadata:
                print("   But we have existing metadata, creating files...")
                self.create_vits_files_incremental(all_metadata, speaker_info)
            return
        
        print(f"\nFound {len(speaker_dirs)} speaker folders: {[d.name for d in speaker_dirs]}\n")
        
        # Process each speaker
        for speaker_id, speaker_dir in enumerate(speaker_dirs):
            speaker_name = speaker_dir.name
            print(f"\n{'='*60}")
            print(f"📁 Processing Speaker: {speaker_name} (ID: {speaker_id})")
            print(f"{'='*60}")
            
            if speaker_name not in speaker_info:
                speaker_info[speaker_name] = {
                    "id": speaker_id,
                    "original_files": 0,
                    "created_segments": 0,
                    "total_duration": 0
                }
            
            # Update speaker ID for consistency
            speaker_info[speaker_name]["id"] = speaker_id
            
            # Get WAV files
            wav_files = list(speaker_dir.glob("*.wav"))
            speaker_info[speaker_name]["original_files"] = len(wav_files)
            
            if not wav_files:
                print(f"   ⚠️ No WAV files found in {speaker_dir}")
                continue
            
            print(f"   Found {len(wav_files)} WAV files")
            
            # Count already processed
            already_processed = 0
            for wav_file in wav_files:
                file_identifier = f"{speaker_name}::{wav_file.name}"
                if file_identifier in processed_files:
                    already_processed += 1
            
            if already_processed > 0:
                print(f"   ✓ {already_processed} files already processed")
            
            # Process each file
            for file_idx, wav_file in enumerate(wav_files, 1):
                # Check if already processed
                file_identifier = f"{speaker_name}::{wav_file.name}"
                if file_identifier in processed_files:
                    continue
                
                print(f"\n   [{file_idx}/{len(wav_files)}] Processing: {wav_file.name}")
                
                try:
                    segments = self.process_single_file_with_saving(
                        wav_file, 
                        speaker_name, 
                        speaker_id
                    )
                    
                    if segments:
                        all_metadata.extend(segments)
                        speaker_info[speaker_name]["created_segments"] += len(segments)
                        speaker_info[speaker_name]["total_duration"] += sum(s["duration"] for s in segments)
                        
                        # Mark file as processed
                        with open(self.processed_files_log, 'a') as f:
                            f.write(file_identifier + '\n')
                        
                        print(f"       ✅ Created {len(segments)} segments (saved to disk)")
                        
                        # Update VITS files periodically (every 5 files)
                        if file_idx % 5 == 0:
                            print(f"       💾 Updating training files...")
                            self.create_vits_files_incremental(all_metadata, speaker_info)
                    else:
                        print(f"       ⚠️ No segments created")
                        # Still mark as processed to avoid retrying
                        with open(self.processed_files_log, 'a') as f:
                            f.write(file_identifier + '\n')
                        
                except KeyboardInterrupt:
                    print("\n\n⚠️ Process interrupted! Saving progress...")
                    self.create_vits_files_incremental(all_metadata, speaker_info)
                    print("✅ Progress saved. Run the script again to resume.")
                    return
                    
                except Exception as e:
                    print(f"       ❌ Error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    # Mark as processed to avoid retrying bad files
                    with open(self.processed_files_log, 'a') as f:
                        f.write(file_identifier + '\n')
                    continue
            
            print(f"\n   Total for {speaker_name}: {speaker_info[speaker_name]['created_segments']} segments")
        
        # Create final files
        if all_metadata:
            print("\n" + "="*60)
            print("📝 Creating final VITS training files...")
            print("="*60)
            
            self.create_vits_files_incremental(all_metadata, speaker_info)
            self.save_preprocessing_log(all_metadata, speaker_info)
            
            print("\n✅ Preprocessing Complete!")
            self.print_summary(all_metadata, speaker_info)
        else:
            print("\n❌ No segments were created. Please check your audio files.")
    
    def process_single_file_with_saving(self, audio_path, speaker_name, speaker_id):
        """Process file and save metadata immediately"""
        
        # Load audio
        try:
            audio, orig_sr = librosa.load(audio_path, sr=None)
            duration_minutes = len(audio) / orig_sr / 60
            print(f"       Duration: {duration_minutes:.2f} minutes")
        except Exception as e:
            print(f"       ❌ Could not load audio: {str(e)}")
            return []
        
        # Transcribe
        print(f"       Transcribing...")
        try:
            result = self.whisper_model.transcribe(
                str(audio_path),
                language="en",
                task="transcribe",
                verbose=False,
                temperature=0,
                best_of=1,
                beam_size=5,
                patience=1.0,
                length_penalty=1.0,
                suppress_tokens="-1",
                condition_on_previous_text=True,
                fp16=True,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                word_timestamps=False,
                initial_prompt="Clear English speech with proper sentence boundaries.",
            )
        except Exception as e:
            print(f"       ❌ Whisper error: {str(e)}")
            return []
        
        segments = result.get("segments", [])
        
        if not segments:
            print(f"       ⚠️ No speech detected")
            return []
        
        print(f"       Found {len(segments)} raw segments from Whisper")
        
        # Process segments to optimal lengths
        processed_segments = self.optimize_segment_lengths(segments)
        print(f"       Optimized to {len(processed_segments)} segments")
        
        # Save segments
        saved_segments = []
        for seg_idx, segment in enumerate(processed_segments):
            # Extract audio
            start_time = max(0, segment["start"] - 0.05)
            end_time = min(len(audio)/orig_sr, segment["end"] + 0.05)
            
            start_sample = int(start_time * orig_sr)
            end_sample = int(end_time * orig_sr)
            
            segment_audio = audio[start_sample:end_sample]
            
            # Check duration
            actual_duration = len(segment_audio) / orig_sr
            if actual_duration < self.min_duration or actual_duration > self.max_duration:
                continue
            
            # Resample to 22050 Hz
            segment_audio_22k = librosa.resample(
                segment_audio,
                orig_sr=orig_sr,
                target_sr=self.target_sr
            )
            
            # Preprocess audio
            segment_audio_22k = self.preprocess_audio(segment_audio_22k)
            
            # Save audio file
            filename = f"{speaker_name}_{audio_path.stem}_{seg_idx:03d}.wav"
            output_path = self.output_dir / "wavs" / filename
            sf.write(output_path, segment_audio_22k, self.target_sr)
            
            # Create metadata
            metadata_item = {
                "path": f"wavs/{filename}",
                "text": segment["text"].strip(),
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "duration": len(segment_audio_22k) / self.target_sr,
                "original_file": audio_path.name
            }
            
            # SAVE IMMEDIATELY to checkpoint
            with open(self.checkpoint_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(metadata_item, ensure_ascii=False) + '\n')
            
            saved_segments.append(metadata_item)
        
        return saved_segments
    
    def create_vits_files_incremental(self, metadata, speaker_info):
        """Create/update VITS files incrementally"""
        
        if not metadata:
            return
        
        # Create train/val lists
        train_list = []
        val_list = []
        
        # 95/5 split
        np.random.seed(42)
        indices = list(range(len(metadata)))
        np.random.shuffle(indices)
        split_idx = int(len(indices) * 0.95)
        
        for idx, i in enumerate(indices):
            item = metadata[i]
            # VITS format: path|speaker|text
            line = f"{item['path']}|{item['speaker_name']}|{item['text']}"
            
            if idx < split_idx:
                train_list.append(line)
            else:
                val_list.append(line)
        
        # Save train.txt
        train_path = self.output_dir / "train.txt"
        with open(train_path, "w", encoding="utf-8") as f:
            f.write("\n".join(train_list))
        
        # Save val.txt
        val_path = self.output_dir / "val.txt"
        with open(val_path, "w", encoding="utf-8") as f:
            f.write("\n".join(val_list))
        
        # Save speakers.json
        speakers_path = self.output_dir / "speakers.json"
        with open(speakers_path, "w") as f:
            json.dump(speaker_info, f, indent=2)
        
        # Save config.json
        config = {
            "train": {
                "training_files": "train.txt",
                "validation_files": "val.txt",
                "batch_size": 32,
                "learning_rate": 2e-4,
                "n_epochs": 1000,
                "seed": 1234
            },
            "data": {
                "sampling_rate": self.target_sr,
                "filter_length": 1024,
                "hop_length": 256,
                "win_length": 1024,
                "n_mel_channels": 80,
                "mel_fmin": 0,
                "mel_fmax": None
            },
            "model": {
                "n_speakers": len(speaker_info),
                "speaker_embedding_dim": 256
            }
        }
        
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    
    def optimize_segment_lengths(self, segments):
        """Optimize segment lengths to prevent mid-word cuts"""
        
        optimized = []
        current_segment = None
        
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            
            start = seg["start"]
            end = seg["end"]
            duration = end - start
            
            is_complete_sentence = any(text.rstrip().endswith(p) for p in ['.', '!', '?'])
            
            if current_segment is None:
                current_segment = {
                    "start": start,
                    "end": end,
                    "text": text,
                    "duration": duration
                }
            else:
                combined_duration = end - current_segment["start"]
                combined_would_be_too_long = combined_duration > self.max_duration
                current_is_complete = any(current_segment["text"].rstrip().endswith(p) 
                                         for p in ['.', '!', '?'])
                
                if current_segment["duration"] < self.min_duration:
                    if not combined_would_be_too_long:
                        current_segment["end"] = end
                        current_segment["text"] += " " + text
                        current_segment["duration"] = combined_duration
                    else:
                        if current_segment["duration"] >= 1.0:
                            optimized.append(current_segment)
                        current_segment = {
                            "start": start,
                            "end": end,
                            "text": text,
                            "duration": duration
                        }
                
                elif current_segment["duration"] >= self.min_duration and \
                     current_segment["duration"] <= self.optimal_duration:
                    if current_is_complete or combined_would_be_too_long:
                        optimized.append(current_segment)
                        current_segment = {
                            "start": start,
                            "end": end,
                            "text": text,
                            "duration": duration
                        }
                    else:
                        if combined_duration <= self.optimal_duration * 1.5:
                            current_segment["end"] = end
                            current_segment["text"] += " " + text
                            current_segment["duration"] = combined_duration
                        else:
                            optimized.append(current_segment)
                            current_segment = {
                                "start": start,
                                "end": end,
                                "text": text,
                                "duration": duration
                            }
                else:
                    optimized.append(current_segment)
                    current_segment = {
                        "start": start,
                        "end": end,
                        "text": text,
                        "duration": duration
                    }
        
        if current_segment and current_segment["duration"] >= self.min_duration:
            optimized.append(current_segment)
        elif current_segment and optimized:
            if optimized[-1]["duration"] + current_segment["duration"] <= self.max_duration:
                optimized[-1]["end"] = current_segment["end"]
                optimized[-1]["text"] += " " + current_segment["text"]
                optimized[-1]["duration"] = optimized[-1]["end"] - optimized[-1]["start"]
        
        return optimized
    
    def preprocess_audio(self, audio):
        """Audio preprocessing for quality"""
        
        # Normalize volume
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95
        
        # Trim silence
        audio, _ = librosa.effects.trim(
            audio, 
            top_db=23,
            frame_length=1024,
            hop_length=256
        )
        
        # Pre-emphasis
        pre_emphasis = 0.97
        audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        
        return audio.astype(np.float32)
    
    def save_preprocessing_log(self, metadata, speaker_info):
        """Save detailed log"""
        
        log_path = self.output_dir / "preprocessing_log.json"
        
        durations = [m['duration'] for m in metadata]
        
        log_data = {
            "timestamp": str(datetime.now()),
            "model_used": "whisper",
            "input_directory": str(self.input_dir),
            "output_directory": str(self.output_dir),
            "statistics": {
                "total_segments": len(metadata),
                "total_duration_minutes": sum(durations) / 60 if durations else 0,
                "average_segment_duration": np.mean(durations) if durations else 0,
                "min_segment_duration": min(durations) if durations else 0,
                "max_segment_duration": max(durations) if durations else 0,
                "std_segment_duration": np.std(durations) if durations else 0
            },
            "speakers": speaker_info,
            "sample_segments": metadata[:5] if metadata else []
        }
        
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    def print_summary(self, metadata, speaker_info):
        """Print final summary"""
        
        print("\n" + "="*70)
        print("📊 VITS PREPROCESSING COMPLETE")
        print("="*70)
        
        print(f"\n📁 Input processed from: {self.input_dir}")
        print(f"📁 Output saved to: {self.output_dir}")
        
        if metadata:
            print(f"\n📊 Statistics:")
            print(f"   Total Segments: {len(metadata)}")
            print(f"   Total Duration: {sum(m['duration'] for m in metadata)/60:.1f} minutes")
            print(f"   Average Segment: {np.mean([m['duration'] for m in metadata]):.1f} seconds")
            durations = [m['duration'] for m in metadata]
            if durations:
                print(f"   Segment Range: {min(durations):.1f}s - {max(durations):.1f}s")
        
        print(f"\n👥 Speaker Breakdown:")
        for speaker, info in speaker_info.items():
            print(f"\n   {speaker}:")
            print(f"      Original files: {info['original_files']}")
            print(f"      Created segments: {info['created_segments']}")
            print(f"      Total duration: {info['total_duration']/60:.1f} minutes")
        
        print(f"\n✅ Files Created:")
        print(f"   ✓ {self.output_dir}/wavs/ ({len(metadata)} audio files)")
        print(f"   ✓ {self.output_dir}/train.txt")
        print(f"   ✓ {self.output_dir}/val.txt")
        print(f"   ✓ {self.output_dir}/speakers.json")
        print(f"   ✓ {self.output_dir}/config.json")
        print(f"   ✓ {self.output_dir}/metadata_checkpoint.jsonl")
        
        print(f"\n🚀 Ready for VITS training!")

# Main execution
if __name__ == "__main__":
    print("="*70)
    print("VITS DATA PREPROCESSOR - RESUME CAPABLE")
    print("Real-time saving | Handles existing splits | No data loss")
    print("="*70)
    
    # Choose model speed
    print("\nModel selection:")
    print("1. Use large-v3 (best quality, slower)")
    print("2. Use medium (5x faster, still good quality)")
    
    choice = input("\nEnter choice (1 or 2, default=2 for 500hrs): ").strip()
    use_faster = choice != "1"
    
    # Create and run preprocessor
    preprocessor = VITSPreprocessor(use_faster_model=use_faster)
    
    try:
        preprocessor.process_all_data()
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted! Your progress has been saved.")
        print("✅ Run the script again to resume from where you left off.")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        print("Your progress has been saved. Fix the issue and run again to resume.")