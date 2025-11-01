"""
Complete Preprocessing Pipeline with All Fixes
Based on newarchv2.md
"""

import os
import cv2
import numpy as np
import torch
import torchaudio
import librosa
import pickle
import json
import random
import shutil
import tempfile
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from contextlib import contextmanager
import mediapipe as mp
from scipy.spatial.transform import Rotation as R
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from phonemizer import phonemize
from dataclasses import dataclass
from tqdm import tqdm
import subprocess
import webrtcvad
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Complete configuration for preprocessing"""
    # Video settings
    fps: int = 25
    mouth_size: Tuple[int, int] = (256, 256)
    face_size: Tuple[int, int] = (512, 512)

    # Audio settings
    audio_sr: int = 16000
    mel_bins: int = 80
    n_fft: int = 1024
    hop_length: int = 256

    # Sequence settings
    window_frames: int = 7
    max_seq_length: int = 75
    sequence_overlap: int = 40

    # Processing options
    use_mfa: bool = True
    normalize_pose: bool = True
    apply_augmentation: bool = True
    cache_dir: str = "./preprocessing_cache"
    use_gpu: bool = True  # Enable GPU acceleration
    gpu_id: int = 0  # GPU device ID

    # Compression settings
    use_jpeg_compression: bool = True  # Enable JPEG compression for face/mouth crops
    jpeg_quality: int = 75  # JPEG quality (0-100, 75 = good balance)

    # VAD settings
    vad_aggressiveness: int = 2


@contextmanager
def temporary_directory():
    """Safe temporary directory handling"""
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class HeadPoseNormalizer:
    """3D head pose normalization for robust mouth cropping"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 3D model points (standard face model)
        self.model_points = self._get_3d_model_points()
        
        # Previous pose for smoothing
        self.prev_pose = None
        self.smooth_factor = 0.7
    
    def estimate_pose(self, landmarks, image_shape):
        """Estimate 3D head pose from landmarks"""
        h, w = image_shape[:2]
        
        # Key landmark indices for pose estimation
        # Nose tip, chin, left eye left corner, right eye right corner, left mouth corner, right mouth corner
        indices = [1, 152, 33, 263, 61, 291]
        
        # Get 2D image points
        image_points = []
        for idx in indices:
            lm = landmarks[idx]
            image_points.append([lm.x * w, lm.y * h])
        image_points = np.array(image_points, dtype=np.float32)
        
        # Camera matrix (assuming centered principal point)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Assuming no lens distortion
        dist_coeffs = np.zeros((4, 1))
        
        # Solve PnP
        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            # Convert rotation vector to Euler angles
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            pose_mat = cv2.hconcat((rotation_mat, translation_vec))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
            
            return rotation_mat, euler_angles.flatten()
        
        return None, None
    
    def _get_3d_model_points(self):
        """Standard 3D face model points"""
        return np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float32)
    
    def normalize_and_crop(self, image, landmarks, target_size=(256, 256)):
        """Extract pose-normalized mouth crop"""
        h, w = image.shape[:2]
        
        # Estimate pose
        rotation_mat, angles = self.estimate_pose(landmarks, image.shape)
        
        # Apply perspective correction if pose is extreme
        if angles is not None:
            pitch, yaw, roll = angles
            if abs(pitch) > 15 or abs(yaw) > 20:
                image = self._apply_perspective_correction(image, angles)
        
        # Get mouth region landmarks (lips)
        mouth_indices = list(range(61, 81)) + list(range(146, 160)) + \
                       list(range(308, 320)) + list(range(375, 397))
        
        mouth_points = []
        for idx in mouth_indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                mouth_points.append([lm.x * w, lm.y * h])
        
        if len(mouth_points) == 0:
            # Fallback: use face bounding box
            logger.warning("No mouth landmarks found, using face bbox")
            x_coords = [lm.x * w for lm in landmarks]
            y_coords = [lm.y * h for lm in landmarks]
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords) + h * 0.1  # Shift down slightly
            size = max(max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)) * 0.4
        else:
            mouth_points = np.array(mouth_points)
            
            # Get bounding box with margin
            x_min, y_min = mouth_points.min(axis=0)
            x_max, y_max = mouth_points.max(axis=0)
            
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            
            width = x_max - x_min
            height = y_max - y_min
            size = max(width, height) * 1.5  # Add 50% margin
        
        # Ensure crop is within image bounds
        x1 = int(max(0, center_x - size / 2))
        y1 = int(max(0, center_y - size / 2))
        x2 = int(min(w, center_x + size / 2))
        y2 = int(min(h, center_y + size / 2))
        
        # Crop mouth region
        mouth_crop = image[y1:y2, x1:x2]
        
        # Resize to target size
        if mouth_crop.size > 0:
            mouth_crop = cv2.resize(mouth_crop, target_size, interpolation=cv2.INTER_LINEAR)
        else:
            # Fallback: return black image
            mouth_crop = np.zeros((*target_size, 3), dtype=np.uint8)
            logger.warning("Empty mouth crop, returning black image")
        
        transform_info = {
            'center': (center_x, center_y),
            'size': size,
            'bbox': (x1, y1, x2, y2)
        }
        
        return mouth_crop, transform_info
    
    def _apply_perspective_correction(self, image, angles):
        """Apply perspective transformation for head pose correction"""
        h, w = image.shape[:2]
        pitch, yaw, roll = angles
        
        # Create rotation matrix
        pitch_rad = np.radians(pitch)
        yaw_rad = np.radians(yaw)
        
        # Simple affine transformation for small rotations
        center = (w / 2, h / 2)
        
        # Rotation matrix
        M = cv2.getRotationMatrix2D(center, roll, 1.0)
        
        # Apply transformation
        corrected = cv2.warpAffine(image, M, (w, h), 
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
        
        return corrected


class ComprehensivePhonemeAligner:
    """Complete phoneme alignment with MFA and fallbacks"""
    
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.vad = webrtcvad.Vad(config.vad_aggressiveness)
        
        # Initialize Wav2Vec2 for transcription
        try:
            self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(
                "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
            )
            self.wav2vec_model = Wav2Vec2ForCTC.from_pretrained(
                "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
            )
        except Exception as e:
            logger.warning(f"Could not load Wav2Vec2: {e}")
            self.wav2vec_processor = None
            self.wav2vec_model = None
    
    def align_phonemes(self, audio_path: str, transcript: Optional[str] = None) -> List[Dict]:
        """Get accurate phoneme boundaries"""
        if self.config.use_mfa and transcript:
            try:
                return self._mfa_alignment(audio_path, transcript)
            except Exception as e:
                logger.warning(f"MFA failed: {e}, falling back to energy-based")
        
        return self._energy_based_alignment(audio_path, transcript)
    
    def _mfa_alignment(self, audio_path: str, transcript: str) -> List[Dict]:
        """Use Montreal Forced Aligner for accurate timing"""
        import tgt  # TextGrid parsing
        
        with temporary_directory() as temp_dir:
            audio_dir = os.path.join(temp_dir, 'audio')
            output_dir = os.path.join(temp_dir, 'output')
            os.makedirs(audio_dir)
            
            # Copy audio file
            audio_name = 'audio.wav'
            temp_audio = os.path.join(audio_dir, audio_name)
            shutil.copy(audio_path, temp_audio)
            
            # Create transcript file
            with open(os.path.join(audio_dir, 'audio.txt'), 'w') as f:
                f.write(transcript)
            
            # Run MFA
            cmd = [
                'mfa', 'align',
                audio_dir,
                'english_us_arpa',
                'english_us_arpa',
                output_dir
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                raise RuntimeError(f"MFA failed: {result.stderr}")
            
            # Parse TextGrid
            textgrid_path = os.path.join(output_dir, 'audio.TextGrid')
            tg = tgt.read_textgrid(textgrid_path)
            
            # Extract phonemes with timing
            phonemes = []
            for tier in tg:
                if 'phones' in tier.name.lower():
                    for interval in tier:
                        if interval.text and interval.text.strip():
                            phonemes.append({
                                'phoneme': interval.text,
                                'start_time': interval.start_time,
                                'end_time': interval.end_time
                            })
            
            return phonemes
    
    def _energy_based_alignment(self, audio_path: str, transcript: Optional[str]) -> List[Dict]:
        """Energy-based alignment with VAD"""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.config.audio_sr)
        
        # Detect speech segments
        speech_segments = self._detect_speech(audio, sr)
        
        # Get phonemes
        if transcript:
            phonemes = phonemize(transcript, language='en-us', backend='espeak')
        else:
            phonemes = self._extract_phonemes_wav2vec(audio_path)
        
        phoneme_list = phonemes.split()
        
        # Distribute phonemes across speech segments
        phoneme_data = []
        total_speech_duration = sum(end - start for start, end in speech_segments)
        
        if total_speech_duration == 0 or len(phoneme_list) == 0:
            return []
        
        phoneme_idx = 0
        for seg_start, seg_end in speech_segments:
            seg_duration = seg_end - seg_start
            num_phonemes = max(1, int(len(phoneme_list) * (seg_duration / total_speech_duration)))
            
            # Distribute phonemes evenly within segment
            for i in range(min(num_phonemes, len(phoneme_list) - phoneme_idx)):
                start_time = seg_start + (i / num_phonemes) * seg_duration
                end_time = seg_start + ((i + 1) / num_phonemes) * seg_duration
                
                phoneme_data.append({
                    'phoneme': phoneme_list[phoneme_idx],
                    'start_time': start_time,
                    'end_time': end_time
                })
                phoneme_idx += 1
                
                if phoneme_idx >= len(phoneme_list):
                    break
            
            if phoneme_idx >= len(phoneme_list):
                break
        
        return phoneme_data
    
    def _detect_speech(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """Detect speech segments using WebRTC VAD"""
        # Convert to 16-bit PCM
        audio_16bit = (audio * 32768).astype(np.int16)
        
        # Frame size for VAD (30 ms)
        frame_duration_ms = 30
        frame_length = int(sr * frame_duration_ms / 1000)
        
        segments = []
        speech_frames = []
        
        for i in range(0, len(audio_16bit) - frame_length, frame_length):
            frame = audio_16bit[i:i + frame_length].tobytes()
            
            try:
                is_speech = self.vad.is_speech(frame, sr)
                
                if is_speech:
                    if not speech_frames:
                        speech_frames.append(i / sr)
                else:
                    if speech_frames:
                        segments.append((speech_frames[0], i / sr))
                        speech_frames = []
            except:
                continue
        
        # Handle final segment
        if speech_frames:
            segments.append((speech_frames[0], len(audio_16bit) / sr))
        
        return segments
    
    def _extract_phonemes_wav2vec(self, audio_path: str) -> str:
        """Extract phonemes using Wav2Vec2"""
        if self.wav2vec_processor is None:
            return ""
        
        audio, sr = librosa.load(audio_path, sr=16000)
        inputs = self.wav2vec_processor(audio, sampling_rate=sr, return_tensors="pt")
        
        with torch.no_grad():
            logits = self.wav2vec_model(inputs.input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.wav2vec_processor.batch_decode(predicted_ids)
        
        return transcription[0] if transcription else ""


class CompleteVisemeMapper:
    """Complete 21-viseme mapping system"""
    
    # Full viseme classes
    VISEME_MAP = {
        'p b m': 0,              # Bilabial closure
        'f v': 1,                # Labiodental
        'T D': 2,                # Interdental  
        't d n': 3,              # Alveolar stop
        'l': 4,                  # Alveolar lateral
        's z': 5,                # Alveolar fricative
        'S Z tS dZ': 6,          # Post-alveolar
        'r': 7,                  # Approximant r
        'j': 8,                  # Palatal approximant
        'w': 9,                  # Labial-velar approximant
        'k g N': 10,             # Velar
        'h': 11,                 # Glottal
        'i I': 12,               # Close front
        'e E': 13,               # Mid front
        'ae': 14,                # Near-open front
        'a A': 15,               # Open
        'o O': 16,               # Mid back
        'u U': 17,               # Close back
        '@': 18,                 # Schwa
        'aI aU OI': 19,          # Diphthongs
        'sil sp': 20             # Silence/pause
    }
    
    def __init__(self):
        self.phoneme_to_viseme = {}
        for phonemes, viseme_id in self.VISEME_MAP.items():
            for phoneme in phonemes.split():
                self.phoneme_to_viseme[phoneme.lower()] = viseme_id
    
    def map_with_context(self, phoneme_sequence: List[str]) -> List[int]:
        """Map phonemes to visemes with coarticulation"""
        visemes = []
        
        for i, phoneme in enumerate(phoneme_sequence):
            base_viseme = self.phoneme_to_viseme.get(phoneme.lower(), 20)
            
            # Simple coarticulation: blend with neighbors
            if i > 0:
                prev_viseme = self.phoneme_to_viseme.get(
                    phoneme_sequence[i-1].lower(), base_viseme
                )
                # If adjacent visemes are similar, use average
                if abs(prev_viseme - base_viseme) <= 2:
                    base_viseme = (prev_viseme + base_viseme) // 2
            
            visemes.append(base_viseme)
        
        return visemes
    
    def generate_articulation_gt(self, viseme: int) -> np.ndarray:
        """Generate ground truth for articulation"""
        articulation = np.zeros(3, dtype=np.float32)  # [teeth_visible, tongue_visible, mouth_openness]
        
        # Teeth visibility for certain phonemes
        if viseme in [1, 5, 6]:  # f, v, s, z, sh
            articulation[0] = 0.8
        elif viseme in [12, 13]:  # i, e sounds
            articulation[0] = 0.5
        
        # Tongue visibility
        if viseme == 2:  # th sounds
            articulation[1] = 0.9
        elif viseme in [3, 4, 5]:  # t, d, l, s
            articulation[1] = 0.6
        
        # Mouth openness
        if viseme in [15, 14]:  # Open vowels
            articulation[2] = 0.9
        elif viseme in [13, 16]:  # Mid vowels
            articulation[2] = 0.6
        elif viseme in [12, 17]:  # Close vowels
            articulation[2] = 0.3
        else:  # Consonants
            articulation[2] = 0.2
        
        return articulation


class GPUAudioProcessor:
    """GPU-accelerated audio processing using torchaudio"""

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.device = torch.device(f'cuda:{config.gpu_id}' if config.use_gpu and torch.cuda.is_available() else 'cpu')
        logger.info(f"GPUAudioProcessor initialized on device: {self.device}")

        # Precompute mel filterbank on GPU
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.audio_sr,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.mel_bins,
            power=2.0
        ).to(self.device)

    def load_audio_gpu(self, audio_path: str) -> torch.Tensor:
        """Load and resample audio on GPU"""
        # Load audio using torchaudio
        waveform, sr = torchaudio.load(audio_path)

        # Move to GPU
        waveform = waveform.to(self.device)

        # Resample if needed
        if sr != self.config.audio_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.config.audio_sr
            ).to(self.device)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform.squeeze(0)  # Remove channel dimension

    def compute_mel_spectrogram_gpu(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram on GPU"""
        # waveform shape: (samples,)
        # Add batch and channel dimensions
        waveform = waveform.unsqueeze(0)  # (1, samples)

        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)  # (1, n_mels, time)

        # Convert to log scale (dB)
        mel_spec_db = torchaudio.functional.amplitude_to_DB(
            mel_spec,
            multiplier=10.0,
            amin=1e-10,
            db_multiplier=0.0,
            top_db=80.0
        )

        # Normalize
        mel_spec_db = mel_spec_db.squeeze(0)  # (n_mels, time)
        mean = mel_spec_db.mean()
        std = mel_spec_db.std()
        mel_spec_db = (mel_spec_db - mean) / (std + 1e-8)

        return mel_spec_db  # (n_mels, time)

    def time_stretch_gpu(self, waveform: torch.Tensor, rate: float) -> torch.Tensor:
        """Time stretch audio on GPU using phase vocoder"""
        # For time stretching, we need to use spectrogram
        spec = torch.stft(
            waveform,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            return_complex=True
        )

        # Stretch using phase vocoder
        stretched = torchaudio.functional.phase_vocoder(
            spec,
            rate=rate,
            phase_advance=self.config.hop_length
        )

        # Inverse STFT
        waveform_stretched = torch.istft(
            stretched,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )

        return waveform_stretched


class DataAugmentation:
    """Complete augmentation pipeline for training"""

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.gpu_processor = GPUAudioProcessor(config) if config.use_gpu else None
    
    def augment_video(self, frames: np.ndarray) -> np.ndarray:
        """Apply video augmentations"""
        augmented = frames.copy()
        
        # Random brightness
        if random.random() < 0.3:
            factor = random.uniform(0.8, 1.2)
            augmented = np.clip(augmented * factor, 0, 255).astype(np.uint8)
        
        # Random contrast
        if random.random() < 0.3:
            factor = random.uniform(0.8, 1.2)
            mean = augmented.mean()
            augmented = np.clip((augmented - mean) * factor + mean, 0, 255).astype(np.uint8)
        
        # Random Gaussian noise
        if random.random() < 0.2:
            noise = np.random.normal(0, 5, augmented.shape)
            augmented = np.clip(augmented + noise, 0, 255).astype(np.uint8)
        
        return augmented
    
    def augment_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply audio augmentations"""
        if self.gpu_processor is not None:
            return self._augment_audio_gpu(audio, sr)
        else:
            return self._augment_audio_cpu(audio, sr)

    def _augment_audio_gpu(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """GPU-accelerated audio augmentation"""
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float().to(self.gpu_processor.device)

        # Random volume
        if random.random() < 0.3:
            factor = random.uniform(0.7, 1.3)
            audio_tensor = audio_tensor * factor

        # Random noise
        if random.random() < 0.2:
            noise = torch.randn_like(audio_tensor) * 0.005
            audio_tensor = audio_tensor + noise

        # Time stretching
        if random.random() < 0.2:
            rate = random.uniform(0.9, 1.1)
            audio_tensor = self.gpu_processor.time_stretch_gpu(audio_tensor, rate)

        # Clip and convert back to numpy
        audio_tensor = torch.clamp(audio_tensor, -1, 1)
        return audio_tensor.cpu().numpy()

    def _augment_audio_cpu(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """CPU-based audio augmentation (fallback)"""
        augmented = audio.copy()

        # Random volume
        if random.random() < 0.3:
            factor = random.uniform(0.7, 1.3)
            augmented = augmented * factor

        # Random noise
        if random.random() < 0.2:
            noise = np.random.normal(0, 0.005, augmented.shape)
            augmented = augmented + noise

        # Time stretching
        if random.random() < 0.2:
            rate = random.uniform(0.9, 1.1)
            augmented = librosa.effects.time_stretch(augmented, rate=rate)

        return np.clip(augmented, -1, 1)


class CompletePreprocessor:
    """Complete preprocessing pipeline with all features"""

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.pose_normalizer = HeadPoseNormalizer()
        self.phoneme_aligner = ComprehensivePhonemeAligner(config)
        self.viseme_mapper = CompleteVisemeMapper()
        self.augmentation = DataAugmentation(config) if config.apply_augmentation else None
        self.gpu_audio_processor = GPUAudioProcessor(config) if config.use_gpu else None

        # Cache
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def process_video(self, video_path: str, audio_path: Optional[str] = None,
                     transcript: Optional[str] = None) -> Dict:
        """Process video and audio into training data"""
        logger.info(f"Processing: {video_path}")
        
        # Check cache
        cache_key = self._get_cache_key(video_path, audio_path)
        cache_path = Path(self.config.cache_dir) / f"{cache_key}.pkl"

        if cache_path.exists():
            logger.info("Loading from cache")
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)

            # Delete cache file immediately after loading to free disk space
            try:
                cache_path.unlink()
                logger.info(f"Deleted cache file: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_path}: {e}")

            return cached_data
        
        # Extract audio if not provided
        if audio_path is None:
            with temporary_directory() as temp_dir:
                temp_audio_path = os.path.join(temp_dir, 'audio.wav')
                cmd = [
                    'ffmpeg', '-i', video_path,
                    '-vn', '-acodec', 'pcm_s16le',
                    '-ar', str(self.config.audio_sr),
                    '-ac', '1', temp_audio_path, '-y'
                ]
                subprocess.run(cmd, capture_output=True, check=True)
                audio_features = self._extract_audio_features(temp_audio_path)

                # Align phonemes while audio file still exists
                phoneme_list = self.phoneme_aligner.align_phonemes(
                    temp_audio_path, transcript
                )
        else:
            audio_features = self._extract_audio_features(audio_path)
            # Align phonemes
            phoneme_list = self.phoneme_aligner.align_phonemes(
                audio_path, transcript
            )

        # Extract video features
        video_features = self._extract_video_features(video_path)
        
        # Align all features
        aligned_data = self._align_features(video_features, phoneme_list, audio_features)

        # Cache results (will be deleted after this function returns, only helps within same preprocessing run)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(aligned_data, f)

            # Immediately delete cache file after creating it to free disk space
            # Cache is only useful for re-processing the same video in the same run
            cache_path.unlink()
            logger.info(f"Created and deleted cache file: {cache_path}")
        except Exception as e:
            logger.warning(f"Cache operation failed: {e}")

        return aligned_data
    
    def _get_cache_key(self, video_path: str, audio_path: Optional[str]) -> str:
        """Generate cache key"""
        key = f"{video_path}_{audio_path}_{self.config.use_mfa}_{self.config.normalize_pose}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def _compress_image_jpeg(self, image: np.ndarray, quality: int = 75) -> bytes:
        """Compress image to JPEG bytes"""
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Encode as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode('.jpg', image_bgr, encode_param)
        if not success:
            raise ValueError("Failed to encode image as JPEG")
        return buffer.tobytes()

    @staticmethod
    def decompress_image_jpeg(jpeg_bytes: bytes) -> np.ndarray:
        """Decompress JPEG bytes to image (static method for use in dataloader)"""
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode JPEG bytes")
        # Convert BGR to RGB
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def _extract_video_features(self, video_path: str) -> Dict:
        """Extract face and mouth crops with pose normalization"""
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        mouth_crops = []
        face_crops = []
        landmarks_list = []
        transform_infos = []
        
        frame_idx = 0
        pbar = tqdm(desc="Extracting video features")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face landmarks
            results = self.pose_normalizer.face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Extract pose-normalized mouth crop
                if self.config.normalize_pose:
                    mouth_crop, transform_info = self.pose_normalizer.normalize_and_crop(
                        frame_rgb, landmarks, self.config.mouth_size
                    )
                else:
                    # Simple crop without pose normalization
                    h, w = frame_rgb.shape[:2]
                    mouth_points = []
                    for idx in range(61, 81):
                        if idx < len(landmarks):
                            lm = landmarks[idx]
                            mouth_points.append([lm.x * w, lm.y * h])
                    
                    if mouth_points:
                        mouth_points = np.array(mouth_points)
                        x_min, y_min = mouth_points.min(axis=0).astype(int)
                        x_max, y_max = mouth_points.max(axis=0).astype(int)
                        
                        # Add margin
                        margin = 20
                        x_min = max(0, x_min - margin)
                        y_min = max(0, y_min - margin)
                        x_max = min(w, x_max + margin)
                        y_max = min(h, y_max + margin)
                        
                        mouth_crop = frame_rgb[y_min:y_max, x_min:x_max]
                        if mouth_crop.size > 0:
                            mouth_crop = cv2.resize(mouth_crop, self.config.mouth_size)
                        else:
                            mouth_crop = np.zeros((*self.config.mouth_size, 3), dtype=np.uint8)
                        
                        transform_info = {'bbox': (x_min, y_min, x_max, y_max)}
                    else:
                        mouth_crop = np.zeros((*self.config.mouth_size, 3), dtype=np.uint8)
                        transform_info = {}
                
                # Extract face crop (for identity encoding)
                face_crop = cv2.resize(frame_rgb, self.config.face_size)
                
                frames.append(frame_rgb)
                mouth_crops.append(mouth_crop)
                face_crops.append(face_crop)
                landmarks_list.append(landmarks)
                transform_infos.append(transform_info)
            
            frame_idx += 1
            pbar.update(1)
        
        cap.release()
        pbar.close()
        
        return {
            'frames': np.array(frames),
            'mouth_crops': np.array(mouth_crops),
            'face_crops': np.array(face_crops),
            'landmarks': landmarks_list,
            'transform_infos': transform_infos,
            'fps': cap.get(cv2.CAP_PROP_FPS)
        }
    
    def _extract_audio_features(self, audio_path: str) -> Dict:
        """Extract comprehensive audio features"""
        if self.gpu_audio_processor is not None:
            return self._extract_audio_features_gpu(audio_path)
        else:
            return self._extract_audio_features_cpu(audio_path)

    def _extract_audio_features_gpu(self, audio_path: str) -> Dict:
        """GPU-accelerated audio feature extraction"""
        # Load audio on GPU
        audio_tensor = self.gpu_audio_processor.load_audio_gpu(audio_path)

        # Apply augmentation if enabled
        if self.augmentation:
            audio_np = audio_tensor.cpu().numpy()
            audio_np = self.augmentation.augment_audio(audio_np, self.config.audio_sr)
            audio_tensor = torch.from_numpy(audio_np).to(self.gpu_audio_processor.device)

        # Compute mel spectrogram on GPU
        mel_spec_db = self.gpu_audio_processor.compute_mel_spectrogram_gpu(audio_tensor)

        # Create sliding windows
        mel_spec_np = mel_spec_db.cpu().numpy()  # (n_mels, time)
        mel_windows = []
        for i in range(mel_spec_np.shape[1] - self.config.window_frames + 1):
            window = mel_spec_np[:, i:i + self.config.window_frames]
            mel_windows.append(window)

        return {
            'audio': audio_tensor.cpu().numpy(),
            'mel_spec': mel_spec_np,
            'mel_windows': np.array(mel_windows),
            'sr': self.config.audio_sr
        }

    def _extract_audio_features_cpu(self, audio_path: str) -> Dict:
        """CPU-based audio feature extraction (fallback)"""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.config.audio_sr)

        # Apply augmentation if enabled
        if self.augmentation:
            audio = self.augmentation.augment_audio(audio, sr)

        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.mel_bins
        )

        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)

        # Create sliding windows
        mel_windows = []
        for i in range(mel_spec_db.shape[1] - self.config.window_frames + 1):
            window = mel_spec_db[:, i:i + self.config.window_frames]
            mel_windows.append(window)

        return {
            'audio': audio,
            'mel_spec': mel_spec_db,
            'mel_windows': np.array(mel_windows),
            'sr': sr
        }
    
    def _align_features(self, video_features: Dict, phoneme_list: List[Dict],
                       audio_features: Dict) -> Dict:
        """Align video, audio, and phoneme features"""
        fps = video_features['fps']
        num_frames = len(video_features['mouth_crops'])
        
        # Map phonemes to visemes
        phonemes = [p['phoneme'] for p in phoneme_list]
        visemes = self.viseme_mapper.map_with_context(phonemes)
        
        # Align phonemes to frames
        frame_to_viseme = np.ones(num_frames, dtype=np.int64) * 20  # Default: silence
        
        for phoneme_data, viseme in zip(phoneme_list, visemes):
            start_frame = int(phoneme_data['start_time'] * fps)
            end_frame = int(phoneme_data['end_time'] * fps)
            
            start_frame = max(0, min(start_frame, num_frames - 1))
            end_frame = max(0, min(end_frame, num_frames))
            
            frame_to_viseme[start_frame:end_frame] = viseme
        
        # Create sequences
        sequences = []
        num_mel_windows = len(audio_features['mel_windows'])
        
        # Align mel windows with frames
        mel_per_frame = num_mel_windows / num_frames if num_frames > 0 else 1
        
        for i in range(0, num_frames - self.config.max_seq_length + 1, 
                      self.config.max_seq_length - self.config.sequence_overlap):
            
            seq_end = min(i + self.config.max_seq_length, num_frames)
            seq_len = seq_end - i
            
            # Get corresponding mel windows
            mel_start = int(i * mel_per_frame)
            mel_end = int(seq_end * mel_per_frame)
            mel_end = min(mel_end, num_mel_windows)
            
            if mel_end <= mel_start:
                continue

            # Compress images if enabled
            if self.config.use_jpeg_compression:
                mouth_crops_data = [
                    self._compress_image_jpeg(img, self.config.jpeg_quality)
                    for img in video_features['mouth_crops'][i:seq_end]
                ]
                face_crops_data = [
                    self._compress_image_jpeg(img, self.config.jpeg_quality)
                    for img in video_features['face_crops'][i:seq_end]
                ]
            else:
                mouth_crops_data = video_features['mouth_crops'][i:seq_end]
                face_crops_data = video_features['face_crops'][i:seq_end]

            sequence = {
                'mouth_crops': mouth_crops_data,
                'face_crops': face_crops_data,
                'visemes': frame_to_viseme[i:seq_end],
                'mel_windows': audio_features['mel_windows'][mel_start:mel_end],
                'frame_indices': np.arange(i, seq_end),
                'is_jpeg_compressed': self.config.use_jpeg_compression  # Flag for dataloader
            }

            sequences.append(sequence)
        
        return {
            'sequences': sequences,
            'num_frames': num_frames,
            'fps': fps,
            'audio_sr': audio_features['sr']
        }


if __name__ == "__main__":
    # Example usage
    config = PreprocessConfig(
        use_mfa=True,
        normalize_pose=True,
        apply_augmentation=True
    )
    
    preprocessor = CompletePreprocessor(config)
    
    # Process a video
    result = preprocessor.process_video(
        video_path="path/to/video.mp4",
        transcript="Hello world"
    )
    
    print(f"Processed {result['num_frames']} frames")
    print(f"Generated {len(result['sequences'])} sequences")

