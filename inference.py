"""
Production Inference Pipeline
Based on arch3.md with optimizations for deployment
"""

import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
from tqdm import tqdm

from model import CompleteLipSyncModel
from pipeline import CompletePreprocessor, PreprocessConfig


class ProductionInference:
    """
    Production-ready inference pipeline with optimizations:
    - Batch processing for efficiency
    - Model pruning support
    - ONNX export capability
    - TensorRT optimization (optional)
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        use_half_precision: bool = True,
        use_compile: bool = True,
        use_pruned_model: bool = False
    ):
        """
        Initialize production inference pipeline
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
            use_half_precision: Use FP16 for faster inference
            use_compile: Use torch.compile for optimization
            use_pruned_model: Whether checkpoint contains pruned model
        """
        self.device = torch.device(device)
        self.use_half_precision = use_half_precision and device == 'cuda'
        
        print("="*60)
        print("Initializing Production Inference Pipeline")
        print("="*60)
        
        # Load model
        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model = CompleteLipSyncModel(
            num_visemes=21,
            use_checkpoint=False,  # Disable checkpointing for inference
            use_sparse_attention=True
        ).to(self.device)

        # Load weights (handle both DataParallel and regular checkpoints)
        if 'generator_state_dict' in checkpoint:
            state_dict = checkpoint['generator_state_dict']
        elif 'generator_state' in checkpoint:
            state_dict = checkpoint['generator_state']
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present (from DataParallel)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)
        
        self.model.eval()
        print(f"  ✓ Model loaded successfully")
        
        # Apply half precision
        if self.use_half_precision:
            self.model = self.model.half()
            print(f"  ✓ Half precision (FP16) enabled")
        
        # Compile model for optimization
        if use_compile:
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print(f"  ✓ Model compiled with torch.compile")
            except Exception as e:
                print(f"  ⚠ torch.compile failed: {e}")
        
        # Initialize preprocessor
        self.preprocessor = CompletePreprocessor(PreprocessConfig())
        print(f"  ✓ Preprocessor initialized")
        
        print("="*60)
        print("Inference pipeline ready!")
        print("="*60 + "\n")
    
    @torch.no_grad()
    def process_single_frame(self, batch: Dict) -> torch.Tensor:
        """
        Process a single batch of frames
        
        Args:
            batch: Preprocessed batch dict
            
        Returns:
            Generated mouth crops [B, T, C, H, W]
        """
        # Move to device and convert dtype
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
                if self.use_half_precision and batch[key].dtype == torch.float32:
                    batch[key] = batch[key].half()
        
        # Generate
        outputs = self.model(batch)
        generated_mouths = outputs['generated_mouths']
        
        return generated_mouths
    
    def inference_on_video(
        self,
        video_path: str,
        audio_path: Optional[str] = None,
        output_path: Optional[str] = None,
        batch_size: int = 8,
        blend_ratio: float = 0.8
    ) -> str:
        """
        Run inference on a complete video
        
        Args:
            video_path: Path to input video
            audio_path: Path to audio (if separate from video)
            output_path: Path to save output video
            batch_size: Batch size for processing
            blend_ratio: Blending ratio for mouth compositing (0-1)
            
        Returns:
            Path to output video
        """
        print(f"\nProcessing video: {video_path}")
        
        # Extract audio if needed
        if audio_path is None:
            audio_path = video_path.replace('.mp4', '_audio.wav')
            if not os.path.exists(audio_path):
                print(f"Extracting audio...")
                os.system(f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{audio_path}" -y -loglevel quiet')
        
        # Preprocess video
        print("Preprocessing video...")
        preprocessed_data = self.preprocessor.process_video(
            video_path,
            audio_path,
            apply_augmentation=False
        )
        
        sequences = preprocessed_data['sequences']
        num_sequences = len(sequences)
        
        print(f"  ✓ Created {num_sequences} sequences")
        
        # Process sequences in batches
        all_generated_mouths = []
        
        print("Generating lip-sync mouths...")
        for i in tqdm(range(0, num_sequences, batch_size)):
            batch_sequences = sequences[i:i + batch_size]
            
            # Prepare batch
            batch = self._prepare_batch(batch_sequences)
            
            # Generate
            generated_mouths = self.process_single_frame(batch)
            
            # Convert to numpy
            generated_mouths = generated_mouths.cpu().float().numpy()
            all_generated_mouths.append(generated_mouths)
        
        # Concatenate all batches
        all_generated_mouths = np.concatenate(all_generated_mouths, axis=0)
        
        # Reconstruct full video
        print("Compositing final video...")
        if output_path is None:
            output_path = video_path.replace('.mp4', '_lipsync.mp4')
        
        output_path = self._composite_video(
            video_path,
            all_generated_mouths,
            preprocessed_data,
            output_path,
            blend_ratio
        )
        
        print(f"  ✓ Output saved to: {output_path}")
        
        return output_path
    
    def _prepare_batch(self, sequences: List[Dict]) -> Dict:
        """Prepare batch from list of sequences"""
        batch = {}
        
        # Stack sequences
        for key in sequences[0].keys():
            values = [seq[key] for seq in sequences]
            
            if key == 'face_crops' or key == 'mouth_crops':
                # [B, T, H, W, C] -> [B, T, C, H, W]
                values = np.array(values)
                values = torch.FloatTensor(values).permute(0, 1, 4, 2, 3)
                values = values / 127.5 - 1.0  # Normalize to [-1, 1]
            elif key == 'mel_windows':
                values = torch.FloatTensor(np.array(values))
            elif key == 'visemes':
                values = torch.LongTensor(np.array(values))
            else:
                continue
            
            batch[key] = values
        
        # Reference frame (first frame of sequence)
        if 'face_crops' in batch:
            batch['reference_frame'] = batch['face_crops'][:, 0]
        
        return batch
    
    def _composite_video(
        self,
        video_path: str,
        generated_mouths: np.ndarray,
        preprocessed_data: Dict,
        output_path: str,
        blend_ratio: float = 0.8
    ) -> str:
        """
        Composite generated mouths back into original video
        
        Args:
            video_path: Original video path
            generated_mouths: Generated mouth crops [N, T, C, H, W]
            preprocessed_data: Preprocessed data with landmarks and transforms
            output_path: Output video path
            blend_ratio: Blending ratio (0=original, 1=generated)
        """
        # Flatten sequences to frames
        N, T = generated_mouths.shape[:2]
        generated_mouths_flat = generated_mouths.reshape(N * T, *generated_mouths.shape[2:])
        
        # Denormalize to [0, 255]
        generated_mouths_flat = ((generated_mouths_flat + 1.0) * 127.5).astype(np.uint8)
        
        # Convert from [N, C, H, W] to [N, H, W, C]
        generated_mouths_flat = np.transpose(generated_mouths_flat, (0, 2, 3, 1))
        
        # Read original video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Get mouth regions
        sequences = preprocessed_data['sequences']
        mouth_transforms = []
        for seq in sequences:
            mouth_transforms.extend(seq.get('mouth_transforms', []))
        
        frame_idx = 0
        gen_idx = 0
        
        while cap.isOpened() and frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Composite mouth if we have a generated version
            if gen_idx < len(generated_mouths_flat) and gen_idx < len(mouth_transforms):
                transform = mouth_transforms[gen_idx]
                generated_mouth = generated_mouths_flat[gen_idx]
                
                # Resize generated mouth to match transform
                mouth_h, mouth_w = transform.get('height', 256), transform.get('width', 256)
                generated_mouth_resized = cv2.resize(generated_mouth, (mouth_w, mouth_h))
                
                # Get position
                x, y = transform.get('x', 0), transform.get('y', 0)
                
                # Ensure within bounds
                if 0 <= y < height - mouth_h and 0 <= x < width - mouth_w:
                    # Blend generated mouth with original
                    roi = frame[y:y+mouth_h, x:x+mouth_w]
                    blended = cv2.addWeighted(
                        roi, 1.0 - blend_ratio,
                        generated_mouth_resized, blend_ratio,
                        0
                    )
                    frame[y:y+mouth_h, x:x+mouth_w] = blended
                
                gen_idx += 1
            
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        
        # Add audio back
        output_with_audio = output_path.replace('.mp4', '_final.mp4')
        os.system(f'ffmpeg -i "{output_path}" -i "{video_path}" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest "{output_with_audio}" -y -loglevel quiet')
        
        # Clean up intermediate file
        if os.path.exists(output_with_audio):
            os.remove(output_path)
            os.rename(output_with_audio, output_path)
        
        return output_path
    
    def export_to_onnx(
        self,
        output_path: str,
        batch_size: int = 1,
        sequence_length: int = 75
    ):
        """
        Export model to ONNX format for deployment
        
        Args:
            output_path: Path to save ONNX model
            batch_size: Batch size for export
            sequence_length: Sequence length for export
        """
        print(f"\nExporting model to ONNX: {output_path}")
        
        # Create dummy inputs
        dummy_batch = {
            'visemes': torch.randint(0, 21, (batch_size, sequence_length)).to(self.device),
            'mel_windows': torch.randn(batch_size, sequence_length, 80, 7).to(self.device),
            'reference_frame': torch.randn(batch_size, 3, 512, 512).to(self.device)
        }
        
        if self.use_half_precision:
            for key in dummy_batch:
                if dummy_batch[key].dtype == torch.float32:
                    dummy_batch[key] = dummy_batch[key].half()
        
        # Export
        torch.onnx.export(
            self.model,
            (dummy_batch,),
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['visemes', 'mel_windows', 'reference_frame'],
            output_names=['generated_mouths'],
            dynamic_axes={
                'visemes': {0: 'batch_size', 1: 'sequence_length'},
                'mel_windows': {0: 'batch_size', 1: 'sequence_length'},
                'reference_frame': {0: 'batch_size'},
                'generated_mouths': {0: 'batch_size', 1: 'sequence_length'}
            }
        )
        
        print(f"  ✓ ONNX model exported to: {output_path}")
        print(f"  Input shapes:")
        print(f"    visemes: [batch, sequence]")
        print(f"    mel_windows: [batch, sequence, 80, 7]")
        print(f"    reference_frame: [batch, 3, 512, 512]")
        print(f"  Output shape:")
        print(f"    generated_mouths: [batch, sequence, 3, 256, 256]")


class BatchInference:
    """Utility for batch processing multiple videos"""
    
    def __init__(self, inference_pipeline: ProductionInference):
        self.pipeline = inference_pipeline
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        pattern: str = "*.mp4",
        batch_size: int = 8
    ):
        """
        Process all videos in a directory
        
        Args:
            input_dir: Input directory with videos
            output_dir: Output directory for processed videos
            pattern: File pattern to match
            batch_size: Batch size for processing
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        video_files = list(input_path.glob(pattern))
        print(f"\nFound {len(video_files)} videos to process")
        
        for video_file in tqdm(video_files, desc="Processing videos"):
            try:
                output_file = output_path / f"{video_file.stem}_lipsync.mp4"
                
                self.pipeline.inference_on_video(
                    str(video_file),
                    output_path=str(output_file),
                    batch_size=batch_size
                )
                
                print(f"  ✓ Processed: {video_file.name}")
                
            except Exception as e:
                print(f"  ✗ Failed to process {video_file.name}: {e}")
                continue
        
        print(f"\n✓ Batch processing complete. Output in: {output_dir}")


# Entry point for inference
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Production LipSync Inference")
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--video', help='Input video path')
    parser.add_argument('--audio', default=None, help='Input audio path (optional)')
    parser.add_argument('--output', default=None, help='Output video path')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--half_precision', action='store_true', default=True, help='Use FP16')
    parser.add_argument('--blend_ratio', type=float, default=0.8, help='Mouth blending ratio')
    parser.add_argument('--export_onnx', help='Export model to ONNX format')
    parser.add_argument('--batch_dir', help='Process all videos in directory')
    parser.add_argument('--output_dir', help='Output directory for batch processing')
    
    args = parser.parse_args()
    
    # Initialize inference pipeline
    pipeline = ProductionInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        use_half_precision=args.half_precision,
        use_compile=True
    )
    
    # Export to ONNX
    if args.export_onnx:
        pipeline.export_to_onnx(args.export_onnx)
    
    # Batch processing
    elif args.batch_dir:
        if not args.output_dir:
            args.output_dir = args.batch_dir + '_lipsync'
        
        batch_processor = BatchInference(pipeline)
        batch_processor.process_directory(
            args.batch_dir,
            args.output_dir,
            batch_size=args.batch_size
        )
    
    # Single video inference
    elif args.video:
        pipeline.inference_on_video(
            video_path=args.video,
            audio_path=args.audio,
            output_path=args.output,
            batch_size=args.batch_size,
            blend_ratio=args.blend_ratio
        )
    
    else:
        print("Error: Must specify --video, --batch_dir, or --export_onnx")
        parser.print_help()
