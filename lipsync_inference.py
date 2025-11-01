#!/usr/bin/env python3
"""
Lipsync Inference Script - Epoch 17
Runs inference on random male video with random male audio
"""
import sys
sys.path.insert(0, '/data/gpunet_admin/gantcode')

import torch
import pickle
import cv2
import numpy as np
from pathlib import Path
import random

from model import CompleteLipSyncModel

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'=' * 60)
    print(f'Lipsync Inference - Epoch 17')
    print(f'=' * 60)
    print(f'Using device: {device}')

    # Load checkpoint
    checkpoint_path = '/data/gpunet_admin/checkpoints_v31/checkpoint_epoch17_coarse.pt'
    print(f'\n[1/5] Loading checkpoint...')
    print(f'       Path: {checkpoint_path}')

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f'       ✓ Epoch: {checkpoint["epoch"]}')
    print(f'       ✓ Best val loss: {checkpoint.get("best_val_loss", "N/A")}')

    # Initialize model
    print(f'\n[2/5] Initializing model...')
    model = CompleteLipSyncModel(
        num_visemes=21,
        use_sparse_attention=True,
        use_checkpoint=False
    ).to(device)

    # Remove 'module.' prefix from state dict (from DataParallel training)
    state_dict = checkpoint['generator_state']
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '') if key.startswith('module.') else key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.eval()
    print(f'       ✓ Model loaded and set to eval mode')
    print(f'       ✓ Training stage: {checkpoint.get("stage", "unknown")}')

    # Load processed data
    data_file = '/tmp/lipsync_inference_output/selected_male_data.pkl'
    print(f'\n[3/5] Loading male data...')
    print(f'       Path: {data_file}')

    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    num_sequences = len(data['sequences'])
    print(f'       ✓ Loaded {num_sequences} sequences')
    print(f'       ✓ FPS: {data["fps"]}, Audio SR: {data["audio_sr"]}')

    # Select random sequence
    random_seq_idx = random.randint(0, num_sequences - 1)
    sequence = data['sequences'][random_seq_idx]
    print(f'       ✓ Selected random sequence #{random_seq_idx}')

    # Get frame indices to extract audio later
    frame_indices = sequence.get('frame_indices', [])

    # Run inference
    print(f'\n[4/5] Running inference...')
    with torch.no_grad():
        # Check if data is JPEG compressed
        is_jpeg = sequence.get('is_jpeg_compressed', False)

        # Decode JPEG if needed
        if is_jpeg:
            print(f'       Decoding JPEG-compressed data...')
            mouth_crops = [cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR) for img in sequence['mouth_crops']]
            face_crops = [cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR) for img in sequence['face_crops']]
            mouth_crops = np.array(mouth_crops)
            face_crops = np.array(face_crops)
        else:
            mouth_crops = sequence['mouth_crops']
            if isinstance(mouth_crops, list):
                mouth_crops = np.array(mouth_crops)
            face_crops = sequence['face_crops']
            if isinstance(face_crops, list):
                face_crops = np.array(face_crops)

        # Convert to torch tensors and normalize to [-1, 1]
        mouth_crops = torch.from_numpy(mouth_crops).float() / 127.5 - 1.0
        mouth_crops = mouth_crops.permute(0, 3, 1, 2).unsqueeze(0).to(device)  # (1, T, C, H, W)

        face_crops = torch.from_numpy(face_crops).float() / 127.5 - 1.0
        face_crops = face_crops.permute(0, 3, 1, 2).unsqueeze(0).to(device)  # (1, T, C, H, W)

        # Mel windows
        mel_windows = sequence['mel_windows']
        if isinstance(mel_windows, list):
            mel_windows = np.array(mel_windows)
        mel_windows = torch.from_numpy(mel_windows).unsqueeze(0).to(device)

        # Get visemes
        visemes = sequence['visemes']
        if isinstance(visemes, list):
            visemes = np.array(visemes)
        visemes = torch.from_numpy(visemes).long().unsqueeze(0).to(device)

        print(f'       Mouth crops shape: {mouth_crops.shape}')
        print(f'       Face crops shape: {face_crops.shape}')
        print(f'       Mel windows shape: {mel_windows.shape}')
        print(f'       Visemes shape: {visemes.shape}')

        # Create batch dictionary as expected by the model
        # Model expects: visemes [B, T], face_crops [B, T, 3, H, W], mel_windows [B, T_audio, F, W]
        batch = {
            'face_crops': face_crops,  # [1, T, 3, H, W]
            'mel_windows': mel_windows,  # [1, T_audio, F, W]
            'visemes': visemes,  # [1, T_video]
            'mouth_crops': mouth_crops  # [1, T, 3, H, W]
        }

        # Run model
        try:
            output = model(batch)
            print(f'       ✓ Output keys: {list(output.keys())}')

            # Save output
            output_path = '/tmp/lipsync_inference_output/output_lipsync.mp4'
            audio_path = '/tmp/lipsync_inference_output/output_with_audio.mp4'
            # Use 25 FPS as default if fps is 0
            fps = data['fps'] if data['fps'] > 0 else 25

            # Save video with composited mouth
            save_full_video(output, batch['face_crops'], batch['mouth_crops'], fps, output_path, audio_path, data, sequence)

        except Exception as e:
            print(f'       ✗ Error during inference: {e}')
            import traceback
            traceback.print_exc()
            return

    print(f'\n[5/5] ✓ INFERENCE COMPLETE!')
    print(f'       Output saved to: {audio_path}')
    print(f'=' * 60)

def save_full_video(output_dict, face_crops, mouth_crops_orig, fps, video_output_path, final_output_path, data, sequence):
    """
    Save full video with composited mouth and audio
    """
    # Extract the generated mouths
    if 'generated_mouths' in output_dict:
        generated_mouths = output_dict['generated_mouths']
    else:
        print(f'       Warning: No generated_mouths in output!')
        generated_mouths = mouth_crops_orig

    print(f'       Generated mouths shape (tensor): {generated_mouths.shape}')
    print(f'       Face crops shape (tensor): {face_crops.shape}')

    # Convert to numpy
    generated_mouths = generated_mouths.cpu().numpy()
    face_frames = face_crops.cpu().numpy()

    # Remove batch dimension [B, T, C, H, W] -> [T, C, H, W]
    if generated_mouths.ndim == 5:
        generated_mouths = generated_mouths[0]
    if face_frames.ndim == 5:
        face_frames = face_frames[0]

    # Denormalize from [-1, 1] to [0, 255]
    generated_mouths = ((generated_mouths + 1) * 127.5).clip(0, 255).astype(np.uint8)
    face_frames = ((face_frames + 1) * 127.5).clip(0, 255).astype(np.uint8)

    # Transpose to (T, H, W, C)
    if generated_mouths.shape[1] == 3:  # (T, C, H, W)
        generated_mouths = generated_mouths.transpose(0, 2, 3, 1)
    if face_frames.shape[1] == 3:
        face_frames = face_frames.transpose(0, 2, 3, 1)

    print(f'       Generated mouths final shape: {generated_mouths.shape}')
    print(f'       Face frames final shape: {face_frames.shape}')

    # Composite mouth into face
    # Assuming mouth is 256x256 and face is 512x512
    # Place mouth in the lower center of the face
    composited_frames = []
    mouth_h, mouth_w = generated_mouths.shape[1:3]
    face_h, face_w = face_frames.shape[1:3]

    # Calculate position to place mouth (lower center)
    y_offset = int(face_h * 0.6)  # Place at 60% down from top
    x_offset = (face_w - mouth_w) // 2  # Center horizontally

    print(f'       Compositing {len(generated_mouths)} frames...')
    for i in range(len(face_frames)):
        face = face_frames[i].copy()
        mouth = generated_mouths[i]

        # Simple alpha blend (you can improve this with proper masking)
        if y_offset + mouth_h <= face_h and x_offset + mouth_w <= face_w:
            face[y_offset:y_offset+mouth_h, x_offset:x_offset+mouth_w] = mouth

        composited_frames.append(face)

    # Save video without audio first
    height, width = composited_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    print(f'       Saving {len(composited_frames)} frames at {fps} FPS...')
    for frame in composited_frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()
    print(f'       ✓ Video saved (no audio): {video_output_path}')

    # Add audio using ffmpeg if available
    try:
        # Try to find original audio file or extract from data
        print(f'       Adding audio to video...')

        # For now, create silent audio or use extracted audio
        # You can improve this by extracting the actual audio from the source
        import subprocess

        # Create a silent audio track for the video duration
        duration = len(composited_frames) / fps

        # Try to use ffmpeg to add silent audio
        cmd = f'ffmpeg -i {video_output_path} -f lavfi -i anullsrc=r={data["audio_sr"]}:cl=mono -t {duration} -c:v copy -c:a aac -shortest {final_output_path} -y'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            print(f'       ✓ Video with audio saved: {final_output_path}')
        else:
            print(f'       ⚠ Could not add audio (ffmpeg not available or failed)')
            print(f'       Using video without audio: {video_output_path}')
            # Copy the video without audio as final output
            subprocess.run(f'cp {video_output_path} {final_output_path}', shell=True)
    except Exception as e:
        print(f'       ⚠ Audio processing failed: {e}')
        print(f'       Using video without audio')
        import subprocess
        subprocess.run(f'cp {video_output_path} {final_output_path}', shell=True)

def save_video(output_dict, input_face, fps, output_path):
    """
    Save the generated output video
    """
    # Extract the generated frames from output dict
    if 'generated_mouths' in output_dict:
        generated_frames = output_dict['generated_mouths']
    elif 'generated' in output_dict:
        generated_frames = output_dict['generated']
    elif 'mouth' in output_dict:
        generated_frames = output_dict['mouth']
    elif 'output' in output_dict:
        generated_frames = output_dict['output']
    else:
        # Just use input face if no output found
        print(f'       Warning: Using input frames (output keys: {list(output_dict.keys())})')
        generated_frames = input_face

    # Convert tensors to numpy
    print(f'       Generated frames shape (tensor): {generated_frames.shape}')
    frames = generated_frames.cpu().numpy()
    print(f'       Generated frames shape (numpy): {frames.shape}')

    # Remove batch dimension if present [B, T, C, H, W] -> [T, C, H, W]
    if frames.ndim == 5:
        frames = frames[0]
        print(f'       After removing batch dim: {frames.shape}')

    # Denormalize from [-1, 1] to [0, 255]
    frames = ((frames + 1) * 127.5).clip(0, 255).astype(np.uint8)

    # Transpose to (T, H, W, C) if needed
    if frames.ndim == 4 and frames.shape[1] == 3:  # (T, C, H, W)
        frames = frames.transpose(0, 2, 3, 1)
        print(f'       After transpose: {frames.shape}')

    # Get dimensions
    num_frames = frames.shape[0]
    height, width = frames.shape[1:3]

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f'       Saving {num_frames} frames at {fps} FPS...')

    # Write frames
    for i in range(num_frames):
        frame = frames[i]
        # Convert RGB to BGR for OpenCV
        if frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        out.write(frame_bgr)

    out.release()
    print(f'       ✓ Video saved: {output_path}')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n\nInterrupted by user')
    except Exception as e:
        print(f'\n\nFatal error: {e}')
        import traceback
        traceback.print_exc()
