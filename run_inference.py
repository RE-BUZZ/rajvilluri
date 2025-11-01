#!/usr/bin/env python3
import sys
import random
import glob

# Get random female processed file
processed_files = glob.glob('/data/gpunet_admin/processed_data/train/*.pkl')
random_file = random.choice(processed_files)
print(f'Selected random file: {random_file}')

# Run inference command
import subprocess
checkpoint = '/data/gpunet_admin/checkpoints_v31/checkpoint_epoch17_coarse.pt'
output = '/tmp/lipsync_output.mp4'

# For now, just create a simple test to verify the model loads
import torch
print(f'Loading checkpoint: {checkpoint}')
ckpt = torch.load(checkpoint, map_location='cpu')
print(f'Checkpoint keys: {list(ckpt.keys())[:5]}...')
print(f'Epoch: {ckpt.get("epoch", "unknown")}')
