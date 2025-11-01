#!/bin/bash

# Training Monitor Script
# Shows real-time training status

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Training Monitor${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Find active training process
PID=$(pgrep -f "train.py" | head -1)

if [ -z "$PID" ]; then
    echo -e "${YELLOW}No training process running${NC}"
    exit 0
fi

# Get training command
CMD=$(ps -p $PID -o cmd --no-headers)
echo -e "${CYAN}Process:${NC} PID $PID"
echo -e "${CYAN}Command:${NC} $CMD"
echo ""

# Extract batch size and workers from command
BATCH_SIZE=$(echo "$CMD" | grep -oP '(?<=--batch_size )\d+' || echo "4")
NUM_WORKERS=$(echo "$CMD" | grep -oP '(?<=--num_workers )\d+' || echo "8")
EFFECTIVE_BS=$((BATCH_SIZE * 8))

echo -e "${GREEN}Configuration:${NC}"
echo -e "  Batch size per GPU: $BATCH_SIZE"
echo -e "  Effective batch size: $EFFECTIVE_BS"
echo -e "  Data workers: $NUM_WORKERS"
echo ""

# GPU status
echo -e "${GREEN}GPU Status:${NC}"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | while read line; do
    echo "  GPU $line"
done
echo ""

# Find log file
LOG_FILE=""
for log in training_bs8_w16.log training_bs6_w12.log training_bs4_w8_fallback.log training_113k_resumed.log training_8gpu.log; do
    if [ -f "/data/gpunet_admin/gantcode/$log" ]; then
        # Check if log is recent (modified in last 5 minutes)
        if [ $(find "/data/gpunet_admin/gantcode/$log" -mmin -5 2>/dev/null | wc -l) -gt 0 ]; then
            LOG_FILE="/data/gpunet_admin/gantcode/$log"
            break
        fi
    fi
done

if [ -n "$LOG_FILE" ]; then
    echo -e "${GREEN}Latest training progress:${NC}"
    echo -e "${CYAN}Log file: $(basename $LOG_FILE)${NC}"
    echo ""
    tail -10 "$LOG_FILE" | grep -E "Epoch|Speed|samples/s|Loss" | tail -5
else
    echo -e "${YELLOW}Could not find active log file${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${CYAN}To view live logs:${NC}"
echo -e "  tail -f /data/gpunet_admin/gantcode/training_*.log"
echo -e "${GREEN}========================================${NC}"
