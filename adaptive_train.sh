#!/bin/bash

# Enhanced Adaptive Training Script - Fully Autonomous
# Will auto-switch configs on OOM without user intervention

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

DATA_DIR="/data/gpunet_admin/processed_data"
CHECKPOINT_DIR="/data/gpunet_admin/checkpoints_v31"
LOG_DIR="/data/gpunet_admin/gantcode"
STATUS_FILE="$LOG_DIR/.training_status"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Adaptive Training Launcher v2.0${NC}"
echo -e "${GREEN}Fully Autonomous Mode${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Cleanup function
cleanup_training() {
    echo -e "${YELLOW}Cleaning up any existing training...${NC}"
    pkill -SIGTERM -f "train.py" 2>/dev/null
    sleep 10
    pkill -9 -f "train.py" 2>/dev/null
    sleep 2
    
    # Clear GPU memory
    nvidia-smi --gpu-reset 2>/dev/null || true
    sleep 2
    
    echo -e "${GREEN}✓ Cleanup complete${NC}"
}

# Log function
log_status() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$STATUS_FILE"
}

# Check function - monitors for errors
check_training() {
    local pid=$1
    local log_file=$2
    local check_duration=180  # 3 minutes
    
    echo -e "${YELLOW}Monitoring for $check_duration seconds...${NC}"
    
    for i in $(seq 1 $check_duration); do
        sleep 1
        
        # Check if process died
        if ! kill -0 $pid 2>/dev/null; then
            echo -e "${RED}✗ Process died at ${i}s${NC}"
            log_status "ERROR: Process died early"
            return 1
        fi
        
        # Check for CUDA OOM
        if grep -qi "CUDA out of memory\|OutOfMemoryError\|cuda runtime error" "$log_file" 2>/dev/null; then
            echo -e "${RED}✗ CUDA Out of Memory detected at ${i}s${NC}"
            log_status "ERROR: CUDA OOM"
            kill -9 $pid 2>/dev/null
            sleep 5
            return 1
        fi
        
        # Check for runtime errors (but ignore warnings)
        if grep -E "RuntimeError:|Exception:|Error:" "$log_file" 2>/dev/null | grep -vqi "warning\|deprecated"; then
            # Double check it's not just a warning
            if tail -20 "$log_file" | grep -E "Traceback|Exception" | grep -vqi "warning" 2>/dev/null; then
                echo -e "${RED}✗ Runtime error detected at ${i}s${NC}"
                log_status "ERROR: Runtime error"
                tail -20 "$log_file" | grep -A 5 -B 5 "Error" | head -10
                kill -9 $pid 2>/dev/null
                sleep 5
                return 1
            fi
        fi
        
        # Show progress every 30 seconds
        if [ $((i % 30)) -eq 0 ]; then
            echo -e "${CYAN}  [$i/${check_duration}s] Still running...${NC}"
            
            # Show training progress if available
            if tail -10 "$log_file" 2>/dev/null | grep -q "Epoch\|samples/s"; then
                progress=$(tail -10 "$log_file" | grep -E "Epoch|Speed" | tail -1)
                echo -e "${GREEN}  $progress${NC}"
            fi
            
            # Show GPU memory
            gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
            echo -e "${CYAN}  GPU0 Memory: ${gpu_mem}MB / 81920MB${NC}"
        fi
    done
    
    # Success!
    echo -e "${GREEN}✓ Training stable for $check_duration seconds${NC}"
    log_status "SUCCESS: Training stable"
    return 0
}

# Try configuration
try_config() {
    local batch_size=$1
    local num_workers=$2
    local config_name=$3
    local log_file="$LOG_DIR/$4"
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Config: $config_name${NC}"
    echo -e "${GREEN}  Batch: $batch_size/GPU (${batch_size}×8 = $((batch_size * 8)) total)${NC}"
    echo -e "${GREEN}  Workers: $num_workers${NC}"
    echo -e "${GREEN}  Log: $(basename $log_file)${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    
    log_status "Trying $config_name: bs=$batch_size, workers=$num_workers"
    
    # Clear old log
    > "$log_file"
    
    # Start training
    cd /data/gpunet_admin/gantcode
    python train.py \
        --data_dir "$DATA_DIR" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --batch_size $batch_size \
        --num_workers $num_workers \
        --use_wandb \
        --run_name "v31-bs${batch_size}-w${num_workers}" \
        >> "$log_file" 2>&1 &
    
    local pid=$!
    echo -e "${YELLOW}Started PID: $pid${NC}"
    echo "$pid" > "$LOG_DIR/.training_pid"
    
    # Monitor
    if check_training $pid "$log_file"; then
        # Success - training is stable
        echo ""
        echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║  ✓ TRAINING STARTED SUCCESSFULLY      ║${NC}"
        echo -e "${GREEN}║                                        ║${NC}"
        echo -e "${GREEN}║  Config: $config_name${NC}"
        echo -e "${GREEN}║  Batch: $((batch_size * 8)) (${batch_size} per GPU)${NC}"
        echo -e "${GREEN}║  Workers: $num_workers${NC}"
        echo -e "${GREEN}║  PID: $pid${NC}"
        echo -e "${GREEN}║  Log: $(basename $log_file)${NC}"
        echo -e "${GREEN}║                                        ║${NC}"
        echo -e "${GREEN}║  Monitor: ./monitor_training.sh        ║${NC}"
        echo -e "${GREEN}║  Logs: tail -f $(basename $log_file)${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
        echo ""
        
        log_status "SUCCESS: Running with $config_name (PID: $pid)"
        return 0
    else
        # Failed
        echo -e "${RED}✗ $config_name failed${NC}"
        log_status "FAILED: $config_name"
        cleanup_training
        return 1
    fi
}

# Main execution
log_status "========== Starting Adaptive Training =========="
cleanup_training

echo -e "${CYAN}System Check:${NC}"
echo -e "  GPUs: $(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)"
echo -e "  Free GPU Memory: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) MB"
echo -e "  Dataset: 113,826 sequences"
echo ""

# Try configs in order: Aggressive → Medium → Current
if try_config 8 16 "AGGRESSIVE" "training_bs8_w16.log"; then
    echo -e "${GREEN}🚀 Running with AGGRESSIVE config (~2.0x speedup)${NC}"
    exit 0
fi

echo -e "${YELLOW}Aggressive failed, trying Medium...${NC}"
sleep 10

if try_config 6 12 "MEDIUM" "training_bs6_w12.log"; then
    echo -e "${GREEN}🚀 Running with MEDIUM config (~1.95x speedup)${NC}"
    exit 0
fi

echo -e "${YELLOW}Medium failed, trying Current...${NC}"
sleep 10

if try_config 4 8 "CURRENT" "training_bs4_w8.log"; then
    echo -e "${GREEN}🚀 Running with CURRENT config (~1.81x speedup)${NC}"
    exit 0
fi

# All failed
echo ""
echo -e "${RED}╔════════════════════════════════════════╗${NC}"
echo -e "${RED}║  ✗ ALL CONFIGURATIONS FAILED          ║${NC}"
echo -e "${RED}║                                        ║${NC}"
echo -e "${RED}║  Check logs:                           ║${NC}"
echo -e "${RED}║  - training_bs8_w16.log                ║${NC}"
echo -e "${RED}║  - training_bs6_w12.log                ║${NC}"
echo -e "${RED}║  - training_bs4_w8.log                 ║${NC}"
echo -e "${RED}╚════════════════════════════════════════╝${NC}"
echo ""

log_status "ERROR: All configurations failed"
exit 1
