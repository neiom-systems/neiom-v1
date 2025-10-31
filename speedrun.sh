#!/bin/bash
set -e  # Exit on any error

echo "========================================="
echo "Speedrun Setup Script for Fine-tuning"
echo "========================================="

# Get the directory where this script is located (relative path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Parent directory (where we'll clone audio-preprocess)
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
# Make sure we're in the script directory for relative paths
cd "$SCRIPT_DIR" || exit 1
# Repository URL for audio-preprocess
AUDIO_PREPROCESS_REPO="https://github.com/fishaudio/audio-preprocess.git"

echo "Script directory: $SCRIPT_DIR"
echo "Parent directory: $PARENT_DIR"
echo ""

# Step 1: Install system dependencies
echo "Step 1: Installing system dependencies..."
echo "Installing portaudio19-dev, libsox-dev, and ffmpeg (required for audio processing)..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq portaudio19-dev libsox-dev ffmpeg || {
        echo "Warning: Failed to install some system dependencies. Continuing anyway..."
    }
else
    echo "Warning: apt-get not found. Please install portaudio19-dev, libsox-dev, and ffmpeg manually."
fi

# Step 2: Create virtual environment
echo ""
echo "Step 2: Creating virtual environment..."
cd "$SCRIPT_DIR"
if [ -d "venv" ]; then
    echo "Virtual environment already exists, skipping creation."
else
    python3 -m venv venv
    echo "Virtual environment created."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Step 3: Install neiom-v1 dependencies
echo ""
echo "Step 3: Installing neiom-v1 dependencies..."
# Use pip with extras syntax (works in venv, uv sync requires pyproject.toml/uv.lock setup)
echo "Installing with pip (CUDA 12.8 extras)..."
pip install --upgrade pip
# Install with cu128 extra (adjust if you need cu126, cu129, or cpu)
pip install -e ".[cu128]"
# Ensure torchaudio includes libsox/ffmpeg backend so list_audio_backends is available
pip install --force-reinstall --no-deps --upgrade --index-url https://download.pytorch.org/whl/cu128 torchaudio
# Re-pin core dependencies expected by fish-speech
pip install --force-reinstall "numpy<=1.26.4" "protobuf==3.19.6"

# Install additional dependencies for loudness check
echo ""
echo "Installing dependencies for loudness check (numpy, soundfile, pyloudnorm)..."
pip install --upgrade numpy soundfile pyloudnorm
pip install hf-transfer
pip install torchcodec

# Step 4: Clone audio-preprocess repo adjacently
echo ""
echo "Step 4: Cloning audio-preprocess repository..."
cd "$PARENT_DIR"
if [ -d "audio-preprocess" ]; then
    echo "audio-preprocess directory already exists, skipping clone."
    echo "If you want to update it, please remove it first and run this script again."
else
    git clone "$AUDIO_PREPROCESS_REPO"
    echo "audio-preprocess repository cloned."
fi

# Step 5: Install audio-preprocess
echo ""
echo "Step 5: Installing audio-preprocess..."
cd "$PARENT_DIR/audio-preprocess"
pip install -e .
# Re-apply version constraints after dependency installs
pip install --force-reinstall "numpy<=1.26.4" "protobuf==3.19.6"

# Step 6: Download and prepare finetuning dataset
echo ""
echo "Step 6: Downloading and preparing finetuning dataset..."
cd "$SCRIPT_DIR"
python tools/llama/prepare_finetuning_dataset.py \
    --work-dir data \
    --output-dir data \
    --copy-audio

# Step 7: Run loudness normalization
echo ""
echo "Step 7: Running loudness normalization on dataset..."
# Normalize all speaker folders
for speaker_dir in data/SPK*/; do
    # Remove trailing slash for comparison
    speaker_clean="${speaker_dir%/}"
    if [ -d "$speaker_dir" ]; then
        echo "Normalizing loudness for $speaker_clean..."
        fap loudness-norm "$speaker_dir" "$speaker_dir" --overwrite --loudness -23.0
    fi
done

# Step 8: Run loudness check
echo ""
echo "Step 8: Running loudness check..."
# Suppress verbose progress messages, only show final report
python tools/llama/check_loudness.py data --speaker SPK1 2>&1 | grep -v "Processed.*files" || python tools/llama/check_loudness.py data --speaker SPK1

# Step 9: Check and download VQGAN weights if needed
echo ""
echo "Step 9: Checking VQGAN weights..."
cd "$SCRIPT_DIR"
CHECKPOINT_DIR="checkpoints/openaudio-s1-mini"
MODEL_FILE="$CHECKPOINT_DIR/model.pth"
CODEC_FILE="$CHECKPOINT_DIR/codec.pth"

if [ -f "$MODEL_FILE" ] && [ -f "$CODEC_FILE" ]; then
    echo "VQGAN weights already exist at $CHECKPOINT_DIR, skipping download."
else
    echo "VQGAN weights not found. Downloading from Hugging Face..."
    # Install huggingface_hub if not already installed (should be part of dependencies)
    pip install -q huggingface_hub || true
    
    # Download using huggingface-cli (will use HF_TOKEN from environment)
    # Try hf CLI first (newer), fallback to huggingface-cli if needed
    if command -v hf &> /dev/null; then
        hf download fishaudio/openaudio-s1-mini --local-dir "$CHECKPOINT_DIR"
    elif command -v huggingface-cli &> /dev/null; then
        huggingface-cli download fishaudio/openaudio-s1-mini \
            --local-dir "$CHECKPOINT_DIR" \
            --local-dir-use-symlinks False
    else
        echo "Error: Neither 'hf' nor 'huggingface-cli' found. Installing huggingface_hub..."
        pip install -q huggingface_hub
        huggingface-cli download fishaudio/openaudio-s1-mini \
            --local-dir "$CHECKPOINT_DIR" \
            --local-dir-use-symlinks False
    fi
    
    echo "VQGAN weights downloaded successfully."
fi

# Step 10: Extract semantic tokens
echo ""
echo "Step 10: Extracting semantic tokens..."
if [ ! -f "$CODEC_FILE" ]; then
    echo "Error: Expected codec weights at $CODEC_FILE but none were found."
    echo "Please ensure the download in Step 8 completed successfully."
    exit 1
fi
GPU_COUNT=$(python - <<'PY'
import os
try:
    import torch
    print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
except Exception:
    print(0)
PY
)

if [ -z "${VQ_NUM_WORKERS:-}" ]; then
    if [ "$GPU_COUNT" -le 1 ]; then
        VQ_NUM_WORKERS=1
        echo "Detected single (or no) GPU; setting VQ_NUM_WORKERS=1 for token extraction."
    else
        VQ_NUM_WORKERS="$GPU_COUNT"
    fi
else
    echo "Using user-provided VQ_NUM_WORKERS=${VQ_NUM_WORKERS}."
fi

if [ -z "${VQ_BATCH_SIZE:-}" ]; then
    if [ "$GPU_COUNT" -le 1 ]; then
        VQ_BATCH_SIZE=32
    else
        VQ_BATCH_SIZE=64
    fi
else
    echo "Using user-provided VQ_BATCH_SIZE=${VQ_BATCH_SIZE}."
fi

export VQ_NUM_WORKERS VQ_BATCH_SIZE
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$SCRIPT_DIR"
readarray -t SPEAKER_DIRS < <(find data -maxdepth 1 -mindepth 1 -type d -name "SPK*")

if [ ${#SPEAKER_DIRS[@]} -eq 0 ]; then
    echo "Error: No speaker directories (SPK*) found under data/. Aborting token extraction."
    exit 1
fi

for speaker_dir in "${SPEAKER_DIRS[@]}"; do
    echo "Extracting tokens for ${speaker_dir}..."
    python tools/vqgan/extract_vq.py "$speaker_dir" \
        --num-workers "$VQ_NUM_WORKERS" \
        --batch-size "$VQ_BATCH_SIZE" \
        --config-name "modded_dac_vq" \
        --checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth"
done

# Step 11: Pack the dataset into protobuf
echo ""
echo "Step 11: Packing dataset into protobuf..."
cd "$SCRIPT_DIR"

echo "Debug: verifying protobuf installation before building dataset..."
python - <<'PY'
import sys
try:
    import google.protobuf
    version = getattr(google.protobuf, "__version__", "unknown")
    module_file = getattr(google.protobuf, "__file__", "unknown")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"google.protobuf version: {version}")
    print(f"google.protobuf module path: {module_file}")
    try:
        from google.protobuf.internal import builder as _builder
        print(f"'builder' import succeeded: {_builder}")
    except Exception as sub_exc:
        print(f"'builder' import failed: {sub_exc}")
except Exception as exc:
    print(f"Failed to import google.protobuf: {exc}")
PY

python tools/llama/build_dataset.py \
    --input "data" \
    --output "data/protos" \
    --text-extension .lab \
    --num-workers 16

# Step 12: Start fine-tuning with LoRA
echo ""
echo "Step 12: Starting fine-tuning with LoRA for Luxembourgish..."
cd "$SCRIPT_DIR"
echo "Note: Fine-tuning will run with the optimized Luxembourgish config."
echo "Training parameters can be adjusted in fish_speech/configs/text2semantic_finetune_luxembourgish_optimized.yaml"
echo ""
echo "Starting training..."
python fish_speech/train.py \
    --config-name text2semantic_finetune_luxembourgish_optimized \
    project=text2semantic_finetune_luxembourgish \
    +lora@model.model.lora_config=r_32_alpha_64

# Step 13: Clean up checkpoints to save disk space
echo ""
echo "Step 13: Cleaning up checkpoints to save disk space..."
cd "$SCRIPT_DIR"

PROJECT_NAME="text2semantic_finetune_luxembourgish"
CHECKPOINT_DIR="results/$PROJECT_NAME/checkpoints"

if [ -d "$CHECKPOINT_DIR" ]; then
    # Keep these specific checkpoints
    KEEP_CHECKPOINTS=("step_000002000.ckpt" "step_000004000.ckpt")
    
    # Find the 2 latest checkpoints (highest step numbers)
    LATEST_CHECKPOINTS=$(ls -t "$CHECKPOINT_DIR"/step_*.ckpt 2>/dev/null | head -2)
    
    echo "Keeping checkpoints:"
    # Mark specific checkpoints to keep
    for ckpt in "${KEEP_CHECKPOINTS[@]}"; do
        if [ -f "$CHECKPOINT_DIR/$ckpt" ]; then
            echo "  - $ckpt"
            touch "$CHECKPOINT_DIR/.keep_${ckpt}"  # Mark file to keep
        fi
    done
    
    # Mark latest 2 to keep
    for ckpt in $LATEST_CHECKPOINTS; do
        ckpt_name=$(basename "$ckpt")
        echo "  - $ckpt_name (latest)"
        touch "$CHECKPOINT_DIR/.keep_${ckpt_name}"  # Mark file to keep
    done
    
    # Delete all checkpoints except marked ones
    DELETED_COUNT=0
    for ckpt in "$CHECKPOINT_DIR"/step_*.ckpt; do
        if [ -f "$ckpt" ]; then
            ckpt_name=$(basename "$ckpt")
            if [ ! -f "$CHECKPOINT_DIR/.keep_${ckpt_name}" ]; then
                echo "  Deleting: $ckpt_name"
                rm -f "$ckpt"
                ((DELETED_COUNT++))
            fi
        fi
    done
    
    # Clean up marker files
    rm -f "$CHECKPOINT_DIR"/.keep_*.ckpt
    
    echo "Cleanup complete. Deleted $DELETED_COUNT checkpoint(s)."
    echo "Kept checkpoints:"
    ls -1 "$CHECKPOINT_DIR"/step_*.ckpt 2>/dev/null | while read ckpt; do
        echo "  - $(basename "$ckpt")"
    done
else
    echo "Warning: Checkpoint directory $CHECKPOINT_DIR does not exist."
fi

# Step 14: Merge LoRA weights to regular weights
echo ""
echo "Step 14: Merging LoRA weights to regular weights..."
cd "$SCRIPT_DIR"

PROJECT_NAME="text2semantic_finetune_luxembourgish"
CHECKPOINT_DIR="results/$PROJECT_NAME/checkpoints"
OUTPUT_BASE_DIR="checkpoints/openaudio-s1-mini-luxembourgish"

if [ -d "$CHECKPOINT_DIR" ]; then
    readarray -t MERGE_TARGETS < <(find "$CHECKPOINT_DIR" -maxdepth 1 -name 'step_*.ckpt' -print | sort)

    if [ ${#MERGE_TARGETS[@]} -eq 0 ]; then
        echo "Warning: No checkpoints found in $CHECKPOINT_DIR"
        echo "Skipping LoRA merge. You may need to run training first or merge manually later."
    else
        echo "Preparing to merge ${#MERGE_TARGETS[@]} checkpoint(s):"
        for ckpt in "${MERGE_TARGETS[@]}"; do
            echo "  - $(basename "$ckpt")"
        done

        for ckpt in "${MERGE_TARGETS[@]}"; do
            ckpt_name=$(basename "$ckpt" .ckpt)
            OUTPUT_DIR="$OUTPUT_BASE_DIR-$ckpt_name"

            echo ""
            echo "Merging $ckpt into $OUTPUT_DIR..."
            python tools/llama/merge_lora.py \
                --lora-config r_8_alpha_16 \
                --base-weight checkpoints/openaudio-s1-mini \
                --lora-weight "$ckpt" \
                --output "$OUTPUT_DIR"
        done

        echo ""
        echo "LoRA merges completed. Outputs stored under $OUTPUT_BASE_DIR-<checkpoint_step>"
    fi
else
    echo "Warning: Checkpoint directory $CHECKPOINT_DIR does not exist."
    echo "Skipping LoRA merge. Training may not have completed successfully."
fi

echo ""
echo "========================================="
echo "All steps complete!"
echo "========================================="
echo ""
echo "Summary:"
echo "  - Merged model saved to: $OUTPUT_DIR"
echo "  - You can now use this model for inference."
echo "  - To merge a different checkpoint, see the merge_lora.py command above."
