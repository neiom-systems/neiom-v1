#!/usr/bin/env python3
"""
Test script for VQGAN semantic token extraction (Step 10).

Tests extraction on a single audio file to verify the process works correctly.
This is a minimal test that can be run locally without full dataset.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import fish_speech modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torchaudio
from hydra import compose, initialize
from hydra.utils import instantiate
from loguru import logger

from fish_speech.utils.file import AUDIO_EXTENSIONS

# Set up logging
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")


def test_single_file_extraction(
    audio_file: Path,
    config_name: str = "modded_dac_vq",
    checkpoint_path: str = "checkpoints/openaudio-s1-mini/codec.pth",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Test VQGAN extraction on a single audio file.
    
    Args:
        audio_file: Path to the test audio file
        config_name: VQGAN config name
        checkpoint_path: Path to codec checkpoint
        device: Device to use (cuda or cpu)
    """
    logger.info(f"Testing extraction on: {audio_file}")
    
    # Check if file exists
    if not audio_file.exists():
        logger.error(f"Audio file not found: {audio_file}")
        return False
    
    # Check if it's an audio file
    if audio_file.suffix.lower() not in AUDIO_EXTENSIONS:
        logger.error(f"File is not an audio file (extensions: {AUDIO_EXTENSIONS})")
        return False
    
    # Check if checkpoint exists
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Make sure you've downloaded the VQGAN weights (Step 9)")
        return False
    
    logger.info(f"Using checkpoint: {checkpoint_path}")
    logger.info(f"Using device: {device}")
    
    # Try to determine backend
    try:
        backends = torchaudio.list_audio_backends()
        if "ffmpeg" in backends:
            backend = "ffmpeg"
        else:
            backend = "soundfile"
    except AttributeError:
        backend = "soundfile"
        logger.warning("torchaudio.list_audio_backends() not available, using soundfile backend")
    
    logger.info(f"Using audio backend: {backend}")
    
    # Load model
    logger.info("Loading VQGAN model...")
    try:
        with initialize(version_base="1.3", config_path="../fish_speech/configs"):
            cfg = compose(config_name=config_name)
        
        model = instantiate(cfg)
        state_dict = torch.load(checkpoint, map_location=device, weights_only=False)
        
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        
        if any("generator" in k for k in state_dict):
            state_dict = {
                k.replace("generator.", ""): v
                for k, v in state_dict.items()
                if "generator." in k
            }
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.to(device)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False
    
    # Load audio
    logger.info(f"Loading audio file: {audio_file}")
    try:
        wav, sr = torchaudio.load(str(audio_file), backend=backend)
        
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        logger.info(f"Audio loaded: shape={wav.shape}, sample_rate={sr}, duration={len(wav[0])/sr:.2f}s")
        
        # Resample if needed
        if sr != model.sample_rate:
            logger.info(f"Resampling from {sr} Hz to {model.sample_rate} Hz")
            wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        return False
    
    # Extract tokens
    logger.info("Extracting semantic tokens...")
    try:
        wav = wav.to(device)
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        if len(wav.shape) == 2:
            wav = wav.unsqueeze(0)  # Add batch dimension: (1, channels, samples)
        
        audio_lengths = torch.tensor([wav.shape[-1]], device=device, dtype=torch.long)
        
        with torch.inference_mode():
            indices, feature_lengths = model.encode(wav, audio_lengths)
        
        # Get the output
        output = indices.cpu().numpy()
        feature_length = feature_lengths[0].item()
        
        # Extract the actual tokens (remove padding)
        tokens = output[0, :feature_length]
        
        logger.info(f"Extraction successful!")
        logger.info(f"  - Output shape: {tokens.shape}")
        logger.info(f"  - Feature length: {feature_length}")
        logger.info(f"  - Token range: [{tokens.min()}, {tokens.max()}]")
        logger.info(f"  - Unique tokens: {len(np.unique(tokens))}")
        
        # Save test output
        output_file = audio_file.with_suffix(".npy")
        np.save(output_file, tokens)
        logger.info(f"Saved tokens to: {output_file}")
        
        # Verify saved file
        loaded_tokens = np.load(output_file)
        if np.array_equal(tokens, loaded_tokens):
            logger.info("✓ Verification passed: Saved file matches extracted tokens")
        else:
            logger.error("✗ Verification failed: Saved file doesn't match extracted tokens")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to extract tokens: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test VQGAN semantic token extraction")
    parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to test audio file (.wav, .mp3, .flac, etc.)"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="modded_dac_vq",
        help="VQGAN config name"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default="checkpoints/openaudio-s1-mini/codec.pth",
        help="Path to codec checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detects if not specified."
    )
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info("=" * 60)
    logger.info("VQGAN Semantic Token Extraction Test")
    logger.info("=" * 60)
    
    success = test_single_file_extraction(
        audio_file=args.audio_file,
        config_name=args.config_name,
        checkpoint_path=str(args.checkpoint_path),
        device=device,
    )
    
    if success:
        logger.info("")
        logger.info("=" * 60)
        logger.info("✓ Test PASSED")
        logger.info("=" * 60)
        return 0
    else:
        logger.info("")
        logger.info("=" * 60)
        logger.info("✗ Test FAILED")
        logger.info("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

