#!/usr/bin/env python3

"""
Check loudness levels of audio files in a dataset directory.

Analyzes LUFS (Loudness Units relative to Full Scale) to determine if
loudness normalization is needed. Reports statistics and recommends
whether normalization should be applied.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("[ERROR] soundfile and numpy are required. Install with: pip install soundfile numpy", file=sys.stderr)
    sys.exit(1)

try:
    import pyloudnorm as pyln
    LOUDNORM_AVAILABLE = True
except ImportError:
    LOUDNORM_AVAILABLE = False
    print("[WARNING] pyloudnorm not available. Install with: pip install pyloudnorm", file=sys.stderr)
    print("[WARNING] Falling back to RMS-based loudness estimation (less accurate)", file=sys.stderr)


def measure_loudness_pyloudnorm(audio_path: Path) -> dict[str, Any]:
    """Measure loudness using pyloudnorm (accurate LUFS measurement)."""
    try:
        data, rate = sf.read(str(audio_path))
        
        # Handle stereo/mono
        if len(data.shape) > 1:
            # Convert stereo to mono by taking mean
            data = np.mean(data, axis=1)
        
        # Measure integrated loudness (LUFS)
        meter = pyln.Meter(rate)  # Create BS.1770 meter
        loudness = meter.integrated_loudness(data)
        
        # Also measure peak
        peak = np.max(np.abs(data))
        
        return {
            "lufs": loudness,
            "peak": peak,
            "method": "pyloudnorm",
            "error": None,
        }
    except Exception as e:
        return {
            "lufs": None,
            "peak": None,
            "method": "pyloudnorm",
            "error": str(e),
        }


def measure_loudness_rms(audio_path: Path) -> dict[str, Any]:
    """Estimate loudness using RMS (less accurate fallback)."""
    try:
        data, rate = sf.read(str(audio_path))
        
        # Handle stereo/mono
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # Calculate RMS
        rms = np.sqrt(np.mean(data**2))
        
        # Convert RMS to approximate LUFS (very rough estimation)
        # This is not accurate but gives a relative measure
        # Typical conversion: LUFS ≈ 20 * log10(RMS) + offset
        # We'll use this as a relative measure only
        rms_db = 20 * np.log10(rms + 1e-10)  # Add small epsilon to avoid log(0)
        # Approximate LUFS (this is NOT accurate, just for comparison)
        estimated_lufs = rms_db - 20  # Rough offset
        
        peak = np.max(np.abs(data))
        
        return {
            "lufs": estimated_lufs,
            "peak": peak,
            "rms": rms,
            "method": "rms_estimate",
            "error": None,
        }
    except Exception as e:
        return {
            "lufs": None,
            "peak": None,
            "method": "rms_estimate",
            "error": str(e),
        }


def check_directory(data_dir: Path, speaker_folder: str = "SPK1", check_content: bool = True) -> dict[str, Any]:
    """Check loudness levels for all audio files in a speaker directory."""
    speaker_dir = data_dir / speaker_folder
    
    if not speaker_dir.exists():
        return {"error": f"Speaker directory {speaker_dir} does not exist"}
    
    if not speaker_dir.is_dir():
        return {"error": f"{speaker_dir} is not a directory"}
    
    # Find all audio files
    audio_extensions = ['.mp3', '.wav', '.flac', '.m4a']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(list(speaker_dir.glob(f"*{ext}")))
    
    if not audio_files:
        return {"error": f"No audio files found in {speaker_dir}"}
    
    print(f"Found {len(audio_files)} audio files in {speaker_dir}")
    print(f"Analyzing loudness levels...", file=sys.stderr)
    
    loudness_values = []
    peak_values = []
    errors = []
    
    # Measure loudness for each file
    measure_func = measure_loudness_pyloudnorm if LOUDNORM_AVAILABLE else measure_loudness_rms
    
    for i, audio_file in enumerate(audio_files, 1):
        # Only print progress every 5000 files to reduce verbosity
        if i % 5000 == 0:
            print(f"  Processed {i}/{len(audio_files)} files...", file=sys.stderr)
        
        result = measure_func(audio_file)
        
        if result["error"]:
            errors.append((audio_file.name, result["error"]))
        else:
            if result["lufs"] is not None:
                # Only include valid LUFS values (pyloudnorm can return -inf for silence)
                if np.isfinite(result["lufs"]) and result["lufs"] > -100:
                    loudness_values.append(result["lufs"])
                    peak_values.append(result["peak"])
                else:
                    errors.append((audio_file.name, "Silent or invalid audio"))
    
    if not loudness_values:
        return {
            "error": "No valid loudness measurements could be obtained",
            "errors": errors[:10],  # Show first 10 errors
        }
    
    loudness_array = np.array(loudness_values)
    peak_array = np.array(peak_values)
    
    stats = {
        "total_files": len(audio_files),
        "valid_measurements": len(loudness_values),
        "errors": len(errors),
        "mean_lufs": float(np.mean(loudness_array)),
        "std_lufs": float(np.std(loudness_array)),
        "min_lufs": float(np.min(loudness_array)),
        "max_lufs": float(np.max(loudness_array)),
        "median_lufs": float(np.median(loudness_array)),
        "mean_peak": float(np.mean(peak_array)),
        "max_peak": float(np.max(peak_array)),
        "method": measure_func.__name__,
    }
    
    # Determine if normalization is needed
    # Criteria:
    # 1. Standard deviation > 3 LUFS (significant variation)
    # 2. Range > 6 LUFS (wide spread)
    # 3. Mean significantly different from standard (-16 to -23 LUFS for broadcast)
    std_threshold = 3.0
    range_threshold = 6.0
    target_lufs_range = (-23.0, -16.0)
    
    needs_normalization = False
    reasons = []
    
    if stats["std_lufs"] > std_threshold:
        needs_normalization = True
        reasons.append(f"High variation (std: {stats['std_lufs']:.2f} LUFS > {std_threshold} LUFS)")
    
    loudness_range = stats["max_lufs"] - stats["min_lufs"]
    if loudness_range > range_threshold:
        needs_normalization = True
        reasons.append(f"Wide range ({loudness_range:.2f} LUFS > {range_threshold} LUFS)")
    
    if not (target_lufs_range[0] <= stats["mean_lufs"] <= target_lufs_range[1]):
        needs_normalization = True
        reasons.append(f"Mean loudness ({stats['mean_lufs']:.2f} LUFS) outside recommended range ({target_lufs_range[0]} to {target_lufs_range[1]} LUFS)")
    
    stats["needs_normalization"] = needs_normalization
    stats["reasons"] = reasons
    stats["loudness_range"] = loudness_range
    
    if errors:
        stats["sample_errors"] = errors[:5]  # Include first 5 errors
    
    return stats


def print_report(stats: dict[str, Any]) -> None:
    """Print a detailed report of loudness analysis."""
    if "error" in stats:
        print(f"\n[ERROR] {stats['error']}", file=sys.stderr)
        if "errors" in stats and stats["errors"]:
            print("\nSample errors:", file=sys.stderr)
            for filename, error in stats.get("sample_errors", []):
                print(f"  {filename}: {error}", file=sys.stderr)
        return
    
    print("\n" + "=" * 80)
    print("LOUDNESS ANALYSIS REPORT")
    print("=" * 80)
    
    print(f"\nFiles Analyzed:")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Valid measurements: {stats['valid_measurements']}")
    if stats["errors"] > 0:
        print(f"  Errors: {stats['errors']}")
    
    print(f"\nLoudness Statistics ({stats['method']}):")
    print(f"  Mean LUFS: {stats['mean_lufs']:.2f} dB")
    print(f"  Median LUFS: {stats['median_lufs']:.2f} dB")
    print(f"  Standard deviation: {stats['std_lufs']:.2f} dB")
    print(f"  Range: {stats['min_lufs']:.2f} to {stats['max_lufs']:.2f} dB")
    print(f"  Range width: {stats['loudness_range']:.2f} dB")
    
    print(f"\nPeak Levels:")
    print(f"  Mean peak: {stats['mean_peak']:.4f}")
    print(f"  Max peak: {stats['max_peak']:.4f}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    
    if stats["needs_normalization"]:
        print("\n⚠️  LOUDNESS NORMALIZATION IS RECOMMENDED")
        print("\nReasons:")
        for reason in stats["reasons"]:
            print(f"  • {reason}")
        print("\nSuggested action:")
        print("  Run: fap loudness-norm data-raw data --clean")
        print("  Or use fish-audio-preprocess to normalize your dataset")
    else:
        print("\n✓ LOUDNESS NORMALIZATION NOT NEEDED")
        print("\nYour audio files have consistent loudness levels.")
        print(f"  • Low variation (std: {stats['std_lufs']:.2f} LUFS)")
        print(f"  • Acceptable range ({stats['loudness_range']:.2f} LUFS)")
        print(f"  • Mean loudness within recommended range ({stats['mean_lufs']:.2f} LUFS)")
    
    if stats["errors"] > 0:
        print(f"\n⚠️  Note: {stats['errors']} files could not be analyzed:")
        for filename, error in stats.get("sample_errors", []):
            print(f"    {filename}: {error}")
    
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check loudness levels of audio files to determine if normalization is needed."
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Path to the dataset directory containing speaker folders",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="SPK1",
        help="Speaker folder to check (default: SPK1)",
    )
    parser.add_argument(
        "--no-content-check",
        action="store_true",
        help="Skip audio content validation (not used, kept for compatibility)",
    )
    
    args = parser.parse_args()
    
    data_dir = args.data_dir
    if not data_dir.exists():
        print(f"[ERROR] Data directory does not exist: {data_dir}", file=sys.stderr)
        return 1
    
    if not AUDIO_AVAILABLE:
        print("[ERROR] Required audio libraries not available", file=sys.stderr)
        return 1
    
    stats = check_directory(data_dir, speaker_folder=args.speaker)
    print_report(stats)
    
    # Return error code if normalization is needed or if there were errors
    if "error" in stats:
        return 1
    if stats.get("needs_normalization", False) or stats.get("errors", 0) > 0:
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

