#!/usr/bin/env python3

"""
Download the Luxembourgish LOD male TTS dataset and prepare it for LLAMA fine-tuning.

Creates the folder structure required for fine-tuning:
data/
├── SPK1
│   ├── 21.15-26.44.lab
│   ├── 21.15-26.44.mp3
│   ├── 27.51-29.98.lab
│   ├── 27.51-29.98.mp3
│   └── ...
└── SPK2
    ├── 38.79-40.85.lab
    └── 38.79-40.85.mp3

The format matches the requirements in docs/en/finetune.md:
- Speaker folders (SPK1, SPK2, etc.)
- Audio files (.mp3, .wav, or .flac)
- .lab annotation files with transcriptions
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from huggingface_hub import snapshot_download


MALE_TTS_REPO = "denZLS/Luxembourgish-Male-TTS-for-LOD"


@dataclass
class Sample:
    source: str
    audio_path: Path
    text: str


def ensure_unzipped(zip_path: Path, target_dir: Path) -> Path:
    """Unpack a zip archive if the target folder is absent."""
    if target_dir.exists():
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(target_dir)
    return target_dir


def prepare_male_tts_dataset(work_dir: Path, token: str | None) -> Path:
    """Download and unpack the male TTS dataset."""
    base_dir = work_dir / "luxembourgish_male_tts"
    if (base_dir / "wavs").exists():
        return base_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    archive_path = base_dir / "LOD-male-dataset.zip"

    if not archive_path.exists():
        snapshot_download(
            repo_id=MALE_TTS_REPO,
            repo_type="dataset",
            local_dir=base_dir,
            token=token,
        )
    elif not (base_dir / "extracted").exists():
        print(f"[INFO] Using existing archive at {archive_path}")

    ensure_unzipped(archive_path, base_dir / "extracted")
    wavs_dir = base_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    extracted_wavs = base_dir / "extracted" / "LOD-male-dataset" / "wavs"
    if extracted_wavs.exists():
        for src in extracted_wavs.iterdir():
            target = wavs_dir / src.name
            if not target.exists():
                shutil.move(str(src), target)
        shutil.rmtree(extracted_wavs.parent.parent)

    return base_dir


def load_tts_samples(csv_path: Path, wavs_dir: Path, source_prefix: str) -> list[Sample]:
    """Load samples from TTS CSV (pipe-delimited)."""
    rows: list[tuple[str, str]] = []
    for encoding in ("utf-8", "latin-1"):
        try:
            with csv_path.open("r", encoding=encoding) as handle:
                reader = csv.reader(handle, delimiter="|")
                for row in reader:
                    if len(row) < 2:
                        continue
                    file_id, text = row[0].strip(), row[1].strip()
                    if file_id and text:
                        rows.append((file_id, text))
            break
        except UnicodeDecodeError:
            if encoding == "latin-1":
                raise
            continue

    available_files = {p.stem: p for p in wavs_dir.glob("*.wav")}

    samples: list[Sample] = []
    missing_audio: list[str] = []
    for file_id, text in rows:
        audio_path = available_files.get(file_id)
        if audio_path is not None:
            samples.append(Sample(source=source_prefix, audio_path=audio_path, text=text))
        else:
            missing_audio.append(file_id)

    if missing_audio:
        # Allow a fallback when identifiers never match filenames but counts do.
        if len(samples) == 0 and len(rows) == len(available_files):
            samples = []
            for (_, text), wav_path in zip(rows, sorted(available_files.values())):
                samples.append(Sample(source=source_prefix, audio_path=wav_path, text=text))
        else:
            preview = ", ".join(missing_audio[:5])
            raise RuntimeError(
                f"Missing audio files for {len(missing_audio)} entries (examples: {preview}). "
                "Please ensure text/audio pairs are aligned."
            )

    if not samples:
        return samples

    if len(samples) != len(rows):
        raise RuntimeError(
            f"Aligned {len(samples)} samples but input CSV contains {len(rows)} rows. "
            "Please ensure text/audio pairs are aligned."
        )

    return samples


def prepare_finetuning_format(
    samples: Sequence[Sample],
    output_dir: Path,
    speakers_per_folder: int = 1,
    speaker_prefix: str = "SPK",
    copy_audio: bool = True,
) -> None:
    """
    Prepare dataset in fine-tuning format: speaker folders with audio and .lab files.
    
    Args:
        samples: List of samples to process
        output_dir: Output directory (will contain SPK1, SPK2, etc. folders)
        speakers_per_folder: Number of samples per speaker folder
        speaker_prefix: Prefix for speaker folder names (default: "SPK")
        copy_audio: If True, copy audio files; if False, create symlinks
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group samples into speaker folders
    speaker_idx = 1
    current_speaker_samples = []
    
    for sample in samples:
        current_speaker_samples.append(sample)
        
        # Create speaker folder when we reach the threshold or at the end
        if len(current_speaker_samples) >= speakers_per_folder:
            speaker_name = f"{speaker_prefix}{speaker_idx}"
            speaker_dir = output_dir / speaker_name
            speaker_dir.mkdir(exist_ok=True)
            
            for sample_item in current_speaker_samples:
                # Use the original audio file name, or create a unique name if needed
                audio_name = sample_item.audio_path.name
                base_name = sample_item.audio_path.stem
                
                # Determine audio extension
                audio_ext = sample_item.audio_path.suffix
                if audio_ext not in ['.mp3', '.wav', '.flac']:
                    # Convert to .wav if extension is not supported
                    audio_ext = '.wav'
                
                # Create audio file path
                audio_target = speaker_dir / f"{base_name}{audio_ext}"
                
                # Create .lab file path
                lab_target = speaker_dir / f"{base_name}.lab"
                
                # Copy or link audio file
                if copy_audio:
                    shutil.copy2(sample_item.audio_path, audio_target)
                else:
                    if audio_target.exists():
                        audio_target.unlink()
                    os.symlink(sample_item.audio_path.resolve(), audio_target)
                
                # Write transcription to .lab file
                lab_target.write_text(sample_item.text.strip(), encoding="utf-8")
            
            print(f"Created {speaker_name} with {len(current_speaker_samples)} samples")
            current_speaker_samples = []
            speaker_idx += 1
    
    # Handle remaining samples
    if current_speaker_samples:
        speaker_name = f"{speaker_prefix}{speaker_idx}"
        speaker_dir = output_dir / speaker_name
        speaker_dir.mkdir(exist_ok=True)
        
        for sample_item in current_speaker_samples:
            base_name = sample_item.audio_path.stem
            audio_ext = sample_item.audio_path.suffix
            if audio_ext not in ['.mp3', '.wav', '.flac']:
                audio_ext = '.wav'
            
            audio_target = speaker_dir / f"{base_name}{audio_ext}"
            lab_target = speaker_dir / f"{base_name}.lab"
            
            if copy_audio:
                shutil.copy2(sample_item.audio_path, audio_target)
            else:
                if audio_target.exists():
                    audio_target.unlink()
                os.symlink(sample_item.audio_path.resolve(), audio_target)
            
            lab_target.write_text(sample_item.text.strip(), encoding="utf-8")
        
        print(f"Created {speaker_name} with {len(current_speaker_samples)} samples")


def prepare_dataset(
    work_dir: Path,
    output_dir: Path,
    token: str | None,
    copy_audio: bool,
    speakers_per_folder: int,
) -> None:
    """Download dataset and prepare it in fine-tuning format."""
    downloads_dir = work_dir / "artifacts"

    dataset_dir = prepare_male_tts_dataset(downloads_dir, token)
    samples = load_tts_samples(dataset_dir / "LOD-male.csv", dataset_dir / "wavs", "tts_male")
    if not samples:
        raise RuntimeError("No samples could be loaded from the LOD male dataset.")

    # Shuffle samples for better distribution across speaker folders
    random.Random(42).shuffle(samples)

    prepare_finetuning_format(
        samples,
        output_dir,
        speakers_per_folder=speakers_per_folder,
        copy_audio=copy_audio,
    )

    print(f"\nFine-tuning dataset written to {output_dir}")
    print(f"Total samples: {len(samples)}")
    print(f"Speaker folders created: {len(list(output_dir.glob('SPK*')))}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the LOD male TTS dataset and prepare it for LLAMA fine-tuning."
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("data") / "artifacts",
        help="Workspace directory for downloads (temporary files).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for fine-tuning dataset (will contain SPK1, SPK2, etc.).",
    )
    parser.add_argument(
        "--copy-audio",
        action="store_true",
        help="Copy audio files instead of creating symlinks (default: symlinks).",
    )
    parser.add_argument(
        "--speakers-per-folder",
        type=int,
        default=100,
        help="Number of samples per speaker folder (default: 100).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    token = os.environ.get("HF_TOKEN")

    try:
        prepare_dataset(
            work_dir=args.work_dir,
            output_dir=args.output_dir,
            token=token,
            copy_audio=args.copy_audio,
            speakers_per_folder=args.speakers_per_folder,
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

