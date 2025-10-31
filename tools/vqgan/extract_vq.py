import os
import subprocess as sp
import sys
import time
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from random import Random

import click
import numpy as np
import torch
import torchaudio
from hydra import compose, initialize
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

from fish_speech.utils.file import AUDIO_EXTENSIONS, list_files, load_filelist

# register eval resolver
OmegaConf.register_new_resolver("eval", eval)
# This file is used to convert the audio files to text files using the Whisper model.
# It's mainly used to generate the training data for the VQ model.

PREFERRED_BACKENDS = (
    "ffmpeg",
    "sox_io",
    "soundfile",
    None,  # let torchaudio decide as last resort
)

RANK = int(os.environ.get("SLURM_PROCID", 0))
WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", 1))

logger_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "{extra[rank]} - <level>{message}</level>"
)
logger.configure(extra={"rank": f"RANK: {RANK} / {WORLD_SIZE}"})
logger.remove()
logger.add(sys.stderr, format=logger_format)


@lru_cache(maxsize=1)
def get_model(
    config_name: str = "modded_dac_vq",
    checkpoint_path: str = "checkpoints/openaudio-s1-mini/codec.pth",
    device: str | torch.device = "cuda",
):
    with initialize(version_base="1.3", config_path="../../fish_speech/configs"):
        cfg = compose(config_name=config_name)

    model = instantiate(cfg)
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
    )
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
    
    # Log model memory footprint
    if torch.cuda.is_available():
        model_params = sum(p.numel() for p in model.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        logger.info(f"Loaded model: {model_params:,} parameters, ~{model_size_mb:.2f}MB")
        logger.info(f"Model device: {next(model.parameters()).device}, {_get_gpu_memory_info()}")

    return model


@torch.inference_mode()
def process_batch(files: list[Path], model, batch_num: int = 0) -> float:
    return _process_batch_internal(files, model, batch_num)


def _get_gpu_memory_info() -> str:
    """Get detailed GPU memory information for debugging."""
    if not torch.cuda.is_available():
        return "CUDA not available"
    try:
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
        free = total - reserved
        return (f"GPU {device}: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB, "
                f"free={free:.2f}GB, total={total:.2f}GB")
    except Exception as e:
        return f"Error getting GPU memory info: {e}"


def _get_file_info(file: Path) -> str:
    """Get file size and duration info for debugging."""
    try:
        size_mb = file.stat().st_size / 1024**2
        # Try to get duration if possible
        try:
            import soundfile as sf
            info = sf.info(file)
            duration = info.duration
            sr = info.samplerate
            return f"size={size_mb:.2f}MB, duration={duration:.2f}s, sr={sr}Hz"
        except:
            return f"size={size_mb:.2f}MB"
    except Exception as e:
        return f"Error getting file info: {e}"


def _process_batch_internal(files: list[Path], model, batch_num: int = 0) -> float:
    try:
        return _process_batch_once(files, model, batch_num)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        error_msg = str(exc).lower()
        if not isinstance(exc, torch.cuda.OutOfMemoryError) and "out of memory" not in error_msg:
            raise

        # Add detailed debugging info
        logger.error(f"[OOM DEBUG] ========== CUDA OUT OF MEMORY ERROR ==========")
        logger.error(f"[OOM DEBUG] Exception type: {type(exc).__name__}")
        logger.error(f"[OOM DEBUG] Exception message: {str(exc)}")
        logger.error(f"[OOM DEBUG] Batch size: {len(files)}")
        logger.error(f"[OOM DEBUG] {_get_gpu_memory_info()}")
        if torch.cuda.is_available():
            try:
                import traceback
                logger.error(f"[OOM DEBUG] Traceback (last 3 frames):")
                tb_lines = traceback.format_exc().split('\n')
                for line in tb_lines[-6:]:  # Last few lines of traceback
                    if line.strip():
                        logger.error(f"[OOM DEBUG] {line}")
            except:
                pass
        if len(files) <= 5:
            # Log details for small batches
            for f in files:
                logger.error(f"[OOM DEBUG] File: {f.name} - {_get_file_info(f)}")
        else:
            # For larger batches, just log first and last
            logger.error(f"[OOM DEBUG] First file: {files[0].name} - {_get_file_info(files[0])}")
            logger.error(f"[OOM DEBUG] Last file: {files[-1].name} - {_get_file_info(files[-1])}")

        if len(files) == 1:
            logger.error(
                f"Out of memory while processing {files[0]}. Skipping this file. "
                f"File info: {_get_file_info(files[0])}"
            )
            torch.cuda.empty_cache()
            return 0.0

        torch.cuda.empty_cache()
        mid = len(files) // 2
        left = files[:mid]
        right = files[mid:]
        logger.warning(
            f"Out of memory encountered with batch size {len(files)}. "
            f"Splitting into {len(left)} and {len(right)}. "
            f"GPU memory before split: {_get_gpu_memory_info()}"
        )
        return _process_batch_internal(left, model, batch_num) + _process_batch_internal(right, model, batch_num)


def _process_batch_once(files: list[Path], model, batch_num: int = 0) -> float:
    # Log GPU memory before processing batch (only for first few batches to reduce verbosity)
    if torch.cuda.is_available() and len(files) > 1 and batch_num < 3:
        logger.info(f"[BATCH DEBUG] Starting batch #{batch_num} of {len(files)} files. {_get_gpu_memory_info()}")
    
    wavs = []
    audio_lengths = []
    new_files = []
    max_length = total_time = 0

    for file in files:
        wav = None
        sr = None
        last_error = None

        for candidate_backend in PREFERRED_BACKENDS:
            try:
                wav, sr = torchaudio.load(
                    str(file), backend=candidate_backend
                )  # Need to install libsox-dev
                break
            except Exception as e:  # noqa: BLE001
                last_error = e
                continue

        if wav is None:
            logger.error(
                f"Failed to read {file} with torchaudio backends "
                f"{PREFERRED_BACKENDS}: {last_error}"
            )
            continue

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        wav = torchaudio.functional.resample(wav.cuda(), sr, model.sample_rate)[0]
        total_time += len(wav) / model.sample_rate
        max_length = max(max_length, len(wav))

        wavs.append(wav)
        audio_lengths.append(len(wav))
        new_files.append(file)

    files = new_files

    if len(wavs) == 0:
        return 0.0

    # Pad to max length
    for i, wav in enumerate(wavs):
        wavs[i] = torch.nn.functional.pad(wav, (0, max_length - len(wav)), "constant")

    audios = torch.stack(wavs, dim=0)[:, None]
    audio_lengths = torch.tensor(audio_lengths, device=model.device, dtype=torch.long)
    
    # Log memory info before encoding
    if torch.cuda.is_available():
        total_samples = sum(audio_lengths.cpu().numpy()) if len(audio_lengths) > 0 else 0
        logger.info(f"[ENCODE DEBUG] Before encoding: batch_size={len(wavs)}, max_audio_len={max_length}, "
                    f"total_samples={total_samples}, {_get_gpu_memory_info()}")

    # Calculate lengths
    indices, feature_lengths = model.encode(audios, audio_lengths)
    
    # Log memory after encoding
    if torch.cuda.is_available():
        logger.info(f"[ENCODE DEBUG] After encoding: {_get_gpu_memory_info()}")

    # Save to disk
    outputs = indices.cpu().numpy()
    
    # Log memory after moving to CPU
    if torch.cuda.is_available():
        logger.info(f"[ENCODE DEBUG] After moving to CPU: {_get_gpu_memory_info()}")

    for file, length, feature, audio_length in zip(
        files, feature_lengths, outputs, audio_lengths
    ):
        feature = feature[:, :length]

        # (T,)
        with open(file.with_suffix(".npy"), "wb") as f:
            np.save(f, feature)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"[BATCH DEBUG] After batch completion: {_get_gpu_memory_info()}")

    return total_time


@click.command()
@click.argument("folder")
@click.option("--num-workers", default=1)
@click.option("--config-name", default="modded_dac_vq")
@click.option(
    "--checkpoint-path",
    default="checkpoints/openaudio-s1-mini/codec.pth",
)
@click.option("--batch-size", default=64)
@click.option("--filelist", default=None, type=Path)
def main(
    folder: str,
    num_workers: int,
    config_name: str,
    checkpoint_path: str,
    batch_size: int,
    filelist: Path,
):
    if num_workers > 1 and WORLD_SIZE != num_workers:
        assert WORLD_SIZE == 1, "You should either use SLURM or this launcher, not both"

        logger.info(f"Spawning {num_workers} workers")

        if torch.cuda.is_available():
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if visible_devices is None:
                visible_devices = list(range(torch.cuda.device_count()))
            else:
                visible_devices = visible_devices.split(",")
        else:
            # Set to empty string to avoid using GPU
            visible_devices = [""]

        processes = []
        for i in range(num_workers):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(visible_devices[i % len(visible_devices)])
            env["SLURM_PROCID"] = str(i)
            env["SLURM_NTASKS"] = str(num_workers)

            processes.append(
                sp.Popen(
                    [sys.executable] + sys.argv.copy(),
                    env=env,
                )
            )

        for p in processes:
            p.wait()

        logger.info(f"All workers finished")
        return

    # This is a worker
    logger.info(f"Starting worker")
    if filelist:
        files = [i[0] for i in load_filelist(filelist)]
    else:
        files = list_files(folder, AUDIO_EXTENSIONS, recursive=True, sort=False)

    print(f"Found {len(files)} files")
    files = [Path(f) for f in files if not Path(f).with_suffix(".npy").exists()]

    total_files = len(files)
    files = files[RANK::WORLD_SIZE]
    logger.info(f"Processing {len(files)}/{total_files} files")

    # Log initial GPU state
    if torch.cuda.is_available():
        logger.info(f"[INIT DEBUG] Initial GPU state: {_get_gpu_memory_info()}")
        logger.info(f"[INIT DEBUG] Batch size: {batch_size}, Num workers: {num_workers}")

    # Batch processing
    total_time = 0
    begin_time = time.time()
    processed_files = 0
    model = get_model(config_name, checkpoint_path)

    for n_batch, idx in enumerate(range(0, len(files), batch_size)):
        batch = files[idx : idx + batch_size]
        batch_time = process_batch(batch, model, batch_num=n_batch)

        total_time += batch_time
        processed_files += len(batch)

        if (n_batch + 1) % 10 == 0:
            eta = (
                (time.time() - begin_time)
                / processed_files
                * (len(files) - processed_files)
            )
            logger.info(
                f"Processed {processed_files} files, {total_time / 3600:.2f} hours of audio, "
                + f"ETA: {timedelta(seconds=round(eta))}s"
            )

    logger.info(
        f"Finished processing {len(files)} files, {total_time / 3600:.2f} hours of audio"
    )


if __name__ == "__main__":
    main()
