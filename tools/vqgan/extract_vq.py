import os
import subprocess as sp
import sys
import time
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import click
import numpy as np
import torch
import torchaudio
from hydra import compose, initialize
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

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
logger.add(sys.stderr, format=logger_format, level="ERROR")

_GLOBAL_CONFIG_NAME: str | None = None
_GLOBAL_CHECKPOINT_PATH: str | None = None


@lru_cache(maxsize=None)
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

    return model


@torch.inference_mode()
def process_batch(files: list[Path], model, batch_num: int = 0) -> float:
    return _process_batch_internal(files, model, batch_num)


def _model_device(model: torch.nn.Module) -> torch.device:
    for param in model.parameters():
        return param.device
    return torch.device("cpu")


def _process_batch_internal(
    files: list[Path],
    model: torch.nn.Module,
    batch_num: int = 0,
    allow_split: bool = True,
) -> float:
    try:
        return _process_batch_once(files, model, batch_num)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        error_msg = str(exc).lower()
        if not isinstance(exc, torch.cuda.OutOfMemoryError) and "out of memory" not in error_msg:
            raise

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if len(files) > 1 and allow_split:
            mid = len(files) // 2
            left = files[:mid]
            right = files[mid:]
            return _process_batch_internal(left, model, batch_num, allow_split=True) + _process_batch_internal(
                right, model, batch_num, allow_split=True
            )

        if not _GLOBAL_CONFIG_NAME or not _GLOBAL_CHECKPOINT_PATH:
            raise RuntimeError("Model metadata unavailable for CPU fallback.") from exc

        tqdm.write(
            f"GPU memory exhausted on '{files[0].name}'. Falling back to CPU for this file."
        )
        cpu_model = get_model(_GLOBAL_CONFIG_NAME, _GLOBAL_CHECKPOINT_PATH, device="cpu")
        return _process_batch_once(files, cpu_model, batch_num)


def _process_batch_once(files: list[Path], model: torch.nn.Module, batch_num: int = 0) -> float:
    device = _model_device(model)
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

        wav = wav.to(device)
        wav = torchaudio.functional.resample(wav, sr, model.sample_rate)[0]
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

    audios = torch.stack(wavs, dim=0)[:, None].to(device)
    audio_lengths_tensor = torch.tensor(audio_lengths, device=device, dtype=torch.long)

    indices, feature_lengths = model.encode(audios, audio_lengths_tensor)

    outputs = indices.detach().cpu().numpy()
    feature_lengths = feature_lengths.detach().cpu().numpy()

    for file, length, feature in zip(files, feature_lengths, outputs):
        feature = feature[:, :length]

        # (T,)
        with open(file.with_suffix(".npy"), "wb") as f:
            np.save(f, feature)

    if device.type == "cuda":
        torch.cuda.empty_cache()

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
    global _GLOBAL_CONFIG_NAME, _GLOBAL_CHECKPOINT_PATH
    _GLOBAL_CONFIG_NAME = config_name
    _GLOBAL_CHECKPOINT_PATH = checkpoint_path

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
    if filelist:
        files = [i[0] for i in load_filelist(filelist)]
    else:
        files = list_files(folder, AUDIO_EXTENSIONS, recursive=True, sort=False)

    files = [Path(f) for f in files if not Path(f).with_suffix(".npy").exists()]

    total_files = len(files)
    files = files[RANK::WORLD_SIZE]

    # Batch processing
    total_time = 0
    begin_time = time.time()
    processed_files = 0
    model = get_model(config_name, checkpoint_path)

    show_progress = WORLD_SIZE == 1
    progress = tqdm(
        total=len(files),
        unit="file",
        dynamic_ncols=True,
        disable=not show_progress,
        desc="Extracting tokens",
    )

    for n_batch, idx in enumerate(range(0, len(files), batch_size)):
        batch = files[idx : idx + batch_size]
        batch_time = process_batch(batch, model, batch_num=n_batch)

        total_time += batch_time
        processed_files += len(batch)

        if show_progress:
            progress.update(len(batch))
            progress.set_postfix({"hours": f"{total_time / 3600:.2f}"})

    if show_progress:
        progress.close()

    elapsed = timedelta(seconds=round(time.time() - begin_time))
    tqdm.write(
        f"Finished processing {processed_files} files "
        f"({total_time / 3600:.2f} hours audio) in {elapsed}."
    )


if __name__ == "__main__":
    main()
