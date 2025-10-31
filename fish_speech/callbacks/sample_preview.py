from __future__ import annotations

from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
from lightning.pytorch.callbacks import Callback
from loguru import logger

from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import (
    GenerateResponse,
    decode_one_token_ar,
    generate_long,
)


class SampleGenerationCallback(Callback):
    """Generate audio previews from the current LoRA weights during training."""

    def __init__(
        self,
        sample_text: str,
        output_dir: str,
        decoder_config_name: str = "modded_dac_vq",
        decoder_checkpoint_path: str = "checkpoints/openaudio-s1-mini/codec.pth",
        every_n_steps: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.7,
        repetition_penalty: float = 1.5,
        max_new_tokens: int = 512,
        chunk_length: int = 300,
        sample_prefix: str = "step",
    ) -> None:
        super().__init__()
        self.sample_text = sample_text
        self.output_dir = Path(output_dir)
        self.decoder_config_name = decoder_config_name
        self.decoder_checkpoint_path = decoder_checkpoint_path
        self.every_n_steps = max(1, every_n_steps)
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.chunk_length = chunk_length
        self.sample_prefix = sample_prefix

        self._decoder_model: Optional[torch.nn.Module] = None
        self._decoder_device: Optional[torch.device] = None
        self._last_step: int = -1

    def on_fit_start(self, trainer, pl_module) -> None:
        if trainer.is_global_zero:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_end(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return

        global_step = trainer.global_step
        if global_step <= 0:
            return

        if global_step == self._last_step:
            return

        if global_step % self.every_n_steps != 0:
            return

        try:
            self._ensure_decoder(pl_module)
        except Exception as exc:
            logger.exception(f"[SampleGeneration] Failed to prepare decoder: {exc}")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._generate_preview(pl_module, step=global_step)

    def _ensure_decoder(self, pl_module) -> None:
        if self._decoder_model is not None:
            return

        device = pl_module.device
        logger.info(
            f"[SampleGeneration] Loading decoder on device {device} "
            f"using weights {self.decoder_checkpoint_path}"
        )
        self._decoder_model = load_decoder_model(
            self.decoder_config_name,
            self.decoder_checkpoint_path,
            device=str(device),
        )
        self._decoder_device = device

    def _generate_preview(self, pl_module, step: int) -> None:
        model = pl_module.model
        was_training = model.training

        sample_path = self.output_dir / f"{self.sample_prefix}_{step:09d}.wav"

        with torch.no_grad():
            try:
                model.eval()
                # Force cache rebuild for autoregressive generation.
                if hasattr(model, "_cache_setup_done"):
                    model._cache_setup_done = False

                generator = generate_long(
                    model=model,
                    device=pl_module.device,
                    decode_one_token=decode_one_token_ar,
                    text=self.sample_text,
                    num_samples=1,
                    max_new_tokens=self.max_new_tokens,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    temperature=self.temperature,
                    chunk_length=self.chunk_length,
                )

                code_segments = []
                for response in generator:
                    if (
                        isinstance(response, GenerateResponse)
                        and response.action == "sample"
                        and response.codes is not None
                    ):
                        code_segments.append(response.codes.detach().cpu())

                if not code_segments:
                    logger.warning(
                        f"[SampleGeneration] No codes produced for step {step}, "
                        "skipping preview."
                    )
                    return

                codes = torch.cat(code_segments, dim=1)
                indices = codes.to(self._decoder_device).long()
                if indices.dim() == 3:
                    indices = indices.squeeze(0)
                indices = indices.unsqueeze(0)

                indices_lens = torch.tensor(
                    [indices.shape[-1]],
                    device=self._decoder_device,
                    dtype=torch.long,
                )

                generated_audio, _ = self._decoder_model.decode(indices, indices_lens)
                audio = generated_audio[0, 0].float().cpu().numpy()

                sf.write(sample_path, audio, self._decoder_model.sample_rate)
                logger.info(
                    f"[SampleGeneration] Saved preview for step {step} to {sample_path}"
                )
                self._last_step = step
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception(
                    f"[SampleGeneration] Failed to generate preview at step {step}: {exc}"
                )
            finally:
                if was_training:
                    model.train()
