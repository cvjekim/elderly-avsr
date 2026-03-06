"""LLaMA-AVSR: Audio-Visual Speech Recognition with LLaMA decoder.

Reference: https://github.com/umbertocappellazzo/Llama-AVSR
Architecture:
  - Visual encoder: AV-HuBERT (from local .pt or HF)
  - Audio encoder: AV-HuBERT audio stream or Wav2Vec2
  - LLM decoder: meta-llama/Llama-2-7b-hf
  - Linear projection layers bridging encoder to LLM embedding space
  - Length adapter (conv-based downsampling) to reduce encoder sequence length
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    PreTrainedModel,
    PretrainedConfig,
    AutoModel,
)


class LLaMAVSRConfig(PretrainedConfig):
    model_type = "llama_avsr"

    def __init__(
        self,
        visual_encoder_path: str = "nguyenvulebinh/AV-HuBERT",
        audio_encoder_path: str = "nguyenvulebinh/AV-HuBERT",
        llm_id: str = "meta-llama/Llama-2-7b-hf",
        encoder_hidden_size: int = 1024,
        llm_hidden_size: int = 4096,
        projection_hidden_size: int = 2048,
        length_adapter_kernel: int = 3,
        length_adapter_stride: int = 2,
        max_new_tokens: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.visual_encoder_path = visual_encoder_path
        self.audio_encoder_path = audio_encoder_path
        self.llm_id = llm_id
        self.encoder_hidden_size = encoder_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.projection_hidden_size = projection_hidden_size
        self.length_adapter_kernel = length_adapter_kernel
        self.length_adapter_stride = length_adapter_stride
        self.max_new_tokens = max_new_tokens


class LengthAdapter(nn.Module):
    """Conv1D-based downsampling to reduce encoder sequence length before LLM."""

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 3,
        stride: int = 2,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            hidden_size, hidden_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            (batch, seq_len // stride, hidden_size)
        """
        # - Conv1d expects (B, C, L)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        return self.norm(x)


class EncoderProjection(nn.Module):
    """Project encoder hidden states to LLM embedding space."""

    def __init__(
        self,
        encoder_hidden_size: int,
        llm_hidden_size: int,
        projection_hidden_size: int,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(encoder_hidden_size, projection_hidden_size),
            nn.GELU(),
            nn.Linear(projection_hidden_size, llm_hidden_size),
        )
        self.norm = nn.LayerNorm(llm_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class AVEncoder(nn.Module):
    """Shared AV-HuBERT encoder wrapper for both audio and visual streams.

    Loads from:
      - Local .pt fairseq checkpoint
      - HF repo ID (nguyenvulebinh/AV-HuBERT)
      - Local directory with safetensors/bin
    """

    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = None

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "AVEncoder":
        instance = cls()
        path = Path(model_name_or_path)

        if path.is_file() and path.suffix == ".pt":
            # - Load fairseq-style .pt checkpoint
            ckpt = torch.load(str(path), map_location="cpu")
            if "model" in ckpt:
                state_dict = ckpt["model"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt

            # - Build transformer encoder matching AV-HuBERT
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=instance.hidden_size,
                nhead=16,
                dim_feedforward=instance.hidden_size * 4,
                batch_first=True,
            )
            instance.encoder = nn.TransformerEncoder(encoder_layer, num_layers=24)

            compatible = {
                k.replace("encoder.", "", 1): v
                for k, v in state_dict.items()
                if k.startswith("encoder.")
            }
            if compatible:
                instance.encoder.load_state_dict(compatible, strict=False)
        elif path.is_dir():
            instance.encoder = AutoModel.from_pretrained(str(path))
            instance.hidden_size = instance.encoder.config.hidden_size
        else:
            # - HuggingFace repo ID
            instance.encoder = AutoModel.from_pretrained(model_name_or_path)
            instance.hidden_size = instance.encoder.config.hidden_size

        return instance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder is None:
            raise RuntimeError("Encoder not loaded. Call from_pretrained() first.")

        if hasattr(self.encoder, "config"):
            outputs = self.encoder(x)
            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state
            return outputs[0]

        return self.encoder(x)


class LLaMAVSR(PreTrainedModel):
    config_class = LLaMAVSRConfig

    def __init__(self, config: LLaMAVSRConfig):
        super().__init__(config)
        self.config = config

        # - Sub-components (loaded via from_pretrained)
        self.audio_encoder: Optional[AVEncoder] = None
        self.visual_encoder: Optional[AVEncoder] = None
        self.llm: Optional[LlamaForCausalLM] = None
        self.tokenizer: Optional[LlamaTokenizer] = None

        # - Length adapters (reduce encoder seq length)
        self.audio_length_adapter = LengthAdapter(
            config.encoder_hidden_size,
            config.length_adapter_kernel,
            config.length_adapter_stride,
        )
        self.visual_length_adapter = LengthAdapter(
            config.encoder_hidden_size,
            config.length_adapter_kernel,
            config.length_adapter_stride,
        )

        # - Projection layers
        self.audio_projection = EncoderProjection(
            config.encoder_hidden_size,
            config.llm_hidden_size,
            config.projection_hidden_size,
        )
        self.visual_projection = EncoderProjection(
            config.encoder_hidden_size,
            config.llm_hidden_size,
            config.projection_hidden_size,
        )

        # - Modality fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(config.llm_hidden_size * 2, config.llm_hidden_size),
            nn.Sigmoid(),
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        visual_encoder_path: str = "nguyenvulebinh/AV-HuBERT",
        audio_encoder_path: str = "nguyenvulebinh/AV-HuBERT",
        llm_id: str = "meta-llama/Llama-2-7b-hf",
        config: Optional[LLaMAVSRConfig] = None,
        **kwargs,
    ) -> "LLaMAVSR":
        """Unified loading interface.

        Args:
            model_name_or_path: Local dir with custom adapter/projection weights,
                                or path to a single checkpoint file.
            visual_encoder_path: HF repo or local .pt for visual encoder.
            audio_encoder_path: HF repo or local .pt for audio encoder.
            llm_id: HF repo ID or local path for LLaMA.
            config: Optional config override.
        """
        if config is None:
            config = LLaMAVSRConfig(
                visual_encoder_path=visual_encoder_path,
                audio_encoder_path=audio_encoder_path,
                llm_id=llm_id,
            )

        model = cls(config)

        # - Load encoders
        model.visual_encoder = AVEncoder.from_pretrained(visual_encoder_path)
        model.audio_encoder = AVEncoder.from_pretrained(audio_encoder_path)

        # - Update hidden sizes and rebuild projections if needed
        config.encoder_hidden_size = model.visual_encoder.hidden_size
        model.audio_projection = EncoderProjection(
            config.encoder_hidden_size,
            config.llm_hidden_size,
            config.projection_hidden_size,
        )
        model.visual_projection = EncoderProjection(
            config.encoder_hidden_size,
            config.llm_hidden_size,
            config.projection_hidden_size,
        )
        model.audio_length_adapter = LengthAdapter(
            config.encoder_hidden_size,
            config.length_adapter_kernel,
            config.length_adapter_stride,
        )
        model.visual_length_adapter = LengthAdapter(
            config.encoder_hidden_size,
            config.length_adapter_kernel,
            config.length_adapter_stride,
        )

        # - Load LLM
        model.llm = LlamaForCausalLM.from_pretrained(llm_id, **kwargs)
        model.tokenizer = LlamaTokenizer.from_pretrained(llm_id)

        # - Load custom adapter/projection weights if available
        local_path = Path(model_name_or_path)
        if local_path.is_dir():
            for candidate in ["llama_avsr_weights.bin", "pytorch_model.bin", "adapter_model.bin"]:
                ckpt_file = local_path / candidate
                if ckpt_file.exists():
                    custom_state = torch.load(str(ckpt_file), map_location="cpu")
                    adapter_keys = {
                        k: v for k, v in custom_state.items()
                        if any(p in k for p in [
                            "audio_projection", "visual_projection",
                            "audio_length_adapter", "visual_length_adapter",
                            "fusion_gate",
                        ])
                    }
                    if adapter_keys:
                        model.load_state_dict(adapter_keys, strict=False)
                        print(f"Loaded {len(adapter_keys)} adapter keys from {ckpt_file}")
                    break
        elif local_path.is_file():
            custom_state = torch.load(str(local_path), map_location="cpu")
            adapter_keys = {
                k: v for k, v in custom_state.items()
                if any(p in k for p in [
                    "audio_projection", "visual_projection",
                    "audio_length_adapter", "visual_length_adapter",
                    "fusion_gate",
                ])
            }
            if adapter_keys:
                model.load_state_dict(adapter_keys, strict=False)

        return model

    def _fuse_modalities(
        self,
        audio_embeds: torch.Tensor,
        visual_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Gated fusion of audio and visual embeddings."""
        if visual_embeds is None:
            return audio_embeds

        # - Align sequence lengths
        min_len = min(audio_embeds.size(1), visual_embeds.size(1))
        audio_embeds = audio_embeds[:, :min_len, :]
        visual_embeds = visual_embeds[:, :min_len, :]

        concat = torch.cat([audio_embeds, visual_embeds], dim=-1)
        gate = self.fusion_gate(concat)
        return gate * audio_embeds + (1 - gate) * visual_embeds

    def forward(
        self,
        audio_values: torch.Tensor,
        video_pixels: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Training forward pass with teacher forcing.

        Args:
            audio_values: (batch, seq_len, feat_dim) or raw features for encoder
            video_pixels: (batch, time, C, H, W) or None
            labels: (batch, seq_len) token IDs for teacher forcing
            attention_mask: (batch, seq_len) mask for labels
        """
        # - Encode audio
        audio_out = self.audio_encoder(audio_values)
        audio_out = self.audio_length_adapter(audio_out)
        audio_embeds = self.audio_projection(audio_out)

        # - Encode video
        visual_embeds = None
        if video_pixels is not None and self.visual_encoder is not None:
            visual_out = self.visual_encoder(video_pixels)
            visual_out = self.visual_length_adapter(visual_out)
            visual_embeds = self.visual_projection(visual_out)

        # - Fuse modalities
        encoder_embeds = self._fuse_modalities(audio_embeds, visual_embeds)

        # - Prepare LLM inputs
        if labels is not None:
            label_embeds = self.llm.model.embed_tokens(labels)
            inputs_embeds = torch.cat([encoder_embeds, label_embeds], dim=1)

            # - Labels: -100 for encoder positions
            encoder_len = encoder_embeds.size(1)
            ignore_labels = torch.full(
                (labels.size(0), encoder_len),
                fill_value=-100,
                dtype=labels.dtype,
                device=labels.device,
            )
            shifted_labels = torch.cat([ignore_labels, labels], dim=1)

            # - Attention mask
            encoder_attn = torch.ones(
                labels.size(0), encoder_len,
                dtype=torch.long, device=labels.device,
            )
            if attention_mask is not None:
                full_attn_mask = torch.cat([encoder_attn, attention_mask], dim=1)
            else:
                full_attn_mask = torch.cat([encoder_attn, torch.ones_like(labels)], dim=1)

            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attn_mask,
                labels=shifted_labels,
            )
            return {"loss": outputs.loss, "logits": outputs.logits}

        outputs = self.llm(inputs_embeds=encoder_embeds)
        return {"logits": outputs.logits}

    @torch.no_grad()
    def generate(
        self,
        audio_values: torch.Tensor,
        video_pixels: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        **generate_kwargs,
    ) -> dict[str, list[str]]:
        """Auto-regressive inference."""
        max_new_tokens = max_new_tokens or self.config.max_new_tokens

        # - Encode audio
        audio_out = self.audio_encoder(audio_values)
        audio_out = self.audio_length_adapter(audio_out)
        audio_embeds = self.audio_projection(audio_out)

        # - Encode video
        visual_embeds = None
        if video_pixels is not None and self.visual_encoder is not None:
            visual_out = self.visual_encoder(video_pixels)
            visual_out = self.visual_length_adapter(visual_out)
            visual_embeds = self.visual_projection(visual_out)

        # - Fuse
        encoder_embeds = self._fuse_modalities(audio_embeds, visual_embeds)

        # - Generate
        generated_ids = self.llm.generate(
            inputs_embeds=encoder_embeds,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )

        predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return {"predictions": predictions}


def build_model(config: dict) -> LLaMAVSR:
    """Entry point called by train.py / evaluate.py."""
    model_config = LLaMAVSRConfig(
        visual_encoder_path=config.get("visual_encoder_path", "nguyenvulebinh/AV-HuBERT"),
        audio_encoder_path=config.get("audio_encoder_path", "nguyenvulebinh/AV-HuBERT"),
        llm_id=config.get("llm_id", "meta-llama/Llama-2-7b-hf"),
        encoder_hidden_size=config.get("encoder_hidden_size", 1024),
        llm_hidden_size=config.get("llm_hidden_size", 4096),
        projection_hidden_size=config.get("projection_hidden_size", 2048),
        length_adapter_kernel=config.get("length_adapter_kernel", 3),
        length_adapter_stride=config.get("length_adapter_stride", 2),
        max_new_tokens=config.get("max_new_tokens", 256),
    )

    pretrained_path = config.get("pretrained_path", "./models/llama_avsr/pretrained")
    return LLaMAVSR.from_pretrained(
        pretrained_path,
        visual_encoder_path=model_config.visual_encoder_path,
        audio_encoder_path=model_config.audio_encoder_path,
        llm_id=model_config.llm_id,
        config=model_config,
    )
