"""MMS-LLaMA: Audio-Visual Speech Recognition with MMS encoder + LLaMA decoder.

Reference: https://github.com/JeongHun0716/MMS-LLaMA
Architecture:
  - Audio encoder: facebook/mms-1b-all (Wav2Vec2-based)
  - Visual encoder: AV-HuBERT encoder (from local .pt or HF)
  - LLM decoder: meta-llama/Llama-2-7b-hf
  - Projection layers bridging encoder outputs to LLM embedding space
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    Wav2Vec2Model,
    Wav2Vec2Processor,
    PreTrainedModel,
    PretrainedConfig,
)


class MMSLLaMAConfig(PretrainedConfig):
    model_type = "mms_llama"

    def __init__(
        self,
        audio_encoder_id: str = "facebook/mms-1b-all",
        visual_encoder_path: str = "",
        llm_id: str = "meta-llama/Llama-2-7b-hf",
        audio_hidden_size: int = 1280,
        visual_hidden_size: int = 1024,
        llm_hidden_size: int = 4096,
        projection_hidden_size: int = 2048,
        max_new_tokens: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_encoder_id = audio_encoder_id
        self.visual_encoder_path = visual_encoder_path
        self.llm_id = llm_id
        self.audio_hidden_size = audio_hidden_size
        self.visual_hidden_size = visual_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.projection_hidden_size = projection_hidden_size
        self.max_new_tokens = max_new_tokens


class AudioProjection(nn.Module):
    """Project audio encoder hidden states to LLM embedding space."""

    def __init__(self, audio_hidden_size: int, llm_hidden_size: int, projection_hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(audio_hidden_size, projection_hidden_size),
            nn.GELU(),
            nn.Linear(projection_hidden_size, llm_hidden_size),
        )
        self.norm = nn.LayerNorm(llm_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class VisualProjection(nn.Module):
    """Project visual encoder hidden states to LLM embedding space."""

    def __init__(self, visual_hidden_size: int, llm_hidden_size: int, projection_hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(visual_hidden_size, projection_hidden_size),
            nn.GELU(),
            nn.Linear(projection_hidden_size, llm_hidden_size),
        )
        self.norm = nn.LayerNorm(llm_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class VisualEncoder(nn.Module):
    """AV-HuBERT visual encoder wrapper.

    Loads from either:
      - A local .pt fairseq checkpoint (torch.load inside from_pretrained)
      - A HuggingFace repo (nguyenvulebinh/AV-HuBERT)
    """

    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.hidden_size = hidden_size
        # - Placeholder: actual encoder layers loaded in from_pretrained
        self.encoder = None

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "VisualEncoder":
        instance = cls()
        path = Path(model_name_or_path)

        if path.is_file() and path.suffix == ".pt":
            # - Load fairseq-style .pt checkpoint
            ckpt = torch.load(str(path), map_location="cpu")
            # - Extract encoder state dict from fairseq checkpoint structure
            if "model" in ckpt:
                state_dict = ckpt["model"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt

            # - Build a simple transformer encoder matching AV-HuBERT architecture
            instance.encoder = cls._build_avhubert_encoder(state_dict, instance.hidden_size)
        elif path.is_dir():
            # - Load from local directory with safetensors/bin
            from transformers import AutoModel
            instance.encoder = AutoModel.from_pretrained(str(path))
            instance.hidden_size = instance.encoder.config.hidden_size
        else:
            # - Treat as HuggingFace repo ID
            from transformers import AutoModel
            instance.encoder = AutoModel.from_pretrained(model_name_or_path)
            instance.hidden_size = instance.encoder.config.hidden_size

        return instance

    @staticmethod
    def _build_avhubert_encoder(state_dict: dict, hidden_size: int) -> nn.Module:
        """Build encoder from fairseq state dict.

        This is a simplified reconstruction; adapt layer mapping
        based on the actual AV-HuBERT checkpoint structure.
        """
        # - Use nn.TransformerEncoder as a stand-in for the fairseq encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=16,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
        )
        encoder = nn.TransformerEncoder(encoder_layer, num_layers=24)

        # - Attempt to load matching keys
        compatible_state = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                new_key = k.replace("encoder.", "", 1)
                compatible_state[new_key] = v

        if compatible_state:
            missing, unexpected = encoder.load_state_dict(compatible_state, strict=False)
            if missing:
                print(f"VisualEncoder: {len(missing)} missing keys (expected for architecture mismatch)")

        return encoder

    def forward(self, video_pixels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_pixels: (batch, time, channels, height, width) or encoder-ready features
        Returns:
            hidden_states: (batch, seq_len, hidden_size)
        """
        if self.encoder is None:
            raise RuntimeError("Visual encoder not loaded. Call from_pretrained() first.")

        # - If encoder is a HF model, use its forward
        if hasattr(self.encoder, "config"):
            outputs = self.encoder(video_pixels)
            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state
            return outputs[0]

        # - For raw TransformerEncoder
        return self.encoder(video_pixels)


class MMSLLaMA(PreTrainedModel):
    config_class = MMSLLaMAConfig

    def __init__(self, config: MMSLLaMAConfig):
        super().__init__(config)
        self.config = config

        # - Sub-components initialized as None; loaded via from_pretrained
        self.audio_encoder: Optional[Wav2Vec2Model] = None
        self.visual_encoder: Optional[VisualEncoder] = None
        self.llm: Optional[LlamaForCausalLM] = None
        self.tokenizer: Optional[LlamaTokenizer] = None

        # - Projection layers (always initialized)
        self.audio_projection = AudioProjection(
            config.audio_hidden_size,
            config.llm_hidden_size,
            config.projection_hidden_size,
        )
        self.visual_projection = VisualProjection(
            config.visual_hidden_size,
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
        audio_encoder_id: str = "facebook/mms-1b-all",
        visual_encoder_path: str = "",
        llm_id: str = "meta-llama/Llama-2-7b-hf",
        config: Optional[MMSLLaMAConfig] = None,
        **kwargs,
    ) -> "MMSLLaMA":
        """Unified loading interface.

        Args:
            model_name_or_path: Path to local pretrained dir with custom weights,
                                or HF repo ID. If local dir contains mms_llama_pretrained.bin,
                                it loads projection/adapter weights from there.
            audio_encoder_id: HF repo ID or local path for MMS audio encoder.
            visual_encoder_path: Local .pt path or HF repo for visual encoder.
            llm_id: HF repo ID or local path for LLaMA decoder.
            config: Optional config override.
        """
        if config is None:
            config = MMSLLaMAConfig(
                audio_encoder_id=audio_encoder_id,
                visual_encoder_path=visual_encoder_path,
                llm_id=llm_id,
            )

        model = cls(config)

        # - Load audio encoder (MMS / Wav2Vec2)
        model.audio_encoder = Wav2Vec2Model.from_pretrained(audio_encoder_id)
        config.audio_hidden_size = model.audio_encoder.config.hidden_size
        # - Rebuild audio projection if hidden size changed
        model.audio_projection = AudioProjection(
            config.audio_hidden_size,
            config.llm_hidden_size,
            config.projection_hidden_size,
        )

        # - Load visual encoder
        if visual_encoder_path:
            model.visual_encoder = VisualEncoder.from_pretrained(visual_encoder_path)
            config.visual_hidden_size = model.visual_encoder.hidden_size
            model.visual_projection = VisualProjection(
                config.visual_hidden_size,
                config.llm_hidden_size,
                config.projection_hidden_size,
            )

        # - Load LLM decoder
        model.llm = LlamaForCausalLM.from_pretrained(llm_id, **kwargs)
        model.tokenizer = LlamaTokenizer.from_pretrained(llm_id)

        # - Load custom projection/adapter weights if available
        local_path = Path(model_name_or_path)
        custom_weights_file = local_path / "mms_llama_pretrained.bin"
        if local_path.is_dir() and custom_weights_file.exists():
            custom_state = torch.load(str(custom_weights_file), map_location="cpu")
            # - Load projection and fusion gate weights
            proj_keys = {
                k: v for k, v in custom_state.items()
                if any(prefix in k for prefix in ["audio_projection", "visual_projection", "fusion_gate"])
            }
            if proj_keys:
                missing, unexpected = model.load_state_dict(proj_keys, strict=False)
                print(f"Loaded {len(proj_keys)} custom weight keys from {custom_weights_file}")
        elif local_path.is_file():
            custom_state = torch.load(str(local_path), map_location="cpu")
            proj_keys = {
                k: v for k, v in custom_state.items()
                if any(prefix in k for prefix in ["audio_projection", "visual_projection", "fusion_gate"])
            }
            if proj_keys:
                model.load_state_dict(proj_keys, strict=False)

        return model

    def _fuse_modalities(
        self,
        audio_embeds: torch.Tensor,
        visual_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Fuse audio and visual embeddings via gated mechanism."""
        if visual_embeds is None:
            return audio_embeds

        # - Align sequence lengths (truncate to shorter)
        min_len = min(audio_embeds.size(1), visual_embeds.size(1))
        audio_embeds = audio_embeds[:, :min_len, :]
        visual_embeds = visual_embeds[:, :min_len, :]

        # - Gated fusion
        concat = torch.cat([audio_embeds, visual_embeds], dim=-1)
        gate = self.fusion_gate(concat)
        fused = gate * audio_embeds + (1 - gate) * visual_embeds
        return fused

    def forward(
        self,
        audio_values: torch.Tensor,
        video_pixels: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Training forward pass with teacher forcing.

        Args:
            audio_values: (batch, audio_samples) raw waveform at 16kHz
            video_pixels: (batch, time, C, H, W) or None for audio-only
            labels: (batch, seq_len) token IDs for teacher forcing
            attention_mask: (batch, seq_len) optional attention mask for labels
        Returns:
            dict with 'loss' and 'logits'
        """
        # - Encode audio
        audio_outputs = self.audio_encoder(audio_values).last_hidden_state
        audio_embeds = self.audio_projection(audio_outputs)

        # - Encode video (if provided)
        visual_embeds = None
        if video_pixels is not None and self.visual_encoder is not None:
            visual_outputs = self.visual_encoder(video_pixels)
            visual_embeds = self.visual_projection(visual_outputs)

        # - Fuse modalities
        encoder_embeds = self._fuse_modalities(audio_embeds, visual_embeds)

        # - Prepare LLM inputs: concatenate encoder embeddings with label embeddings
        if labels is not None:
            label_embeds = self.llm.model.embed_tokens(labels)
            # - Concat: [encoder_embeds, label_embeds]
            inputs_embeds = torch.cat([encoder_embeds, label_embeds], dim=1)

            # - Build target labels: -100 for encoder positions, actual labels for decoder
            encoder_len = encoder_embeds.size(1)
            ignore_labels = torch.full(
                (labels.size(0), encoder_len),
                fill_value=-100,
                dtype=labels.dtype,
                device=labels.device,
            )
            shifted_labels = torch.cat([ignore_labels, labels], dim=1)

            # - Build attention mask
            encoder_attn = torch.ones(
                labels.size(0), encoder_len,
                dtype=torch.long,
                device=labels.device,
            )
            if attention_mask is not None:
                full_attn_mask = torch.cat([encoder_attn, attention_mask], dim=1)
            else:
                full_attn_mask = torch.cat(
                    [encoder_attn, torch.ones_like(labels)], dim=1
                )

            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attn_mask,
                labels=shifted_labels,
            )
            return {"loss": outputs.loss, "logits": outputs.logits}

        # - No labels: return encoder embeddings passed through LLM
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
        """Auto-regressive inference.

        Args:
            audio_values: (batch, audio_samples) raw waveform
            video_pixels: (batch, time, C, H, W) or None
            max_new_tokens: override config max_new_tokens
        Returns:
            dict with 'predictions' as list of decoded strings
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens

        # - Encode audio
        audio_outputs = self.audio_encoder(audio_values).last_hidden_state
        audio_embeds = self.audio_projection(audio_outputs)

        # - Encode video
        visual_embeds = None
        if video_pixels is not None and self.visual_encoder is not None:
            visual_outputs = self.visual_encoder(video_pixels)
            visual_embeds = self.visual_projection(visual_outputs)

        # - Fuse
        encoder_embeds = self._fuse_modalities(audio_embeds, visual_embeds)

        # - Generate with LLM
        generated_ids = self.llm.generate(
            inputs_embeds=encoder_embeds,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )

        # - Decode tokens to strings
        predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return {"predictions": predictions}


def build_model(config: dict) -> MMSLLaMA:
    """Entry point called by train.py / evaluate.py."""
    model_config = MMSLLaMAConfig(
        audio_encoder_id=config.get("audio_encoder_id", "facebook/mms-1b-all"),
        visual_encoder_path=config.get("visual_encoder_path", ""),
        llm_id=config.get("llm_id", "meta-llama/Llama-2-7b-hf"),
        audio_hidden_size=config.get("audio_hidden_size", 1280),
        visual_hidden_size=config.get("visual_hidden_size", 1024),
        llm_hidden_size=config.get("llm_hidden_size", 4096),
        projection_hidden_size=config.get("projection_hidden_size", 2048),
        max_new_tokens=config.get("max_new_tokens", 256),
    )

    pretrained_path = config.get("pretrained_path", "")
    if pretrained_path:
        return MMSLLaMA.from_pretrained(
            pretrained_path,
            audio_encoder_id=model_config.audio_encoder_id,
            visual_encoder_path=model_config.visual_encoder_path,
            llm_id=model_config.llm_id,
            config=model_config,
        )

    # - Build from scratch (load sub-components individually)
    return MMSLLaMA.from_pretrained(
        "./models/mms_llama/pretrained",
        audio_encoder_id=model_config.audio_encoder_id,
        visual_encoder_path=model_config.visual_encoder_path,
        llm_id=model_config.llm_id,
        config=model_config,
    )
