"""Whisper-Flamingo / mWhisper-Flamingo: Audio-Visual Speech Recognition.

Reference: https://github.com/roudimit/whisper-flamingo
Architecture:
  - Audio encoder: OpenAI Whisper (openai/whisper-large-v2) encoder
  - Visual encoder: AV-HuBERT (from local .pt or HF)
  - Decoder: Whisper decoder with injected gated cross-attention layers
    (Flamingo-style) that attend to visual encoder outputs
  - The gated cross-attention layers are interleaved with the original
    Whisper decoder layers
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    WhisperModel,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    PreTrainedModel,
    PretrainedConfig,
    AutoModel,
)


class WhisperFlamingoConfig(PretrainedConfig):
    model_type = "whisper_flamingo"

    def __init__(
        self,
        whisper_id: str = "openai/whisper-large-v2",
        visual_encoder_path: str = "nguyenvulebinh/AV-HuBERT",
        whisper_hidden_size: int = 1280,
        visual_hidden_size: int = 1024,
        num_gated_xattn_layers: int = 4,
        xattn_heads: int = 16,
        max_new_tokens: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.whisper_id = whisper_id
        self.visual_encoder_path = visual_encoder_path
        self.whisper_hidden_size = whisper_hidden_size
        self.visual_hidden_size = visual_hidden_size
        self.num_gated_xattn_layers = num_gated_xattn_layers
        self.xattn_heads = xattn_heads
        self.max_new_tokens = max_new_tokens


class VisualProjection(nn.Module):
    """Project visual encoder outputs to Whisper decoder hidden size."""

    def __init__(self, visual_hidden_size: int, whisper_hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(visual_hidden_size, whisper_hidden_size),
            nn.GELU(),
            nn.Linear(whisper_hidden_size, whisper_hidden_size),
        )
        self.norm = nn.LayerNorm(whisper_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class GatedCrossAttentionLayer(nn.Module):
    """Flamingo-style gated cross-attention layer.

    Inserted between Whisper decoder layers to attend to visual features.
    Uses a learnable tanh gate initialized near zero for stable training.
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.ff_norm = nn.LayerNorm(hidden_size)

        # - Learnable gate initialized near zero for stable training
        self.gate_attn = nn.Parameter(torch.tensor(0.0))
        self.gate_ff = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        x: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size) decoder hidden states
            visual_features: (batch, vis_len, hidden_size) projected visual features
        Returns:
            (batch, seq_len, hidden_size)
        """
        # - Gated cross-attention
        residual = x
        x_norm = self.norm(x)
        attn_out, _ = self.cross_attn(
            query=x_norm,
            key=visual_features,
            value=visual_features,
        )
        x = residual + torch.tanh(self.gate_attn) * attn_out

        # - Gated feed-forward
        residual = x
        x = residual + torch.tanh(self.gate_ff) * self.ff(self.ff_norm(x))
        return x


class VisualEncoder(nn.Module):
    """AV-HuBERT visual encoder wrapper."""

    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = None

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "VisualEncoder":
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
            instance.encoder = AutoModel.from_pretrained(model_name_or_path)
            instance.hidden_size = instance.encoder.config.hidden_size

        return instance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder is None:
            raise RuntimeError("Visual encoder not loaded.")

        if hasattr(self.encoder, "config"):
            outputs = self.encoder(x)
            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state
            return outputs[0]

        return self.encoder(x)


class WhisperFlamingo(PreTrainedModel):
    config_class = WhisperFlamingoConfig

    def __init__(self, config: WhisperFlamingoConfig):
        super().__init__(config)
        self.config = config

        # - Sub-components (loaded via from_pretrained)
        self.whisper: Optional[WhisperForConditionalGeneration] = None
        self.processor: Optional[WhisperProcessor] = None
        self.visual_encoder: Optional[VisualEncoder] = None

        # - Visual projection
        self.visual_projection = VisualProjection(
            config.visual_hidden_size,
            config.whisper_hidden_size,
        )

        # - Gated cross-attention layers (Flamingo-style)
        self.gated_xattn_layers = nn.ModuleList([
            GatedCrossAttentionLayer(config.whisper_hidden_size, config.xattn_heads)
            for _ in range(config.num_gated_xattn_layers)
        ])

        # - Mapping: which decoder layers get cross-attention injection
        # - Evenly spaced across decoder layers
        self._xattn_layer_indices: list[int] = []

    def _compute_xattn_indices(self, num_decoder_layers: int) -> list[int]:
        """Compute evenly-spaced decoder layer indices for xattn injection."""
        n = self.config.num_gated_xattn_layers
        if n >= num_decoder_layers:
            return list(range(num_decoder_layers))
        step = num_decoder_layers / n
        return [int(step * i + step / 2) for i in range(n)]

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        whisper_id: str = "openai/whisper-large-v2",
        visual_encoder_path: str = "nguyenvulebinh/AV-HuBERT",
        config: Optional[WhisperFlamingoConfig] = None,
        **kwargs,
    ) -> "WhisperFlamingo":
        """Unified loading interface.

        Args:
            model_name_or_path: Local dir with custom Flamingo weights (.pt),
                                or path to checkpoint file.
            whisper_id: HF repo ID for base Whisper model.
            visual_encoder_path: HF repo or local .pt for visual encoder.
            config: Optional config override.
        """
        if config is None:
            config = WhisperFlamingoConfig(
                whisper_id=whisper_id,
                visual_encoder_path=visual_encoder_path,
            )

        model = cls(config)

        # - Load Whisper
        model.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_id, **kwargs)
        model.processor = WhisperProcessor.from_pretrained(whisper_id)

        # - Update hidden size from loaded Whisper config
        config.whisper_hidden_size = model.whisper.config.d_model
        model.visual_projection = VisualProjection(
            config.visual_hidden_size,
            config.whisper_hidden_size,
        )
        model.gated_xattn_layers = nn.ModuleList([
            GatedCrossAttentionLayer(config.whisper_hidden_size, config.xattn_heads)
            for _ in range(config.num_gated_xattn_layers)
        ])

        # - Compute xattn injection indices
        num_decoder_layers = model.whisper.config.decoder_layers
        model._xattn_layer_indices = model._compute_xattn_indices(num_decoder_layers)

        # - Load visual encoder
        if visual_encoder_path:
            model.visual_encoder = VisualEncoder.from_pretrained(visual_encoder_path)
            config.visual_hidden_size = model.visual_encoder.hidden_size
            model.visual_projection = VisualProjection(
                config.visual_hidden_size,
                config.whisper_hidden_size,
            )

        # - Load custom Flamingo weights (gated xattn + visual projection)
        local_path = Path(model_name_or_path)
        if local_path.is_dir():
            for candidate in ["whisper_flamingo.pt", "mwhisper_flamingo.pt", "pytorch_model.bin"]:
                ckpt_file = local_path / candidate
                if ckpt_file.exists():
                    custom_state = torch.load(str(ckpt_file), map_location="cpu")
                    flamingo_keys = {
                        k: v for k, v in custom_state.items()
                        if any(p in k for p in [
                            "gated_xattn", "visual_projection",
                        ])
                    }
                    if flamingo_keys:
                        model.load_state_dict(flamingo_keys, strict=False)
                        print(f"Loaded {len(flamingo_keys)} Flamingo keys from {ckpt_file}")
                    break
        elif local_path.is_file():
            custom_state = torch.load(str(local_path), map_location="cpu")
            flamingo_keys = {
                k: v for k, v in custom_state.items()
                if any(p in k for p in ["gated_xattn", "visual_projection"])
            }
            if flamingo_keys:
                model.load_state_dict(flamingo_keys, strict=False)

        return model

    def _encode_audio(self, audio_values: torch.Tensor) -> torch.Tensor:
        """Encode audio through Whisper encoder.

        Args:
            audio_values: (batch, n_mels, time) log-mel spectrogram
        Returns:
            (batch, seq_len, whisper_hidden_size)
        """
        encoder_outputs = self.whisper.model.encoder(audio_values)
        return encoder_outputs.last_hidden_state

    def _decode_with_visual_xattn(
        self,
        encoder_hidden_states: torch.Tensor,
        visual_features: Optional[torch.Tensor],
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Custom decoder forward with gated cross-attention to visual features.

        Runs the Whisper decoder layer-by-layer, injecting gated cross-attention
        at specified layer indices.
        """
        decoder = self.whisper.model.decoder

        # - Embed decoder input tokens
        hidden_states = decoder.embed_tokens(decoder_input_ids)
        hidden_states = hidden_states + decoder.embed_positions(decoder_input_ids)
        hidden_states = decoder.layernorm_embedding(hidden_states)

        xattn_idx = 0
        for layer_idx, decoder_layer in enumerate(decoder.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
            hidden_states = layer_outputs[0]

            # - Inject gated cross-attention to visual features
            if (
                visual_features is not None
                and layer_idx in self._xattn_layer_indices
                and xattn_idx < len(self.gated_xattn_layers)
            ):
                hidden_states = self.gated_xattn_layers[xattn_idx](
                    hidden_states, visual_features,
                )
                xattn_idx += 1

        hidden_states = decoder.layer_norm(hidden_states)
        return hidden_states

    def forward(
        self,
        audio_values: torch.Tensor,
        video_pixels: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            audio_values: (batch, n_mels, time) log-mel spectrogram
            video_pixels: (batch, time, C, H, W) or None for audio-only
            labels: (batch, seq_len) target token IDs
            attention_mask: (batch, seq_len) mask for labels
        """
        # - Encode audio via Whisper encoder
        encoder_hidden = self._encode_audio(audio_values)

        # - Encode and project visual features
        visual_features = None
        if video_pixels is not None and self.visual_encoder is not None:
            visual_out = self.visual_encoder(video_pixels)
            visual_features = self.visual_projection(visual_out)

        if labels is not None:
            # - Shift labels for teacher forcing (Whisper convention)
            decoder_input_ids = labels.clone()
            decoder_input_ids[decoder_input_ids == -100] = self.whisper.config.pad_token_id

            # - Custom decoder with visual cross-attention
            decoder_hidden = self._decode_with_visual_xattn(
                encoder_hidden, visual_features, decoder_input_ids,
            )

            # - Compute logits and loss
            lm_logits = self.whisper.proj_out(decoder_hidden)
            loss = nn.functional.cross_entropy(
                lm_logits.view(-1, lm_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            return {"loss": loss, "logits": lm_logits}

        # - No labels: just return encoder output through standard Whisper
        outputs = self.whisper(input_features=audio_values)
        return {"logits": outputs.logits}

    @torch.no_grad()
    def generate(
        self,
        audio_values: torch.Tensor,
        video_pixels: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        **generate_kwargs,
    ) -> dict[str, list[str]]:
        """Auto-regressive inference with visual cross-attention.

        For generation, we use a simplified approach: encode visual features
        and use the standard Whisper generate with encoder outputs, then
        apply visual cross-attention during the decoding loop.
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens

        # - Encode audio
        encoder_hidden = self._encode_audio(audio_values)

        # - Encode visual
        visual_features = None
        if video_pixels is not None and self.visual_encoder is not None:
            visual_out = self.visual_encoder(video_pixels)
            visual_features = self.visual_projection(visual_out)

        batch_size = audio_values.size(0)
        device = audio_values.device

        # - Start with decoder start token
        decoder_start_id = self.whisper.config.decoder_start_token_id
        generated = torch.full(
            (batch_size, 1), decoder_start_id,
            dtype=torch.long, device=device,
        )

        eos_token_id = self.whisper.config.eos_token_id

        for _ in range(max_new_tokens):
            decoder_hidden = self._decode_with_visual_xattn(
                encoder_hidden, visual_features, generated,
            )
            logits = self.whisper.proj_out(decoder_hidden[:, -1:, :])
            next_token = logits.argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == eos_token_id).all():
                break

        # - Decode tokens to strings
        predictions = self.processor.batch_decode(generated, skip_special_tokens=True)
        return {"predictions": predictions}


def build_model(config: dict) -> WhisperFlamingo:
    """Entry point called by train.py / evaluate.py."""
    model_config = WhisperFlamingoConfig(
        whisper_id=config.get("whisper_id", "openai/whisper-large-v2"),
        visual_encoder_path=config.get("visual_encoder_path", "nguyenvulebinh/AV-HuBERT"),
        whisper_hidden_size=config.get("whisper_hidden_size", 1280),
        visual_hidden_size=config.get("visual_hidden_size", 1024),
        num_gated_xattn_layers=config.get("num_gated_xattn_layers", 4),
        xattn_heads=config.get("xattn_heads", 16),
        max_new_tokens=config.get("max_new_tokens", 256),
    )

    pretrained_path = config.get("pretrained_path", "./models/whisper_flamingo/pretrained")
    return WhisperFlamingo.from_pretrained(
        pretrained_path,
        whisper_id=model_config.whisper_id,
        visual_encoder_path=model_config.visual_encoder_path,
        config=model_config,
    )
