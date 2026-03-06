"""auto_avsr: Audio-Visual Speech Recognition baseline.

Reference: https://github.com/mpc001/auto_avsr
HF Models:
  - AV: nguyenvulebinh/auto_avsr_av_trlrwlrs2lrs3vox2avsp_base
  - Visual-only: nguyenvulebinh/auto_avsr_visual_trlrwlrs2lrs3vox2avsp_base
  - Audio-only: nguyenvulebinh/auto_avsr_audio_trlrwlrs2lrs3vox2avsp_base

Architecture:
  - Visual frontend: 3D-CNN + ResNet-18
  - Audio frontend: log filterbank / ResNet-18
  - Backend: Conformer encoder + Transformer decoder (hybrid CTC/Attention)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoTokenizer


class AutoAVSRConfig(PretrainedConfig):
    model_type = "auto_avsr"

    def __init__(
        self,
        model_id: str = "nguyenvulebinh/auto_avsr_av_trlrwlrs2lrs3vox2avsp_base",
        modality: str = "av",
        hidden_size: int = 768,
        num_encoder_layers: int = 12,
        num_decoder_layers: int = 6,
        num_attention_heads: int = 8,
        intermediate_size: int = 3072,
        vocab_size: int = 1024,
        max_new_tokens: int = 256,
        sample_rate: int = 16000,
        n_mels: int = 80,
        video_fps: int = 25,
        img_size: int = 96,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.modality = modality
        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.max_new_tokens = max_new_tokens
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.video_fps = video_fps
        self.img_size = img_size


class VisualFrontend(nn.Module):
    """3D-CNN + ResNet-18 visual feature extractor."""

    def __init__(self, hidden_size: int = 768, img_size: int = 96):
        super().__init__()
        # - 3D convolution for spatiotemporal features
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        # - ResNet-18 backbone (simplified)
        self.resnet = nn.Sequential(
            self._make_layer(64, 64, 2),
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 512, 2, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(512, hidden_size)

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        for _ in range(1, blocks):
            layers.extend([
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, channels, height, width) grayscale video frames
        Returns:
            (batch, time, hidden_size)
        """
        b, t, c, h, w = x.shape
        # - Reshape for 3D conv: (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv3d(x)
        # - Process each frame through ResNet
        _, c2, t2, h2, w2 = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t2, c2, h2, w2)
        x = self.resnet(x)
        x = self.pool(x).flatten(1)
        x = self.proj(x)
        return x.view(b, t2, -1)


class AudioFrontend(nn.Module):
    """Log-filterbank audio feature extractor."""

    def __init__(self, n_mels: int = 80, hidden_size: int = 768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_mels, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, n_mels) log-filterbank features
        Returns:
            (batch, time, hidden_size)
        """
        return self.proj(x)


class AutoAVSR(PreTrainedModel):
    config_class = AutoAVSRConfig

    def __init__(self, config: AutoAVSRConfig):
        super().__init__(config)
        self.config = config

        # - Frontends
        self.visual_frontend = VisualFrontend(config.hidden_size, config.img_size)
        self.audio_frontend = AudioFrontend(config.n_mels, config.hidden_size)

        # - Modality fusion (concatenation + projection)
        fusion_input_size = config.hidden_size * 2 if config.modality == "av" else config.hidden_size
        self.fusion_proj = nn.Linear(fusion_input_size, config.hidden_size)

        # - Conformer-style encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)

        # - Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_decoder_layers)

        # - CTC head
        self.ctc_head = nn.Linear(config.hidden_size, config.vocab_size)

        # - Attention decoder head
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.output_head = nn.Linear(config.hidden_size, config.vocab_size)

        # - CTC loss weight
        self.ctc_weight = 0.3

        # - HF model backend (loaded via from_pretrained if available)
        self._hf_model = None

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        config: Optional[AutoAVSRConfig] = None,
        **kwargs,
    ) -> "AutoAVSR":
        """Unified loading: accepts HF repo ID or local path."""
        path = Path(model_name_or_path)

        if config is None:
            config = AutoAVSRConfig(model_id=model_name_or_path)

        model = cls(config)

        # - Try loading as HF model
        try:
            hf_model = AutoModel.from_pretrained(model_name_or_path, **kwargs)
            model._hf_model = hf_model
            return model
        except Exception:
            pass

        # - Try loading as local checkpoint
        if path.is_file():
            ckpt = torch.load(str(path), map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            model.load_state_dict(state_dict, strict=False)
        elif path.is_dir():
            ckpt_file = path / "pytorch_model.bin"
            if ckpt_file.exists():
                state_dict = torch.load(str(ckpt_file), map_location="cpu")
                model.load_state_dict(state_dict, strict=False)

        return model

    def _encode(
        self,
        audio_values: Optional[torch.Tensor] = None,
        video_pixels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode audio and/or video inputs."""
        features = []

        if audio_values is not None:
            audio_feat = self.audio_frontend(audio_values)
            features.append(audio_feat)

        if video_pixels is not None:
            visual_feat = self.visual_frontend(video_pixels)
            features.append(visual_feat)

        if len(features) == 2:
            # - Align temporal dimensions
            min_len = min(features[0].size(1), features[1].size(1))
            features = [f[:, :min_len, :] for f in features]
            fused = torch.cat(features, dim=-1)
        elif len(features) == 1:
            fused = features[0]
        else:
            raise ValueError("At least one of audio_values or video_pixels must be provided")

        fused = self.fusion_proj(fused)
        encoder_out = self.encoder(fused)
        return encoder_out

    def forward(
        self,
        audio_values: Optional[torch.Tensor] = None,
        video_pixels: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Training forward with hybrid CTC/Attention loss.

        Args:
            audio_values: (batch, time, n_mels) or (batch, audio_samples)
            video_pixels: (batch, time, C, H, W)
            labels: (batch, seq_len) target token IDs
            attention_mask: (batch, seq_len) mask for labels
        """
        # - If HF model is loaded, delegate
        if self._hf_model is not None:
            return self._forward_hf(audio_values, video_pixels, labels, attention_mask)

        encoder_out = self._encode(audio_values, video_pixels)

        if labels is None:
            ctc_logits = self.ctc_head(encoder_out)
            return {"logits": ctc_logits}

        # - CTC loss
        ctc_logits = self.ctc_head(encoder_out)
        ctc_log_probs = ctc_logits.log_softmax(dim=-1).permute(1, 0, 2)
        input_lengths = torch.full((encoder_out.size(0),), encoder_out.size(1), dtype=torch.long, device=encoder_out.device)
        target_lengths = (labels != -100).sum(dim=-1)
        ctc_labels = labels.clone()
        ctc_labels[ctc_labels == -100] = 0

        ctc_loss = nn.functional.ctc_loss(
            ctc_log_probs, ctc_labels, input_lengths, target_lengths, blank=0, zero_infinity=True,
        )

        # - Attention decoder loss
        tgt_embeds = self.token_embedding(labels)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(labels.size(1), device=labels.device)
        decoder_out = self.decoder(tgt_embeds, encoder_out, tgt_mask=tgt_mask)
        att_logits = self.output_head(decoder_out)
        att_loss = nn.functional.cross_entropy(
            att_logits.view(-1, self.config.vocab_size),
            labels.view(-1),
            ignore_index=-100,
        )

        # - Combined loss
        loss = self.ctc_weight * ctc_loss + (1 - self.ctc_weight) * att_loss
        return {"loss": loss, "logits": att_logits}

    def _forward_hf(
        self,
        audio_values: Optional[torch.Tensor],
        video_pixels: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward pass delegating to HF model backend."""
        # - Construct inputs based on what the HF model expects
        inputs = {}
        if audio_values is not None:
            inputs["audio_values"] = audio_values
        if video_pixels is not None:
            inputs["video_pixels"] = video_pixels
        if labels is not None:
            inputs["labels"] = labels
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask

        outputs = self._hf_model(**inputs)
        result = {}
        if hasattr(outputs, "loss") and outputs.loss is not None:
            result["loss"] = outputs.loss
        if hasattr(outputs, "logits"):
            result["logits"] = outputs.logits
        return result

    @torch.no_grad()
    def generate(
        self,
        audio_values: Optional[torch.Tensor] = None,
        video_pixels: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        **generate_kwargs,
    ) -> dict[str, list[str]]:
        """Auto-regressive greedy decoding.

        Returns:
            dict with 'predictions' (list of token ID sequences)
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        encoder_out = self._encode(audio_values, video_pixels)

        batch_size = encoder_out.size(0)
        device = encoder_out.device

        # - Start with BOS token (assume token 1)
        generated = torch.ones(batch_size, 1, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            tgt_embeds = self.token_embedding(generated)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(generated.size(1), device=device)
            decoder_out = self.decoder(tgt_embeds, encoder_out, tgt_mask=tgt_mask)
            logits = self.output_head(decoder_out[:, -1:, :])
            next_token = logits.argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)

            # - Stop if all sequences produced EOS (assume token 2)
            if (next_token == 2).all():
                break

        return {"predictions": generated.tolist()}


def build_model(config: dict) -> AutoAVSR:
    """Entry point called by train.py / evaluate.py."""
    model_id = config.get("model_id", "nguyenvulebinh/auto_avsr_av_trlrwlrs2lrs3vox2avsp_base")
    modality = config.get("modality", "av")

    model_config = AutoAVSRConfig(
        model_id=model_id,
        modality=modality,
        hidden_size=config.get("hidden_size", 768),
        vocab_size=config.get("vocab_size", 1024),
        max_new_tokens=config.get("max_new_tokens", 256),
        n_mels=config.get("n_mels", 80),
        img_size=config.get("img_size", 96),
    )

    pretrained_path = config.get("pretrained_path", model_id)
    return AutoAVSR.from_pretrained(pretrained_path, config=model_config)
