"""AV-HuBERT: Audio-Visual Hidden Unit BERT for Speech Recognition.

Reference: https://github.com/facebookresearch/av_hubert
HF Model: nguyenvulebinh/AV-HuBERT (safetensors format)

Architecture:
  - Visual frontend: 3D-CNN + ResNet-18 (lip region)
  - Audio frontend: log filterbank features
  - Multimodal encoder: Transformer (HuBERT-style self-supervised)
  - CTC / seq2seq decoder for fine-tuning
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, AutoModel


class AVHuBERTConfig(PretrainedConfig):
    model_type = "av_hubert"

    def __init__(
        self,
        model_id: str = "nguyenvulebinh/AV-HuBERT",
        hidden_size: int = 1024,
        num_encoder_layers: int = 24,
        num_attention_heads: int = 16,
        intermediate_size: int = 4096,
        vocab_size: int = 1024,
        max_new_tokens: int = 256,
        sample_rate: int = 16000,
        n_mels: int = 80,
        video_fps: int = 25,
        img_size: int = 96,
        ctc_weight: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.max_new_tokens = max_new_tokens
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.video_fps = video_fps
        self.img_size = img_size
        self.ctc_weight = ctc_weight


class VisualFrontend(nn.Module):
    """3D-CNN + ResNet-18 for lip ROI feature extraction."""

    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
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
            x: (batch, time, 1, H, W) grayscale lip ROI frames
        Returns:
            (batch, time', hidden_size)
        """
        b, t, c, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv3d(x)
        _, c2, t2, h2, w2 = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t2, c2, h2, w2)
        x = self.resnet(x)
        x = self.pool(x).flatten(1)
        x = self.proj(x)
        return x.view(b, t2, -1)


class AudioFrontend(nn.Module):
    """Log-filterbank feature projection for audio stream."""

    def __init__(self, n_mels: int = 80, hidden_size: int = 1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_mels, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, n_mels)
        Returns:
            (batch, time, hidden_size)
        """
        return self.proj(x)


class MultimodalFusion(nn.Module):
    """Concatenation + linear projection for AV fusion."""

    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.proj = nn.Linear(hidden_size * 2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        audio_feat: torch.Tensor,
        visual_feat: torch.Tensor,
    ) -> torch.Tensor:
        # - Align temporal dimensions
        min_len = min(audio_feat.size(1), visual_feat.size(1))
        audio_feat = audio_feat[:, :min_len, :]
        visual_feat = visual_feat[:, :min_len, :]
        fused = torch.cat([audio_feat, visual_feat], dim=-1)
        return self.norm(self.proj(fused))


class AVHuBERT(PreTrainedModel):
    config_class = AVHuBERTConfig

    def __init__(self, config: AVHuBERTConfig):
        super().__init__(config)
        self.config = config

        # - Frontends
        self.visual_frontend = VisualFrontend(config.hidden_size)
        self.audio_frontend = AudioFrontend(config.n_mels, config.hidden_size)
        self.fusion = MultimodalFusion(config.hidden_size)

        # - HuBERT-style Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)

        # - Feature projection (post-encoder)
        self.feature_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # - CTC head
        self.ctc_head = nn.Linear(config.hidden_size, config.vocab_size)

        # - Seq2seq decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.output_head = nn.Linear(config.hidden_size, config.vocab_size)

        # - HF model backend
        self._hf_model = None

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        config: Optional[AVHuBERTConfig] = None,
        **kwargs,
    ) -> "AVHuBERT":
        """Unified loading: HF repo ID, local .pt (fairseq), or local directory."""
        path = Path(model_name_or_path)

        if config is None:
            config = AVHuBERTConfig(model_id=model_name_or_path)

        model = cls(config)

        # - Case 1: Local .pt file (fairseq checkpoint)
        if path.is_file() and path.suffix == ".pt":
            ckpt = torch.load(str(path), map_location="cpu")
            if "model" in ckpt:
                state_dict = ckpt["model"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
            model.load_state_dict(state_dict, strict=False)
            return model

        # - Case 2: Local directory with pytorch_model.bin
        if path.is_dir():
            ckpt_file = path / "pytorch_model.bin"
            if ckpt_file.exists():
                state_dict = torch.load(str(ckpt_file), map_location="cpu")
                model.load_state_dict(state_dict, strict=False)
                return model

        # - Case 3: HuggingFace repo ID
        try:
            hf_model = AutoModel.from_pretrained(model_name_or_path, **kwargs)
            model._hf_model = hf_model
        except Exception:
            print(f"Warning: Could not load from {model_name_or_path}. Using random init.")

        return model

    def _encode(
        self,
        audio_values: Optional[torch.Tensor] = None,
        video_pixels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode multimodal inputs through frontends + transformer encoder."""
        if self._hf_model is not None:
            # - Delegate to HF model encoder
            inputs = {}
            if audio_values is not None:
                inputs["audio_values"] = audio_values
            if video_pixels is not None:
                inputs["video_pixels"] = video_pixels
            outputs = self._hf_model(**inputs)
            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state
            return outputs[0]

        if audio_values is not None and video_pixels is not None:
            audio_feat = self.audio_frontend(audio_values)
            visual_feat = self.visual_frontend(video_pixels)
            fused = self.fusion(audio_feat, visual_feat)
        elif audio_values is not None:
            fused = self.audio_frontend(audio_values)
        elif video_pixels is not None:
            fused = self.visual_frontend(video_pixels)
        else:
            raise ValueError("At least one of audio_values or video_pixels required")

        encoder_out = self.encoder(fused)
        encoder_out = self.feature_proj(encoder_out)
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
            audio_values: (batch, time, n_mels) log-filterbank features
            video_pixels: (batch, time, 1, H, W) grayscale lip ROI
            labels: (batch, seq_len) target token IDs
            attention_mask: (batch, seq_len) mask for labels
        """
        encoder_out = self._encode(audio_values, video_pixels)

        if labels is None:
            ctc_logits = self.ctc_head(encoder_out)
            return {"logits": ctc_logits}

        # - CTC loss
        ctc_logits = self.ctc_head(encoder_out)
        ctc_log_probs = ctc_logits.log_softmax(dim=-1).permute(1, 0, 2)
        input_lengths = torch.full(
            (encoder_out.size(0),), encoder_out.size(1),
            dtype=torch.long, device=encoder_out.device,
        )
        target_lengths = (labels != -100).sum(dim=-1)
        ctc_labels = labels.clone()
        ctc_labels[ctc_labels == -100] = 0

        ctc_loss = nn.functional.ctc_loss(
            ctc_log_probs, ctc_labels, input_lengths, target_lengths,
            blank=0, zero_infinity=True,
        )

        # - Attention decoder loss
        tgt_embeds = self.token_embedding(labels)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            labels.size(1), device=labels.device,
        )
        decoder_out = self.decoder(tgt_embeds, encoder_out, tgt_mask=tgt_mask)
        att_logits = self.output_head(decoder_out)
        att_loss = nn.functional.cross_entropy(
            att_logits.view(-1, self.config.vocab_size),
            labels.view(-1),
            ignore_index=-100,
        )

        # - Combined loss
        loss = self.config.ctc_weight * ctc_loss + (1 - self.config.ctc_weight) * att_loss
        return {"loss": loss, "logits": att_logits}

    @torch.no_grad()
    def generate(
        self,
        audio_values: Optional[torch.Tensor] = None,
        video_pixels: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        **generate_kwargs,
    ) -> dict[str, list]:
        """Auto-regressive greedy decoding."""
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        encoder_out = self._encode(audio_values, video_pixels)

        batch_size = encoder_out.size(0)
        device = encoder_out.device

        # - BOS token = 1, EOS token = 2
        generated = torch.ones(batch_size, 1, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            tgt_embeds = self.token_embedding(generated)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                generated.size(1), device=device,
            )
            decoder_out = self.decoder(tgt_embeds, encoder_out, tgt_mask=tgt_mask)
            logits = self.output_head(decoder_out[:, -1:, :])
            next_token = logits.argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == 2).all():
                break

        return {"predictions": generated.tolist()}


def build_model(config: dict) -> AVHuBERT:
    """Entry point called by train.py / evaluate.py."""
    model_id = config.get("model_id", "nguyenvulebinh/AV-HuBERT")

    model_config = AVHuBERTConfig(
        model_id=model_id,
        hidden_size=config.get("hidden_size", 1024),
        num_encoder_layers=config.get("num_encoder_layers", 24),
        num_attention_heads=config.get("num_attention_heads", 16),
        intermediate_size=config.get("intermediate_size", 4096),
        vocab_size=config.get("vocab_size", 1024),
        max_new_tokens=config.get("max_new_tokens", 256),
        n_mels=config.get("n_mels", 80),
        img_size=config.get("img_size", 96),
        ctc_weight=config.get("ctc_weight", 0.3),
    )

    pretrained_path = config.get("pretrained_path", model_id)
    return AVHuBERT.from_pretrained(pretrained_path, config=model_config)
