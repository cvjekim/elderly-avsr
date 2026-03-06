#!/usr/bin/env bash
# Download pre-trained weights for AVSR baseline models.
# Usage: bash scripts/download_weights.sh
# Requirements: wget, gdown (pip install gdown)
#
# NOTE on HF models: Models fully available on Hugging Face (auto_avsr, AV-HuBERT, facebook/mms-1b-all,
# meta-llama/Llama-2-7b-hf, openai/whisper-large-v2) are loaded directly via from_pretrained() in model.py.
# This script only downloads NON-HF custom weights (.pt, .pth, .bin) from official GitHub repos.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${PROJECT_ROOT}/models"

echo "============================================"
echo " AVSR Baseline - Custom Weight Downloader"
echo "============================================"
echo "Models directory: ${MODELS_DIR}"
echo ""

# -------------------------------------------------------
# 1. AV-HuBERT
#    Repo: https://github.com/facebookresearch/av_hubert
#    Custom .pt checkpoints (fairseq format) from official release.
#    These are NOT on Hugging Face in native format.
# -------------------------------------------------------
AV_HUBERT_DIR="${MODELS_DIR}/av_hubert/pretrained"
mkdir -p "${AV_HUBERT_DIR}"

echo "[1/3] Downloading AV-HuBERT custom weights..."

# TODO: Replace <PLACEHOLDER_URL> with the exact download URL from the official AV-HuBERT repo.
# The official repo (https://github.com/facebookresearch/av_hubert) hosts weights on fbaipublicfiles.
# Check the repo README "Pre-trained Models" table for the correct links.

# AV-HuBERT Large (LRS3, iter5) - self-supervised pre-trained
wget -c -O "${AV_HUBERT_DIR}/large_vox_iter5.pt" \
    "<PLACEHOLDER_URL_AVHUBERT_LARGE_PRETRAIN>" \
    || echo "WARNING: Failed to download AV-HuBERT large pretrained. Replace <PLACEHOLDER_URL_AVHUBERT_LARGE_PRETRAIN> with the real URL."

# AV-HuBERT Large fine-tuned (LRS3, 433h)
wget -c -O "${AV_HUBERT_DIR}/large_lrs3_433h.pt" \
    "<PLACEHOLDER_URL_AVHUBERT_LARGE_FT_433H>" \
    || echo "WARNING: Failed to download AV-HuBERT large fine-tuned. Replace <PLACEHOLDER_URL_AVHUBERT_LARGE_FT_433H> with the real URL."

echo "[1/3] AV-HuBERT weights -> ${AV_HUBERT_DIR}"
echo ""

# -------------------------------------------------------
# 2. Whisper-Flamingo / mWhisper-Flamingo
#    Repo: https://github.com/roudimit/whisper-flamingo
#    Custom checkpoint files (not on Hugging Face Hub).
#    The base Whisper model (openai/whisper-large-v2) is loaded via HF.
# -------------------------------------------------------
WHISPER_FLAMINGO_DIR="${MODELS_DIR}/whisper_flamingo/pretrained"
mkdir -p "${WHISPER_FLAMINGO_DIR}"

echo "[2/3] Downloading Whisper-Flamingo custom weights..."

# TODO: Replace <PLACEHOLDER_URL> with the exact download URL from the official whisper-flamingo repo.
# Check https://github.com/roudimit/whisper-flamingo README for Google Drive / direct links.

# Whisper-Flamingo checkpoint (audio-visual gated cross-attention weights)
wget -c -O "${WHISPER_FLAMINGO_DIR}/whisper_flamingo.pt" \
    "<PLACEHOLDER_URL_WHISPER_FLAMINGO>" \
    || echo "WARNING: Failed to download Whisper-Flamingo. Replace <PLACEHOLDER_URL_WHISPER_FLAMINGO> with the real URL."

# mWhisper-Flamingo checkpoint (multilingual variant)
wget -c -O "${WHISPER_FLAMINGO_DIR}/mwhisper_flamingo.pt" \
    "<PLACEHOLDER_URL_MWHISPER_FLAMINGO>" \
    || echo "WARNING: Failed to download mWhisper-Flamingo. Replace <PLACEHOLDER_URL_MWHISPER_FLAMINGO> with the real URL."

echo "[2/3] Whisper-Flamingo weights -> ${WHISPER_FLAMINGO_DIR}"
echo ""

# -------------------------------------------------------
# 3. MMS-LLaMA
#    Repo: https://github.com/JeongHun0716/MMS-LLaMA
#    Custom pre-trained weights from the repo's `pre-trained` folder.
#    The base models (facebook/mms-1b-all, meta-llama/Llama-2-7b-hf) are loaded via HF.
# -------------------------------------------------------
MMS_LLAMA_DIR="${MODELS_DIR}/mms_llama/pretrained"
mkdir -p "${MMS_LLAMA_DIR}"

echo "[3/3] Downloading MMS-LLaMA custom weights..."

# TODO: Replace <PLACEHOLDER_URL> with the exact download URL from the official MMS-LLaMA repo.
# Check https://github.com/JeongHun0716/MMS-LLaMA README for Google Drive / direct links.

# MMS-LLaMA projection / adapter weights
wget -c -O "${MMS_LLAMA_DIR}/mms_llama_pretrained.bin" \
    "<PLACEHOLDER_URL_MMS_LLAMA>" \
    || echo "WARNING: Failed to download MMS-LLaMA weights. Replace <PLACEHOLDER_URL_MMS_LLAMA> with the real URL."

echo "[3/3] MMS-LLaMA weights -> ${MMS_LLAMA_DIR}"
echo ""

# -------------------------------------------------------
# Models loaded entirely via HuggingFace (NO download needed here):
#   - auto_avsr: nguyenvulebinh/auto_avsr_av_trlrwlrs2lrs3vox2avsp_base (HF)
#   - LLaMA-AVSR base LLM: meta-llama/Llama-2-7b-hf (HF, requires auth)
#   - MMS encoder: facebook/mms-1b-all (HF)
#   - Whisper base: openai/whisper-large-v2 (HF)
# -------------------------------------------------------

echo "============================================"
echo " Download complete!"
echo "============================================"
echo ""
echo "Custom weights directory layout:"
echo "  models/av_hubert/pretrained/"
echo "    ├── large_vox_iter5.pt"
echo "    └── large_lrs3_433h.pt"
echo "  models/whisper_flamingo/pretrained/"
echo "    ├── whisper_flamingo.pt"
echo "    └── mwhisper_flamingo.pt"
echo "  models/mms_llama/pretrained/"
echo "    └── mms_llama_pretrained.bin"
echo ""
echo "HF models (loaded via from_pretrained in code):"
echo "  - nguyenvulebinh/auto_avsr_av_trlrwlrs2lrs3vox2avsp_base"
echo "  - nguyenvulebinh/AV-HuBERT"
echo "  - facebook/mms-1b-all"
echo "  - meta-llama/Llama-2-7b-hf"
echo "  - openai/whisper-large-v2"
echo ""
