## License & Acknowledgements

The core framework, data processing pipelines (e.g., KMSAV dataset integration), and training scripts of this repository are licensed under the **MIT License**. See the `LICENSE` file in the root directory for more details.

However, the baseline models included in the `models/` directory are heavily based on external open-source projects. We express our deep gratitude to the original authors for releasing their codebase and pre-trained weights to the research community. These specific subdirectories are subject to their own original licenses:

* **[AV-HuBERT](https://github.com/facebookresearch/av_hubert):** Licensed under the MIT License. Copyright © Meta Platforms, Inc. and affiliates.
* **[Auto-AVSR](https://github.com/mpc001/auto_avsr):** Licensed under the MIT License. Copyright © Imperial College London.
* **[Whisper-Flamingo / mWhisper-Flamingo](https://github.com/roudimit/whisper-flamingo):** Licensed under the MIT License.
* **[LLaMA-AVSR](https://github.com/umbertocappellazzo/Llama-AVSR) & [MMS-LLaMA](https://github.com/JeongHun0716/MMS-LLaMA):** These projects are built upon the Llama-2 backbone. Therefore, their usage is strictly subject to the **[Meta Llama 2 Community License Agreement](https://ai.meta.com/llama/license/)**. Please note that certain commercial uses may be restricted or require explicit permission from Meta.

*Note: Please refer to the original `LICENSE` files located within each respective model's subdirectory for the exact terms and conditions before using or distributing the code.*