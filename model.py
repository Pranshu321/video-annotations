# model_pipeline.py

import os
import torch
from transformers import (
    AutoModelForVideoClassification,
    AutoImageProcessor,
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

class Models:
    """
    A unified container for loading and managing models used in
    action recognition and image captioning pipelines.
    """

    def __init__(self, dtype: torch.dtype = torch.float16):
        """
        Initialize and load all required models.

        Args:
            dtype (torch.dtype): Preferred floating-point precision for model weights.
                                 Defaults to torch.float16 for GPU, torch.float32 for CPU.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if self.device.type == "cuda" else torch.float32

        # Load Action Recognition Model
        self._load_action_model()

        # Load Caption Generation Model
        self._load_caption_model()

    def _load_action_model(self):
        """Load the action recognition model and processor."""
        self.action_model_name = os.getenv(
            "ACTION_MODEL",
            "MCG-NJU/videomae-base-finetuned-kinetics"  # Default: video action recognition model
        )
        print(f"[INFO] Loading action model: {self.action_model_name}")

        self.action_processor = AutoImageProcessor.from_pretrained(self.action_model_name)
        self.action_model = AutoModelForVideoClassification.from_pretrained(
            self.action_model_name,
            torch_dtype=self.dtype
        ).to(self.device).eval()

    def _load_caption_model(self):
        """Load the image captioning model and processor."""
        self.caption_model_name = os.getenv(
            "CAPTION_MODEL",
            "Salesforce/blip-image-captioning-base"  # Default: BLIP base model
        )
        print(f"[INFO] Loading caption model: {self.caption_model_name}")

        try:
            self.caption_processor = BlipProcessor.from_pretrained(self.caption_model_name)
            self.caption_model = BlipForConditionalGeneration.from_pretrained(
                self.caption_model_name,
                torch_dtype=self.dtype
            ).to(self.device).eval()

        except Exception:
            # Fallback for tokenizer + seq2seq model loading
            print(f"[WARN] Falling back to AutoTokenizer + Seq2SeqLM for: {self.caption_model_name}")
            self.caption_processor = AutoTokenizer.from_pretrained(self.caption_model_name)
            self.caption_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.caption_model_name,
                torch_dtype=self.dtype
            ).to(self.device).eval()
