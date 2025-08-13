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
    def __init__(self, dtype=torch.float16):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if self.device.type == "cuda" else torch.float32

        # === Action model ===
        self.action_model_name = os.getenv(
            "ACTION_MODEL",
            "MCG-NJU/videomae-base-finetuned-kinetics"  # better for actions
        )
        print(f"Loading action model: {self.action_model_name}")
        self.action_processor = AutoImageProcessor.from_pretrained(self.action_model_name)
        self.action_model = AutoModelForVideoClassification.from_pretrained(
            self.action_model_name,
            torch_dtype=self.dtype
        ).to(self.device).eval()

        # === Caption model ===
        self.caption_model_name = os.getenv(
            "CAPTION_MODEL",
            "Salesforce/blip-image-captioning-base"
        )
        print(f"Loading caption model: {self.caption_model_name}")
        try:
            self.caption_processor = BlipProcessor.from_pretrained(self.caption_model_name)
            self.caption_model = BlipForConditionalGeneration.from_pretrained(
                self.caption_model_name,
                torch_dtype=self.dtype
            ).to(self.device).eval()
        except Exception:
            self.caption_processor = AutoTokenizer.from_pretrained(self.caption_model_name)
            self.caption_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.caption_model_name,
                torch_dtype=self.dtype
            ).to(self.device).eval()
