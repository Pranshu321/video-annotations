# inference_pipeline_optimized.py
import os
from pathlib import Path
from typing import List
from PIL import Image
import torch

def list_frame_paths(chunk_dir: str) -> List[str]:
    return sorted([str(p) for p in Path(chunk_dir).glob("*.jpg")])

def build_clips_from_frames(frame_paths: List[str], frames_per_clip: int, stride: int) -> List[List[str]]:
    clips = []
    i = 0
    while i + frames_per_clip <= len(frame_paths):
        clips.append(frame_paths[i:i+frames_per_clip])
        i += stride
    # If not enough frames for even one clip, take all frames as one clip
    if not clips and frame_paths:
        clips = [frame_paths]
    return clips

def load_images(paths: List[str]) -> List[Image.Image]:
    return [Image.open(p).convert("RGB") for p in paths]

def run_action_batch(models, clips_pil: List[List[Image.Image]]):
    # VideoMAE / TimeSformer expects (B, C, T, H, W)
    # Use the model's own processor for correct preprocessing
    pixel_values = []
    for clip in clips_pil:
        processed = models.action_processor(clip, return_tensors="pt")
        pixel_values.append(processed["pixel_values"])
    batch_tensor = torch.cat(pixel_values, dim=0).to(models.device, dtype=models.dtype)

    with torch.no_grad():
        outputs = models.action_model(pixel_values=batch_tensor)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        topk = probs.argmax(dim=-1)

    labels = [models.action_model.config.id2label[int(idx)] for idx in topk]
    scores = [float(probs[i, int(topk[i])].item()) for i in range(len(topk))]
    return [{"label": l, "score": s} for l, s in zip(labels, scores)]

def run_caption_batch(models, clips_pil: List[List[Image.Image]], max_length=40):
    # Use the center frame for BLIP-style image captioning
    center_images = [clip[len(clip)//2] for clip in clips_pil]
    inputs = models.caption_processor(images=center_images, return_tensors="pt").to(models.device)
    with torch.no_grad():
        outputs = models.caption_model.generate(**inputs, max_new_tokens=max_length)
    try:
        captions = models.caption_processor.batch_decode(outputs, skip_special_tokens=True)
    except Exception:
        captions = [models.caption_processor.decode(o, skip_special_tokens=True) for o in outputs]
    return [str(c) for c in captions]

def process_chunk_optimized(models, chunk_dir: str,
                            frames_per_clip: int = 16,
                            stride: int = 8,
                            batch_size: int = 4):
    frame_paths = list_frame_paths(chunk_dir)
    if not frame_paths:
        print(f"[WARN] No frames found in {chunk_dir}")
        return []

    clips = build_clips_from_frames(frame_paths, frames_per_clip, stride)

    events = []
    for i in range(0, len(clips), batch_size):
        batch_clips_paths = clips[i:i+batch_size]
        batch_clips_pil = [load_images(clip_paths) for clip_paths in batch_clips_paths]

        # Run action recognition
        action_preds = run_action_batch(models, batch_clips_pil)

        # Run captions
        captions = run_caption_batch(models, batch_clips_pil)

        # Combine results
        for j, (a, cap, clip_paths) in enumerate(zip(action_preds, captions, batch_clips_paths)):
            fname = os.path.basename(clip_paths[0])
            try:
                idx = int(''.join(filter(str.isdigit, fname)))
            except Exception:
                idx = i * stride + j * frames_per_clip
            events.append({
                "start_frame": idx,
                "label": a["label"],
                "score": round(a["score"], 3),
                "caption": cap
            })

    return events
