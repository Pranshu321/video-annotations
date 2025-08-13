# inference_pipeline_multiprocess.py

import os
import time
import queue
import threading
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import torch
from PIL import Image

from model import Models  # Assuming you have a Models class to load your models


def list_frame_paths(chunk_dir: str) -> List[str]:
    """Return a sorted list of frame file paths from a directory."""
    return sorted([str(p) for p in Path(chunk_dir).glob("*.jpg")])


def build_clips_from_frames(frame_paths: List[str], frames_per_clip: int, stride: int) -> List[List[str]]:
    """
    Build a list of clips from frame paths.

    Args:
        frame_paths: List of frame file paths.
        frames_per_clip: Number of frames per clip.
        stride: Step size between clips.

    Returns:
        A list of clips, where each clip is a list of frame paths.
    """
    clips = []
    i = 0
    while i + frames_per_clip <= len(frame_paths):
        clips.append(frame_paths[i:i + frames_per_clip])
        i += stride

    # If not enough frames for even one clip, take all frames as one clip
    if not clips and frame_paths:
        clips = [frame_paths]

    return clips


def load_images_batch(clip_paths_batch: List[List[str]]) -> List[List[Image.Image]]:
    """Load images for multiple clips in parallel using threads."""

    def load_clip_images(clip_paths):
        return [Image.open(p).convert("RGB") for p in clip_paths]

    with ThreadPoolExecutor(max_workers=min(8, len(clip_paths_batch))) as executor:
        futures = {
            executor.submit(load_clip_images, clip_paths): i
            for i, clip_paths in enumerate(clip_paths_batch)
        }

        results = [None] * len(clip_paths_batch)
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    return results


def run_action_batch(models, clips_pil: List[List[Image.Image]]) -> List[Dict[str, Any]]:
    """Run action recognition on a batch of clips."""
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

    # Clean up GPU memory
    del batch_tensor, outputs, logits, probs, topk
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return [{"label": l, "score": s} for l, s in zip(labels, scores)]


def run_caption_batch(models, clips_pil: List[List[Image.Image]], max_length: int = 40) -> List[str]:
    """Run captioning on the center frame of each clip."""
    center_images = [clip[len(clip) // 2] for clip in clips_pil]
    inputs = models.caption_processor(images=center_images, return_tensors="pt").to(models.device)

    with torch.no_grad():
        outputs = models.caption_model.generate(**inputs, max_new_tokens=max_length)

    try:
        captions = models.caption_processor.batch_decode(outputs, skip_special_tokens=True)
    except Exception:
        captions = [models.caption_processor.decode(o, skip_special_tokens=True) for o in outputs]

    # Clean up GPU memory
    del inputs, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return [str(c) for c in captions]


def process_batch_worker(models, batch_clips_paths: List[List[str]],
                         batch_idx: int, frames_per_clip: int, stride: int) -> List[Dict[str, Any]]:
    """Worker function to process a batch of clips."""
    try:
        batch_clips_pil = load_images_batch(batch_clips_paths)
        action_preds = run_action_batch(models, batch_clips_pil)
        captions = run_caption_batch(models, batch_clips_pil)

        events = []
        for j, (a, cap, clip_paths) in enumerate(zip(action_preds, captions, batch_clips_paths)):
            fname = os.path.basename(clip_paths[0])
            try:
                idx = int(''.join(filter(str.isdigit, fname)))
            except Exception:
                idx = batch_idx * len(batch_clips_paths) + j * frames_per_clip

            events.append({
                "start_frame": idx,
                "label": a["label"],
                "score": round(a["score"], 3),
                "caption": cap,
                "batch_idx": batch_idx,
                "clip_idx": j
            })

        return events

    except Exception as e:
        print(f"Error in batch worker {batch_idx}: {e}")
        return []


class ModelManager:
    """Thread-safe model manager with memory optimization."""

    def __init__(self, models):
        self.models = models
        self._lock = threading.Lock()

    def get_models(self):
        with self._lock:
            return self.models


def process_chunk_multiprocess(models, chunk_dir: str,
                               frames_per_clip: int = 16,
                               stride: int = 8,
                               batch_size: int = 4,
                               max_workers: int = None,
                               use_threading: bool = True) -> List[Dict[str, Any]]:
    """
    Multiprocess version with options for threading or processing.
    """
    start_time = time.time()

    frame_paths = list_frame_paths(chunk_dir)
    if not frame_paths:
        print(f"[WARN] No frames found in {chunk_dir}")
        return []

    clips = build_clips_from_frames(frame_paths, frames_per_clip, stride)
    print(f"Processing {len(clips)} clips with {len(frame_paths)} frames")

    batches = [clips[i:i + batch_size] for i in range(0, len(clips), batch_size)]

    if max_workers is None:
        if use_threading:
            max_workers = min(len(batches), mp.cpu_count() * 2)
        else:
            max_workers = min(len(batches), mp.cpu_count())

    print(f"Using {max_workers} workers with {'threading' if use_threading else 'multiprocessing'}")
    all_events = []

    if use_threading:
        model_manager = ModelManager(models)

        def thread_worker(batch_data):
            batch_idx, batch_clips_paths = batch_data
            return process_batch_worker(model_manager.get_models(),
                                        batch_clips_paths, batch_idx, frames_per_clip, stride)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(thread_worker, (i, batch)): i
                for i, batch in enumerate(batches)
            }
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    events = future.result()
                    all_events.extend(events)
                    print(f"Completed batch {batch_idx + 1}/{len(batches)}")
                except Exception as e:
                    print(f"Batch {batch_idx} failed: {e}")

    else:
        print("Warning: Multiprocessing may not work well with GPU models")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                (executor.submit(process_batch_worker, models, batch, i, frames_per_clip, stride), i)
                for i, batch in enumerate(batches)
            ]
            for future, batch_idx in futures:
                try:
                    events = future.result()
                    all_events.extend(events)
                    print(f"Completed batch {batch_idx + 1}/{len(batches)}")
                except Exception as e:
                    print(f"Batch {batch_idx} failed: {e}")

    all_events.sort(key=lambda x: x["start_frame"])

    processing_time = time.time() - start_time
    print(f"Processing completed in {processing_time:.2f}s for {len(all_events)} events")
    print(f"Average time per event: {processing_time / max(1, len(all_events)):.3f}s")

    return all_events


def process_chunk_async_queue(models, chunk_dir: str,
                              frames_per_clip: int = 16,
                              stride: int = 8,
                              batch_size: int = 4,
                              queue_size: int = 10) -> List[Dict[str, Any]]:
    """
    Producer-consumer pattern with queues for maximum throughput and minimal latency.
    """
    start_time = time.time()

    frame_paths = list_frame_paths(chunk_dir)
    if not frame_paths:
        print(f"[WARN] No frames found in {chunk_dir}")
        return []

    clips = build_clips_from_frames(frame_paths, frames_per_clip, stride)
    batches = [clips[i:i + batch_size] for i in range(0, len(clips), batch_size)]

    load_queue = queue.Queue(maxsize=queue_size)
    inference_queue = queue.Queue(maxsize=queue_size)
    results = []

    def image_loader():
        for batch_idx, batch in enumerate(batches):
            batch_clips_pil = load_images_batch(batch)
            load_queue.put((batch_idx, batch, batch_clips_pil))
        load_queue.put(None)

    def inference_worker():
        while True:
            item = load_queue.get()
            if item is None:
                inference_queue.put(None)
                break

            batch_idx, batch_clips_paths, batch_clips_pil = item
            action_preds = run_action_batch(models, batch_clips_pil)
            captions = run_caption_batch(models, batch_clips_pil)

            events = []
            for j, (a, cap, clip_paths) in enumerate(zip(action_preds, captions, batch_clips_paths)):
                fname = os.path.basename(clip_paths[0])
                try:
                    idx = int(''.join(filter(str.isdigit, fname)))
                except Exception:
                    idx = batch_idx * len(batch_clips_paths) + j * frames_per_clip

                events.append({
                    "start_frame": idx,
                    "label": a["label"],
                    "score": round(a["score"], 3),
                    "caption": cap
                })

            inference_queue.put((batch_idx, events))
            load_queue.task_done()

    loader_thread = threading.Thread(target=image_loader)
    inference_thread = threading.Thread(target=inference_worker)

    loader_thread.start()
    inference_thread.start()

    processed_batches = 0
    while True:
        item = inference_queue.get()
        if item is None:
            break
        batch_idx, events = item
        results.extend(events)
        processed_batches += 1
        print(f"Completed batch {processed_batches}/{len(batches)}")

    loader_thread.join()
    inference_thread.join()

    results.sort(key=lambda x: x["start_frame"])

    processing_time = time.time() - start_time
    print(f"Async processing completed in {processing_time:.2f}s for {len(results)} events")

    return results
