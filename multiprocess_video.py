import cv2 as cv
import time
import multiprocessing as mp
import os
from functools import partial

def get_video_frame_details(file_name):
    """Get video properties like width, height, and frame count"""
    cap = cv.VideoCapture(file_name)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {file_name}")
    
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv.CAP_PROP_FPS)
    
    cap.release()
    
    if width == 0 or height == 0 or frame_count == 0:
        raise ValueError(f"Invalid video properties: width={width}, height={height}, frame_count={frame_count}")
    
    return width, height, frame_count, fps

def process_video_chunk(args):
    """Process a chunk of video frames"""
    file_name, start_frame, end_frame, output_dir = args
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open video capture
    cap = cv.VideoCapture(file_name)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {file_name}")
        return
    
    # Set starting position
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
    
    current_frame = start_frame
    frames_processed = 0
    
    try:
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Cannot read frame {current_frame}")
                break
            
            # Check if frame is valid
            if frame is None or frame.size == 0:
                print(f"Warning: Empty frame at position {current_frame}")
                current_frame += 1
                continue
            
            # Save frame as image
            image_filename = f"{output_dir}/frame_{current_frame:06d}.jpg"
            success = cv.imwrite(image_filename, frame)
            
            if not success:
                print(f"Warning: Failed to save frame {current_frame}")
            else:
                frames_processed += 1
            
            current_frame += 1
        
        print(f"Process completed: frames {start_frame}-{end_frame-1}, saved {frames_processed} frames")
        
    except Exception as e:
        print(f"Error processing frames {start_frame}-{end_frame}: {e}")
    finally:
        cap.release()

def multi_process_frame_extraction(file_name, output_dir="extracted_frames", num_processes=None):
    """Main function to extract frames using multiprocessing"""
    
    # Validate input file
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Video file not found: {file_name}")
    
    # if extracted_frames folder already exit delete that
    if os.path.exists(output_dir):
        print(f"Removing existing output directory: {output_dir}")
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))
        os.rmdir(output_dir)
    try:
        width, height, frame_count, fps = get_video_frame_details(file_name)
        print(f"Video properties:")
        print(f"  Dimensions: {width}x{height}")
        print(f"  Frame count: {frame_count}")
        print(f"  FPS: {fps}")
        
    except Exception as e:
        print(f"Error reading video properties: {e}")
        return
    
    # Determine number of processes
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 8)  # Cap at 8 to avoid too many processes
    
    print(f"Using {num_processes} processes...")
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate frame chunks for each process
    frames_per_process = frame_count // num_processes
    tasks = []
    
    for i in range(num_processes):
        start_frame = i * frames_per_process
        if i == num_processes - 1:
            # Last process handles remaining frames
            end_frame = frame_count
        else:
            end_frame = (i + 1) * frames_per_process
        
        tasks.append((file_name, start_frame, end_frame, output_dir))
    
    print("Starting frame extraction...")
    start_time = time.time()
    
    # Process in parallel
    with mp.Pool(num_processes) as pool:
        pool.map(process_video_chunk, tasks)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Count extracted frames
    extracted_frames = len([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
    
    print(f"\nExtraction completed:")
    print(f"  Time taken: {total_time:.2f} seconds")
    print(f"  Frames extracted: {extracted_frames}/{frame_count}")
    print(f"  Processing speed: {extracted_frames/total_time:.2f} FPS")
    print(f"  Output directory: {output_dir}")