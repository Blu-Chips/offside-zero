"""
Video Processor for Offside Zero
Extracts frames from video for analysis
"""

import cv2
import os
from typing import List, Tuple, Optional
from PIL import Image
import numpy as np


class VideoProcessor:
    def __init__(self, video_path: str):
        """Initialize with video file path."""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
    
    def get_info(self) -> dict:
        """Get video metadata."""
        return {
            "path": self.video_path,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.duration
        }
    
    def extract_frame(self, frame_number: int, retries: int = 3) -> Optional[np.ndarray]:
        """Extract a single frame by frame number with retry logic."""
        for attempt in range(retries):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # If failed and not last attempt, could be a seek issue, try again
            if attempt < retries - 1:
                continue
        return None
    
    def extract_frame_at_time(self, seconds: float) -> Optional[np.ndarray]:
        """Extract frame at specific timestamp."""
        frame_number = int(seconds * self.fps)
        return self.extract_frame(frame_number)
    
    def extract_frames_range(
        self, 
        start_frame: int, 
        end_frame: int, 
        step: int = 1
    ) -> List[np.ndarray]:
        """Extract multiple frames in a range."""
        frames = []
        for i in range(start_frame, min(end_frame, self.frame_count), step):
            frame = self.extract_frame(i)
            if frame is not None:
                frames.append(frame)
        return frames
    
    def extract_frames_around(
        self, 
        center_time: float, 
        window_seconds: float = 1.0,
        num_frames: int = 5
    ) -> List[np.ndarray]:
        """
        Extract frames around a specific timestamp.
        Useful for analyzing a specific incident.
        """
        center_frame = int(center_time * self.fps)
        window_frames = int(window_seconds * self.fps)
        
        start = max(0, center_frame - window_frames // 2)
        end = min(self.frame_count, center_frame + window_frames // 2)
        
        step = max(1, (end - start) // num_frames)
        return self.extract_frames_range(start, end, step)
    
    def save_frames(
        self, 
        frames: List[np.ndarray], 
        output_dir: str, 
        prefix: str = "frame"
    ) -> List[str]:
        """Save frames to disk and return paths."""
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        
        for i, frame in enumerate(frames):
            path = os.path.join(output_dir, f"{prefix}_{i:04d}.jpg")
            img = Image.fromarray(frame)
            img.save(path, "JPEG", quality=90)
            paths.append(path)
        
        return paths
    
    def frames_to_pil(self, frames: List[np.ndarray]) -> List[Image.Image]:
        """Convert numpy frames to PIL Images."""
        return [Image.fromarray(f) for f in frames]
    
    def create_slow_motion(
        self,
        start_time: float,
        end_time: float,
        output_path: str,
        slow_factor: float = 0.25
    ) -> str:
        """
        Create a slow-motion clip from the video.
        
        Args:
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds
            output_path: Path for output video
            slow_factor: Speed factor (0.25 = 4x slower)
        
        Returns:
            Path to output video
        """
        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)
        
        # New FPS for slow motion
        new_fps = self.fps * slow_factor
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, new_fps, (self.width, self.height))
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for _ in range(end_frame - start_frame):
            ret, frame = self.cap.read()
            if not ret:
                break
            out.write(frame)
        
        out.release()
        return output_path
    
    def close(self):
        """Release video capture."""
        self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


def load_video(path: str) -> VideoProcessor:
    """Factory function to load a video."""
    return VideoProcessor(path)
