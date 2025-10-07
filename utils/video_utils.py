"""
Video Processing Utilities
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import time


class VideoProcessor:
    """Handle video input/output operations"""
    
    def __init__(self, video_path: str, output_path: Optional[str] = None,
                 resize_width: Optional[int] = None, 
                 resize_height: Optional[int] = None,
                 skip_frames: int = 1):
        self.video_path = video_path
        self.output_path = output_path
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.skip_frames = skip_frames
        
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        
        if resize_width and resize_height:
            self.output_width = resize_width
            self.output_height = resize_height
        else:
            self.output_width = self.width
            self.output_height = self.height
        
        
        self.writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                output_path, fourcc, self.fps,
                (self.output_width, self.output_height)
            )
        
        
        self.frame_times = []
        self.current_fps = 0.0
        
        print(f"  Video loaded: {video_path}")
        print(f"  - Resolution: {self.width}x{self.height}")
        print(f"  - FPS: {self.fps}")
        print(f"  - Frames: {self.frame_count}")
        print(f"  - Duration: {self.frame_count/self.fps:.1f}s")
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame"""
        ret, frame = self.cap.read()
        
        if not ret:
            return False, None
        
        
        if self.resize_width and self.resize_height:
            frame = cv2.resize(frame, (self.resize_width, self.resize_height))
        
        return True, frame
    
    def write_frame(self, frame: np.ndarray):
        """Write frame to output video"""
        if self.writer is not None:
            self.writer.write(frame)
    
    def update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        
        if len(self.frame_times) > 1:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            if time_diff > 0:
                self.current_fps = (len(self.frame_times) - 1) / time_diff
    
    def get_fps(self) -> float:
        """Get current processing FPS"""
        return self.current_fps
    
    def get_progress(self, current_frame: int) -> float:
        """Get processing progress percentage"""
        if self.frame_count > 0:
            return (current_frame / self.frame_count) * 100
        return 0.0
    
    def release(self):
        """Release video resources"""
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()


class FrameBuffer:
    """Buffer frames for smooth playback"""
    
    def __init__(self, max_size: int = 30):
        self.buffer = []
        self.max_size = max_size
    
    def add(self, frame: np.ndarray):
        """Add frame to buffer"""
        self.buffer.append(frame)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    
    def get(self, index: int = -1) -> Optional[np.ndarray]:
        """Get frame from buffer"""
        if 0 <= index < len(self.buffer):
            return self.buffer[index]
        elif -len(self.buffer) <= index < 0:
            return self.buffer[index]
        return None
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
    
    def size(self) -> int:
        """Get current buffer size"""
        return len(self.buffer)


def create_video_grid(frames: list, grid_size: Tuple[int, int] = None) -> np.ndarray:
    if not frames:
        return None
    
    n_frames = len(frames)
    
    
    if grid_size is None:
        rows = int(np.ceil(np.sqrt(n_frames)))
        cols = int(np.ceil(n_frames / rows))
        grid_size = (rows, cols)
    
    rows, cols = grid_size
    
    
    h, w = frames[0].shape[:2]
    
    
    grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
    
    
    for idx, frame in enumerate(frames):
        if idx >= rows * cols:
            break
        
        row = idx // cols
        col = idx % cols
        
        y1 = row * h
        y2 = (row + 1) * h
        x1 = col * w
        x2 = (col + 1) * w
        
        grid[y1:y2, x1:x2] = frame
    
    return grid


def stabilize_frame(frame: np.ndarray, prev_frame: np.ndarray = None) -> np.ndarray:
    """
    Simple frame stabilization
    
    Args:
        frame: Current frame
        prev_frame: Previous frame
        
    Returns:
        Stabilized frame
    """
    if prev_frame is None:
        return frame
    
    
    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)
    
    if p0 is None:
        return frame
    
    
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)
    
    if p1 is None:
        return frame
    
    
    good_old = p0[st == 1]
    good_new = p1[st == 1]
    
    if len(good_old) < 10:
        return frame
    
    
    transform = cv2.estimateAffinePartial2D(good_old, good_new)[0]
    
    if transform is None:
        return frame
    
    
    h, w = frame.shape[:2]
    stabilized = cv2.warpAffine(frame, transform, (w, h))
    
    return stabilized