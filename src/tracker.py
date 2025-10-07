from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
from typing import List, Dict


class ObjectTracker:
    """DeepSORT-based multi-object tracker"""
    
    def __init__(self, max_age: int = 50, n_init: int = 3, 
                 max_iou_distance: float = 0.7, embedder: str = "mobilenet",
                 embedder_gpu: bool = True, nn_budget: int = 100):
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            embedder=embedder,
            embedder_gpu=embedder_gpu,
            nn_budget=nn_budget
        )
        
        self.active_tracks = {}  
        
        print(f"✓ Tracker initialized: DeepSORT")
        print(f"  - Embedder: {embedder}")
        print(f"  - Max age: {max_age}")
        print(f"  - N init: {n_init}")
    
    def update(self, detections: List[tuple], frame: np.ndarray) -> List[Dict]:
        
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        active_tracks = []
        
        for track in tracks:
            
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            bbox = track.to_ltrb()  
            
            
            class_id = track.get_det_class()
            if class_id is None:
                class_id = -1
            
            
            confidence = track.get_det_conf()
            if confidence is None:
                confidence = 0.0
            
            track_info = {
                'track_id': track_id,
                'bbox': bbox,
                'class_id': int(class_id),
                'confidence': float(confidence),
                'age': track.age,
                'hits': track.hits,
                'time_since_update': track.time_since_update
            }
            
            active_tracks.append(track_info)
            self.active_tracks[track_id] = track_info
        
        return active_tracks
    
    def get_track_info(self, track_id: int) -> Dict:
        """Get information about a specific track"""
        return self.active_tracks.get(track_id, None)
    
    def get_active_track_ids(self) -> List[int]:
        """Get list of currently active track IDs"""
        return list(self.active_tracks.keys())
    
    def get_track_count(self) -> int:
        """Get number of active tracks"""
        return len(self.active_tracks)
    
    def reset(self):
        """Reset tracker"""
        self.tracker = DeepSort(
            max_age=self.tracker.max_age,
            n_init=self.tracker.n_init,
            max_iou_distance=self.tracker.max_iou_distance
        )
        self.active_tracks = {}


class TrackHistory:
    
    def __init__(self, max_history_length: int = 100):
        self.history = {}  
        self.max_length = max_history_length
    
    def update(self, track_id: int, frame_num: int, bbox: list, 
               class_id: int, additional_info: dict = None):
        """Add frame data to track history"""
        if track_id not in self.history:
            self.history[track_id] = []
        
        
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        frame_data = {
            'frame': frame_num,
            'bbox': bbox,
            'center': (cx, cy),
            'class_id': class_id
        }
        
        if additional_info:
            frame_data.update(additional_info)
        
        self.history[track_id].append(frame_data)
        
        
        if len(self.history[track_id]) > self.max_length:
            self.history[track_id].pop(0)
    
    def get_history(self, track_id: int) -> List[Dict]:
        """Get full history for a track"""
        return self.history.get(track_id, [])
    
    def get_trajectory(self, track_id: int) -> List[tuple]:
        """Get list of center points for a track"""
        history = self.get_history(track_id)
        return [frame['center'] for frame in history]
    
    def get_track_length(self, track_id: int) -> int:
        """Get number of frames in track history"""
        return len(self.history.get(track_id, []))
    
    def cleanup_old_tracks(self, active_track_ids: List[int]):
        """Remove tracks that are no longer active"""
        inactive_ids = set(self.history.keys()) - set(active_track_ids)
        for track_id in inactive_ids:
            del self.history[track_id]
    
    def get_all_track_ids(self) -> List[int]:
        """Get all track IDs in history"""
        return list(self.history.keys())