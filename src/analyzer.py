"""
Traffic Analysis Module
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class TrafficAnalyzer:
    """Analyze traffic patterns, count vehicles, detect violations"""
    
    def __init__(self, counting_line_y: int = 400, 
                 counting_line_x: Optional[int] = None,
                 expected_direction: str = "right",
                 min_track_length: int = 10):

        self.counting_line_y = counting_line_y
        self.counting_line_x = counting_line_x
        self.expected_direction = expected_direction
        self.min_track_length = min_track_length
        
        
        self.counted_ids = set()
        self.vehicle_counts = defaultdict(int)
        self.person_count = 0
        
        
        self.wrong_way_ids = set()
        self.violation_logs = []
        
        
        self.total_detections = 0
        
        print(f" Analyzer initialized")
        print(f"  - Counting line: y={counting_line_y}, x={counting_line_x}")
        print(f"  - Expected direction: {expected_direction}")
    
    def analyze_track(self, track_id: int, track_history: List[Dict]) -> Dict:
        if len(track_history) < self.min_track_length:
            return {
                'direction': None,
                'is_wrong_way': False,
                'should_count': False
            }
        
        
        direction = self._calculate_direction(track_history)
        
        
        is_wrong_way = self._is_wrong_way(direction)
        
        
        should_count = self._check_crossing(track_id, track_history)
        
        return {
            'direction': direction,
            'is_wrong_way': is_wrong_way,
            'should_count': should_count
        }
    
    def _calculate_direction(self, track_history: List[Dict]) -> str:
        """Calculate primary movement direction"""
        if len(track_history) < 2:
            return "unknown"
        
        
        start_pos = track_history[0]['center']
        end_pos = track_history[-1]['center']
        
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        
        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "down" if dy > 0 else "up"
    
    def _is_wrong_way(self, direction: str) -> bool:
        """Check if direction is wrong way"""
        if direction == "unknown":
            return False
        
        
        opposites = {
            "up": "down",
            "down": "up",
            "left": "right",
            "right": "left"
        }
        
        return direction == opposites.get(self.expected_direction, None)
    
    def _check_crossing(self, track_id: int, track_history: List[Dict]) -> bool:
        if track_id in self.counted_ids:
            return False
        
        if len(track_history) < 2:
            return False
        
        
        if self.counting_line_x is None:
            
            for i in range(len(track_history) - 1):
                y1 = track_history[i]['center'][1]
                y2 = track_history[i + 1]['center'][1]
                
                
                if (y1 < self.counting_line_y <= y2) or (y1 > self.counting_line_y >= y2):
                    return True
        else:
            
            for i in range(len(track_history) - 1):
                x1 = track_history[i]['center'][0]
                x2 = track_history[i + 1]['center'][0]
                
                if (x1 < self.counting_line_x <= x2) or (x1 > self.counting_line_x >= x2):
                    return True
        
        return False
    
    def count_vehicle(self, track_id: int, class_name: str):
        if track_id not in self.counted_ids:
            self.counted_ids.add(track_id)
            self.vehicle_counts[class_name] += 1
    
    def count_person(self, track_id: int):
        if track_id not in self.counted_ids:
            self.counted_ids.add(track_id)
            self.person_count += 1
    
    def log_violation(self, frame_num: int, track_id: int, 
                      violation_type: str, details: Dict):
        violation = {
            'frame': frame_num,
            'track_id': track_id,
            'type': violation_type,
            'details': details
        }
        self.violation_logs.append(violation)
        
        if violation_type == "wrong_way":
            self.wrong_way_ids.add(track_id)
    
    def get_statistics(self) -> Dict:
        total_vehicles = sum(self.vehicle_counts.values())
        
        return {
            'total_vehicles': total_vehicles,
            'vehicle_breakdown': dict(self.vehicle_counts),
            'total_persons': self.person_count,
            'wrong_way_count': len(self.wrong_way_ids),
            'total_violations': len(self.violation_logs),
            'unique_tracks': len(self.counted_ids)
        }
    
    def get_violations(self) -> List[Dict]:
        return self.violation_logs
    
    def reset(self):
        self.counted_ids = set()
        self.vehicle_counts = defaultdict(int)
        self.person_count = 0
        self.wrong_way_ids = set()
        self.violation_logs = []
        self.total_detections = 0


class SpeedEstimator:
    def __init__(self, fps: int, pixel_per_meter: float = 10.0):
        self.fps = fps
        self.pixel_per_meter = pixel_per_meter
    
    def estimate_speed(self, track_history: List[Dict], 
                       window_frames: int = 10) -> float:

        if len(track_history) < window_frames:
            return 0.0
        
        
        recent_history = track_history[-window_frames:]
        
        
        total_distance_pixels = 0
        for i in range(len(recent_history) - 1):
            p1 = recent_history[i]['center']
            p2 = recent_history[i + 1]['center']
            
            distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            total_distance_pixels += distance
        
        
        distance_meters = total_distance_pixels / self.pixel_per_meter
        
        
        time_seconds = window_frames / self.fps
        
        
        if time_seconds > 0:
            speed_ms = distance_meters / time_seconds
            speed_kmh = speed_ms * 3.6
            return speed_kmh
        
        return 0.0