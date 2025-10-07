import cv2
import numpy as np
from typing import Dict, List, Optional


class TrafficVisualizer:
    """Visualize detection, tracking, and analysis results"""
    
    def __init__(self, 
                 show_bbox: bool = True,
                 show_id: bool = True,
                 show_class: bool = True,
                 show_direction: bool = True,
                 show_stats: bool = True):

        self.show_bbox = show_bbox
        self.show_id = show_id
        self.show_class = show_class
        self.show_direction = show_direction
        self.show_stats = show_stats
        
        
        self.colors = {
            'normal': (0, 255, 0),      
            'wrong_way': (0, 0, 255),   
            'person': (255, 255, 0),    
            'line': (255, 255, 0),      
            'text': (255, 255, 255)     
        }
        
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
    
    def draw_track(self, frame: np.ndarray, track_info: Dict, 
                   class_name: str, direction: str = None,
                   is_wrong_way: bool = False,
                   trajectory: List[tuple] = None) -> np.ndarray:

        track_id = track_info['track_id']
        bbox = track_info['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        
        
        if is_wrong_way:
            color = self.colors['wrong_way']
        elif class_name == "person":
            color = self.colors['person']
        else:
            color = self.colors['normal']
        
        
        if self.show_bbox:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)
        
        
        label_parts = []
        if self.show_id:
            label_parts.append(f"ID:{track_id}")
        if self.show_class:
            label_parts.append(class_name)
        if self.show_direction and direction:
            label_parts.append(f"→{direction}")
        
        if is_wrong_way:
            label_parts.append("[WRONG WAY]")
        
        label = " ".join(label_parts)
        
        
        label_size, _ = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)
        label_w, label_h = label_size
        
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), self.font, 
                   self.font_scale, self.colors['text'], self.thickness)
        
        
        if trajectory and len(trajectory) > 1:
            pts = np.array(trajectory, dtype=np.int32)
            cv2.polylines(frame, [pts], False, color, 2)
        
        return frame
    
    def draw_counting_line(self, frame: np.ndarray, 
                          line_y: int = None, 
                          line_x: int = None) -> np.ndarray:
        """Draw counting line"""
        height, width = frame.shape[:2]
        
        if line_x is None:
            
            cv2.line(frame, (0, line_y), (width, line_y), 
                    self.colors['line'], 3)
            cv2.putText(frame, "COUNTING LINE", (10, line_y - 10),
                       self.font, 0.7, self.colors['line'], 2)
        else:
            
            cv2.line(frame, (line_x, 0), (line_x, height),
                    self.colors['line'], 3)
            cv2.putText(frame, "COUNT", (line_x + 10, 30),
                       self.font, 0.7, self.colors['line'], 2)
        
        return frame
    
    def draw_statistics(self, frame: np.ndarray, 
                       stats: Dict,
                       fps: float = 0.0,
                       frame_num: int = 0) -> np.ndarray:

        if not self.show_stats:
            return frame
        
        
        height, width = frame.shape[:2]
        overlay = frame.copy()
        panel_height = 200
        cv2.rectangle(overlay, (0, 0), (350, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        
        y_offset = 30
        text_color = self.colors['text']
        
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_offset),
                   self.font, 0.7, text_color, 2)
        y_offset += 30
        cv2.putText(frame, f"Frame: {frame_num}", (10, y_offset),
                   self.font, 0.7, text_color, 2)
        y_offset += 40
        
        
        cv2.putText(frame, "=== VEHICLES ===", (10, y_offset),
                   self.font, 0.6, (255, 255, 0), 2)
        y_offset += 25
        
        for vehicle_type, count in stats.get('vehicle_breakdown', {}).items():
            cv2.putText(frame, f"{vehicle_type}: {count}", (10, y_offset),
                       self.font, 0.6, text_color, 2)
            y_offset += 25
        
        
        cv2.putText(frame, f"Total: {stats.get('total_vehicles', 0)}", (10, y_offset),
                   self.font, 0.7, (0, 255, 0), 2)
        y_offset += 30
        
        
        cv2.putText(frame, f"Persons: {stats.get('total_persons', 0)}", (10, y_offset),
                   self.font, 0.6, self.colors['person'], 2)
        y_offset += 30
        
        
        if stats.get('wrong_way_count', 0) > 0:
            cv2.putText(frame, f"Wrong Way: {stats['wrong_way_count']}", (10, y_offset),
                       self.font, 0.6, self.colors['wrong_way'], 2)
        
        return frame
    
    def create_legend(self, frame: np.ndarray) -> np.ndarray:
        """Draw color legend"""
        height, width = frame.shape[:2]
        x_start = width - 250
        y_start = 20
        
        legends = [
            ("Normal", self.colors['normal']),
            ("Wrong Way", self.colors['wrong_way']),
            ("Person", self.colors['person'])
        ]
        
        for i, (label, color) in enumerate(legends):
            y = y_start + i * 30
            cv2.rectangle(frame, (x_start, y), (x_start + 20, y + 20), color, -1)
            cv2.putText(frame, label, (x_start + 30, y + 15),
                       self.font, 0.5, self.colors['text'], 2)
        
        return frame