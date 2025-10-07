from ultralytics import YOLO
import numpy as np
from typing import List, Tuple


class VehicleDetector:
    
    def __init__(self, model_path: str = 'yolov8n.pt', conf: float = 0.4, 
                 device: str = 'cpu', classes: dict = None):
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device
        self.classes = classes or {
            'vehicle': [2, 3, 5, 7],  
            'person': [0]
        }
        
        
        self.valid_classes = []
        for cls_list in self.classes.values():
            self.valid_classes.extend(cls_list)
        
        print(f" Detector initialized: {model_path}")
        print(f"  - Device: {device}")
        print(f"  - Confidence: {conf}")
        print(f"  - Classes: {self.valid_classes}")
    
    def detect(self, frame: np.ndarray, verbose: bool = False) -> List[Tuple]:
        results = self.model(frame, conf=self.conf, device=self.device, verbose=verbose)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                cls_id = int(box.cls[0])
                
                
                if cls_id not in self.valid_classes:
                    continue
                
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class_id': cls_id
                })
        
        return detections
    
    def detect_for_tracking(self, frame: np.ndarray) -> List[Tuple]:
        detections = self.detect(frame)
        
        tracking_format = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            w = x2 - x1
            h = y2 - y1
            
            tracking_format.append((
                [x1, y1, w, h],
                det['confidence'],
                det['class_id']
            ))
        
        return tracking_format
    
    def get_class_name(self, class_id: int) -> str:
        """Get class name from ID"""
        class_names = {
            0: "person",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }
        return class_names.get(class_id, "unknown")
    
    def is_vehicle(self, class_id: int) -> bool:
        """Check if class is a vehicle"""
        return class_id in self.classes['vehicle']
    
    def is_person(self, class_id: int) -> bool:
        """Check if class is a person"""
        return class_id in self.classes['person']