import json
import csv
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List



class TrafficLogger:
    """Log and export traffic analysis results per video"""
    
    def __init__(self, output_dir: str = "output/logs", session_name: str = None):
        """
        Initialize logger

        Args:
            output_dir: Base directory to save logs
            session_name: Video/session name (used as subfolder). If None, use timestamp.
        """
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.session_name = session_name
        self.base_dir = Path(output_dir)
        self.output_dir = self.base_dir / self.session_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logs = []
        
        print(f" Logger initialized: {self.output_dir}")
    
    def log_event(self, frame_num: int, event_type: str, data: Dict):
        """Log a generic event"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'frame': frame_num,
            'event_type': event_type,
            **data
        }
        self.logs.append(log_entry)
    
    def log_crossing(self, frame_num: int, track_id: int, 
                     class_name: str, direction: str, bbox: list):
        """Log vehicle/person crossing"""
        self.log_event(frame_num, 'crossing', {
            'track_id': track_id,
            'class': class_name,
            'direction': direction,
            'bbox': bbox
        })
    
    def log_violation(self, frame_num: int, track_id: int,
                      violation_type: str, details: Dict):
        """Log traffic violation"""
        self.log_event(frame_num, 'violation', {
            'track_id': track_id,
            'violation_type': violation_type,
            **details
        })
    
    def save_csv(self, filename: str = None):
        """Save all logs as CSV"""
        if filename is None:
            filename = f"{self.session_name}.csv"
        
        filepath = self.output_dir / filename
        if not self.logs:
            print("⚠ No logs to save")
            return
        
        df = pd.DataFrame(self.logs)
        df.to_csv(filepath, index=False)
        print(f" CSV saved: {filepath} ({len(self.logs)} entries)")
        return filepath
    
    def save_json(self, filename: str = None):
        """Save all logs as JSON"""
        if filename is None:
            filename = f"{self.session_name}.json"
        
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False)
        print(f" JSON saved: {filepath} ({len(self.logs)} entries)")
        return filepath
    
    def save_summary(self, stats: Dict, filename: str = None):
        """Save summary statistics"""
        if filename is None:
            filename = f"{self.session_name}_summary.json"
        
        filepath = self.output_dir / filename
        summary = {
            'session': self.session_name,
            'generated_at': datetime.now().isoformat(),
            'statistics': stats,
            'total_events': len(self.logs)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f" Summary saved: {filepath}")
        return filepath
    
    def export_violations_only(self, filename: str = None):
        """Export only violation events"""
        violations = [log for log in self.logs if log['event_type'] == 'violation']
        if filename is None:
            filename = f"{self.session_name}_violations.csv"
        
        filepath = self.output_dir / filename
        if violations:
            df = pd.DataFrame(violations)
            df.to_csv(filepath, index=False)
            print(f" Violations exported: {filepath} ({len(violations)} violations)")
        else:
            print("⚠ No violations to export")
        return filepath
    
    def get_statistics(self) -> Dict:
        """Generate statistics from logs"""
        if not self.logs:
            return {}
        
        df = pd.DataFrame(self.logs)
        stats = {
            'total_events': len(self.logs),
            'event_types': df['event_type'].value_counts().to_dict(),
        }
        
        
        crossings = df[df['event_type'] == 'crossing']
        if not crossings.empty:
            stats['total_crossings'] = len(crossings)
            stats['crossings_by_class'] = crossings['class'].value_counts().to_dict()
            stats['crossings_by_direction'] = crossings['direction'].value_counts().to_dict()
        
        
        violations = df[df['event_type'] == 'violation']
        if not violations.empty:
            stats['total_violations'] = len(violations)
            stats['violations_by_type'] = violations['violation_type'].value_counts().to_dict()
        
        return stats
    
    def clear_logs(self):
        """Clear all logs"""
        self.logs.clear()
    
    def export_all(self, stats: Dict = None):
        """Export all formats"""
        print("\n📊 Exporting logs...")
        self.save_csv()
        self.save_json()
        if stats:
            self.save_summary(stats)
        self.export_violations_only()
        print(" All logs exported successfully\n")



class PerformanceLogger:
    """Log system performance metrics"""
    
    def __init__(self):
        """Initialize performance logger"""
        self.metrics = {
            'fps': [],
            'detection_time': [],
            'tracking_time': [],
            'total_time': []
        }
    
    def log_frame_metrics(self, fps: float, detection_time: float = 0,
                          tracking_time: float = 0, total_time: float = 0):
        """Log metrics for a single frame"""
        self.metrics['fps'].append(fps)
        self.metrics['detection_time'].append(detection_time)
        self.metrics['tracking_time'].append(tracking_time)
        self.metrics['total_time'].append(total_time)
    
    def get_average_metrics(self) -> Dict:
        """Get average performance metrics"""
        import numpy as np
        
        return {
            'avg_fps': np.mean(self.metrics['fps']) if self.metrics['fps'] else 0,
            'avg_detection_time': np.mean(self.metrics['detection_time']) if self.metrics['detection_time'] else 0,
            'avg_tracking_time': np.mean(self.metrics['tracking_time']) if self.metrics['tracking_time'] else 0,
            'avg_total_time': np.mean(self.metrics['total_time']) if self.metrics['total_time'] else 0,
            'min_fps': np.min(self.metrics['fps']) if self.metrics['fps'] else 0,
            'max_fps': np.max(self.metrics['fps']) if self.metrics['fps'] else 0
        }
    
    def save_performance_report(self, output_path: str):
        """Save performance report"""
        metrics = self.get_average_metrics()
        
        with open(output_path, 'w') as f:
            f.write("=== Performance Report ===\n\n")
            f.write(f"Average FPS: {metrics['avg_fps']:.2f}\n")
            f.write(f"Min FPS: {metrics['min_fps']:.2f}\n")
            f.write(f"Max FPS: {metrics['max_fps']:.2f}\n")
            f.write(f"Avg Detection Time: {metrics['avg_detection_time']*1000:.2f}ms\n")
            f.write(f"Avg Tracking Time: {metrics['avg_tracking_time']*1000:.2f}ms\n")
            f.write(f"Avg Total Time: {metrics['avg_total_time']*1000:.2f}ms\n")
        
        print(f" Performance report saved: {output_path}")