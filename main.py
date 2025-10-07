import yaml
import cv2
import time
from pathlib import Path
from tqdm import tqdm
import threading
from queue import Queue

from src.detector import VehicleDetector
from src.tracker import ObjectTracker, TrackHistory
from src.analyzer import TrafficAnalyzer
from src.visualizer import TrafficVisualizer
from utils.video_utils import VideoProcessor
from utils.logger import TrafficLogger


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_video_files(folder: str) -> list:
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    video_folder = Path(folder)
    
    if not video_folder.exists():
        print(f"Folder not found: {folder}")
        video_folder.mkdir(parents=True, exist_ok=True)
        return []
    
    video_files = []
    for f in video_folder.iterdir():
        if f.suffix.lower() in video_extensions:
            video_files.append(str(f))
    
    return sorted(video_files)


def process_single_video(video_path: str, config: dict, video_name: str, log_interval: int = 50):
    """Process a single video file with frame-by-frame logging"""
    print(f"\n{'='*60}")
    print(f"   Processing: {video_name}")
    print(f"{'='*60}\n")
    
    # Setup output paths
    output_video_dir = Path("output/videos")
    output_video_dir.mkdir(parents=True, exist_ok=True)
    output_path = None 
    
    output_log_dir = Path("output/logs")
    output_log_dir.mkdir(parents=True, exist_ok=True)
    
    wrong_way_dir = Path("output/wrong_way") / video_name
    wrong_way_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    video_processor = VideoProcessor(
        video_path=video_path,
        output_path=output_path,
        resize_width=config['video']['resize_width'],
        resize_height=config['video']['resize_height'],
        skip_frames=config['video']['skip_frames']
    )
    
    detector = VehicleDetector(
        model_path=config['model']['yolo_model'],
        conf=config['model']['confidence'],
        device=config['model']['device'],
        classes={
            'vehicle': config['classes']['vehicle'],
            'person': config['classes']['person']
        }
    )
    
    tracker = ObjectTracker(
        max_age=config['tracker']['max_age'],
        n_init=config['tracker']['n_init'],
        max_iou_distance=config['tracker']['max_iou_distance'],
        embedder=config['tracker']['embedder'],
        embedder_gpu=config['tracker']['embedder_gpu'],
        nn_budget=config['tracker']['nn_budget']
    )
    
    track_history = TrackHistory(max_history_length=100)
    
    analyzer = TrafficAnalyzer(
        counting_line_y=config['analysis']['counting_line_y'],
        counting_line_x=None,
        expected_direction=config['analysis']['expected_direction'],
        min_track_length=config['analysis']['min_track_length']
    )
    
    visualizer = TrafficVisualizer(
        show_bbox=True,
        show_id=True,
        show_class=True,
        show_direction=True,
        show_stats=True
    )
    
    logger = TrafficLogger(
        output_dir=str(output_log_dir),
        session_name=video_name
    )
    
    frame_num = 0
    saved_wrong_way_ids = set()
    pbar = tqdm(total=video_processor.frame_count, desc=f"[{video_name}]", position=0)
    
    try:
        while True:
            ret, frame = video_processor.read_frame()
            if not ret:
                break
            
            frame_num += 1
            if frame_num % config['video']['skip_frames'] != 0:
                continue
            
            # Detection
            detections = detector.detect_for_tracking(frame)
            
            # Tracking
            tracks = tracker.update(detections, frame)
            
            # Analyze & visualize each track
            for track in tracks:
                track_id = track['track_id']
                bbox = track['bbox']
                class_id = track['class_id']
                class_name = detector.get_class_name(class_id)
                
                track_history.update(track_id, frame_num, bbox, class_id)
                history = track_history.get_history(track_id)
                analysis = analyzer.analyze_track(track_id, history)
                
                direction = analysis['direction']
                is_wrong_way = analysis['is_wrong_way']
                should_count = analysis['should_count']
                
                if should_count:
                    if detector.is_vehicle(class_id):
                        analyzer.count_vehicle(track_id, class_name)
                        logger.log_crossing(frame_num, track_id, class_name, direction, list(map(int, bbox)))
                    elif detector.is_person(class_id):
                        analyzer.count_person(track_id)
                        logger.log_crossing(frame_num, track_id, "person", direction, list(map(int, bbox)))
                
                if is_wrong_way:
                    analyzer.log_violation(
                        frame_num=frame_num,
                        track_id=track_id,
                        violation_type="wrong_way",
                        details={'class': class_name, 'direction': direction}
                    )
                    logger.log_violation(
                        frame_num=frame_num,
                        track_id=track_id,
                        violation_type="wrong_way",
                        details={'class': class_name, 'direction': direction}
                    )
                    
                    # Crop & save wrong-way object (1 ảnh / track_id)
                    if track_id not in saved_wrong_way_ids:
                        x1, y1, x2, y2 = map(int, bbox)
                        crop = frame[y1:y2, x1:x2]
                        crop_path = wrong_way_dir / f"track_{track_id}.jpg"
                        cv2.imwrite(str(crop_path), crop)
                        saved_wrong_way_ids.add(track_id)

                trajectory = track_history.get_trajectory(track_id)
                frame = visualizer.draw_track(
                    frame=frame,
                    track_info=track,
                    class_name=class_name,
                    direction=direction,
                    is_wrong_way=is_wrong_way,
                    trajectory=trajectory[-30:] if len(trajectory) > 1 else None
                )
            
            # Draw counting line and stats
            frame = visualizer.draw_counting_line(frame, line_y=config['analysis']['counting_line_y'])
            stats = analyzer.get_statistics()
            video_processor.update_fps()
            current_fps = video_processor.get_fps()
            frame = visualizer.draw_statistics(frame, stats, current_fps, frame_num)
            frame = visualizer.create_legend(frame)
            
            # Real-time display
            if config['output']['display_realtime']:
                cv2.imshow(f'Traffic Analysis - {video_name}', frame)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:
                    print(f" User stopped {video_name}")
                    break
            
            # Frame-by-frame logging (every log_interval frames)
            if config['output']['save_logs'] and frame_num % log_interval == 0:
                logger.save_csv(filename=f"{video_name}_progress.csv")
                logger.save_json(filename=f"{video_name}_progress.json")
            
            pbar.update(1)
    
    except KeyboardInterrupt:
        print(f" {video_name} interrupted by user")
    
    except Exception as e:
        print(f"\n Error in {video_name}: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        pbar.close()
        video_processor.release()
        cv2.destroyAllWindows()
        
        # Export logs cuối cùng khi video kết thúc hoặc exception
        if config['output']['save_logs']:
            try:
                logger.export_all(analyzer.get_statistics())
            except Exception as e:
                print(f"Failed to export logs: {e}")
    
    # Final stats
    final_stats = analyzer.get_statistics()
    print(f"\n{'='*60}\n   {video_name} - COMPLETE\n{'='*60}")
    print(f"  Total Vehicles: {final_stats.get('total_vehicles', 0)}")
    for v_type, count in final_stats.get('vehicle_breakdown', {}).items():
        print(f"    - {v_type}: {count}")
    print(f"  Total Persons: {final_stats.get('total_persons', 0)}")
    print(f"  Wrong Way: {final_stats.get('wrong_way_count', 0)}")
    
    return final_stats


def process_video_thread(video_path: str, config: dict, video_name: str, results_queue: Queue):
    """Thread worker for processing video"""
    try:
        stats = process_single_video(video_path, config, video_name)
        results_queue.put((video_name, stats, None))
    except Exception as e:
        results_queue.put((video_name, None, str(e)))


def process_all_videos_parallel(video_files: list, config: dict):
    """Process all videos in parallel using threading"""
    if not video_files:
        print(" No videos to process!")
        return
    
    print(f"\n{'='*60}\n   MULTI-VIDEO PROCESSING\n{'='*60}")
    print(f"Found {len(video_files)} videos\nProcessing in parallel...\n")
    
    threads = []
    results_queue = Queue()
    
    for video_path in video_files:
        video_name = Path(video_path).stem
        thread = threading.Thread(
            target=process_video_thread,
            args=(video_path, config, video_name, results_queue)
        )
        thread.start()
        threads.append(thread)
        time.sleep(1)
    
    for thread in threads:
        thread.join()
    
    # Collect results
    print(f"\n{'='*60}\n   ALL VIDEOS PROCESSED\n{'='*60}\n")
    total_vehicles, total_persons = 0, 0
    
    while not results_queue.empty():
        video_name, stats, error = results_queue.get()
        if error:
            print(f" {video_name}: {error}")
        elif stats:
            print(f" {video_name}: Vehicles={stats.get('total_vehicles',0)}, Persons={stats.get('total_persons',0)}")
            total_vehicles += stats.get('total_vehicles', 0)
            total_persons += stats.get('total_persons', 0)
    
    print(f"\n{'='*60}\n   SUMMARY (ALL VIDEOS)\n{'='*60}")
    print(f"Total Videos: {len(video_files)}")
    print(f"Total Vehicles: {total_vehicles}")
    print(f"Total Persons: {total_persons}")
    print("\n All done!\n")


def main():
    """Main entry point - Auto process all videos in folder"""
    print("\n" + "="*60)
    print("   TRAFFIC ANALYSIS SYSTEM - AUTO MODE")
    print("="*60 + "\n")
    
    try:
        config = load_config('config/config.yaml')
    except FileNotFoundError:
        print(" Config file not found: config/config.yaml")
        return
    
    video_folder = config.get('video', {}).get('input_folder', './videos')
    video_files = get_video_files(video_folder)
    
    if not video_files:
        print(f"No videos found in folder: {video_folder}")
        print("Please add videos and try again.")
        return
    
    print(f"Found {len(video_files)} video(s) in '{video_folder}':")
    for i, video in enumerate(video_files, 1):
        print(f"  {i}. {Path(video).name}")
    
    print("\nStarting parallel processing...\n")
    process_all_videos_parallel(video_files, config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Program interrupted by user (Ctrl+C)")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
