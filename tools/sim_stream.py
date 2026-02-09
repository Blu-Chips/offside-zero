import cv2
import requests
import time
import argparse
import os
import sys

def simulate_stream(video_path, server_url, fps=30):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    print(f"Streaming {video_path} to {server_url} at {fps} FPS...")
    
    frame_delay = 1.0 / fps
    frame_count = 0
    
    try:
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("End of video stream.")
                break # Loop? Or stop? Let's stop for now.

            # Encode frame to JPEG
            _, img_encoded = cv2.imencode('.jpg', frame)
            
            # Send to server
            files = {'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
            try:
                response = requests.post(server_url, files=files, timeout=0.5)
                # print(f"Frame {frame_count}: {response.status_code}")
            except Exception as e:
                print(f"Frame {frame_count} failed: {e}")

            frame_count += 1
            
            # Sleep to maintain FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_delay - elapsed)
            time.sleep(sleep_time)
            
            if frame_count % fps == 0:
                print(f"Streamed {frame_count} frames...")

    except KeyboardInterrupt:
        print("Streaming stopped.")
    finally:
        cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a live camera stream.")
    parser.add_argument("--video", required=True, help="Path to source video file")
    parser.add_argument("--url", default="http://localhost:8000/ingest", help="Server ingestion endpoint")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second to stream")
    
    args = parser.parse_args()
    simulate_stream(args.video, args.url, args.fps)
