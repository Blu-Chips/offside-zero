import os
import cv2
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

# New MAS Import
from src.multi_agent_system import MultiAgentOrchestrator
from src.video_processor import load_video
from src.overlay import create_overlay_engine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnalysisService:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Initialize MAS Orchestrator
        self.orchestrator = MultiAgentOrchestrator()

    def analyze_clip(
        self,
        video_path: str,
        timestamp: Optional[float] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a video clip using the Multi-Agent Swarm.
        """
        logger.info(f"Starting SWARM analysis for: {video_path}")
        
        if not os.path.exists(video_path):
             return {"error": f"Video not found: {video_path}"}

        try:
            with load_video(video_path) as video:
                info = video.get_info()
                
                # 1. Extract Frames
                # For MAS, we want high quality keyframes
                if timestamp is not None:
                    frames = video.extract_frames_around(timestamp, window_seconds=1.0, num_frames=3)
                else:
                    # Select 3 evenly spaced frames for efficiency
                    total = info['frame_count']
                    indices = [total // 4, total // 2, (total * 3) // 4]
                    frames = []
                    for i in indices:
                        frame = video.extract_frame(i)
                        if frame is not None:
                            frames.append(frame)
                
                if not frames:
                    return {"error": "Could not extract frames"}
                
                pil_frames = video.frames_to_pil(frames)
                
                # 2. Orchestrate Swarm Analysis
                logger.info("Dispatching to Agent Swarm...")
                analysis = self.orchestrator.process_clip(pil_frames)
                
                # 3. Create Annotated Output (Simplified for now)
                # We'll annotate the first frame based on the result
                overlay_engine = create_overlay_engine(info['width'], info['height'])
                annotated_paths = []
                base_name = Path(video_path).stem
                
                for i, frame in enumerate(frames):
                    # Only annotate if we have entity data
                    # (Current logic maps best data to final result, we might need per-frame mapping later)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    annotated = overlay_engine.create_annotated_frame(frame_bgr, analysis)
                    
                    out_filename = f"{base_name}_swarm_{i}.jpg"
                    out_path = os.path.join(self.output_dir, out_filename)
                    cv2.imwrite(out_path, annotated)
                    annotated_paths.append(out_filename)
                
                analysis["annotated_frames"] = annotated_paths
                
                # Save JSON
                json_path = os.path.join(self.output_dir, f"{base_name}_swarm_analysis.json")
                with open(json_path, "w") as f:
                    json.dump(analysis, f, indent=2)
                
                return analysis

        except Exception as e:
            logger.error(f"Swarm Analysis failed: {e}", exc_info=True)
            return {"error": str(e)}

_service_instance = None

def get_analysis_service(output_dir: str = "output"):
    global _service_instance
    if _service_instance is None:
        _service_instance = AnalysisService(output_dir)
    return _service_instance
