import os
import json
import logging
from typing import List, Dict, Any, Optional
from PIL import Image
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logger = logging.getLogger(__name__)

# Load API Key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class Agent:
    def __init__(self, name: str, role: str, model_name: str = "gemini-2.5-flash"):
        self.name = name
        self.role = role
        self.model = genai.GenerativeModel(model_name)
        self.logger = logging.getLogger(f"Agent-{name}")

    def think(self, prompt: str, images: List[Image.Image] = [], context: Dict = {}) -> Dict[str, Any]:
        """Base thinking method. Returns JSON."""
        try:
            full_prompt = [
                f"Role: {self.role}",
                f"Context: {json.dumps(context)}",
                "Task: " + prompt,
                "Return valid JSON only."
            ] + images
            
            response = self.model.generate_content(
                full_prompt, 
                generation_config={"response_mime_type": "application/json"}
            )
            return json.loads(response.text)
        except Exception as e:
            self.logger.error(f"Thinking failed: {e}")
            return {"error": str(e)}

class GeometryAgent(Agent):
    def __init__(self):
        super().__init__("Geometry", "You are an expert in Projective Geometry and Computer Vision. Your ONLY job is to determine the 3D perspective of the pitch and draw lines PARALLEL to the goal line.")

    def analyze_perspective(self, frame: Image.Image) -> Dict[str, Any]:
        """
        Identifies the offside line coordinates.
        Uses pure geometric reasoning (or suggests code - simplified here to direct analysis for speed).
        """
        prompt = """
        Analyze the image and identify the 'Offside Line' based on the last defender.
        
        CRITICAL: 
        1. Identify the goal line (or implied goal line).
        2. Identify the rearmost geometric point of the second-last opponent (the defender).
        3. Project a line through that point PARALLEL to the goal line.
        
        Return JSON:
        {
            "offside_line": [start_x, start_y, end_x, end_y] (0-1 normalized),
            "vanishing_point": [x, y],
            "confidence": float
        }
        """
        return self.think(prompt, [frame])

class VisionAgent(Agent):
    def __init__(self):
        super().__init__("Vision", "You are an expert Sports Analyst. Your job is to identify players, the ball, and their exact body positions.")

    def detect_entities(self, frame: Image.Image) -> Dict[str, Any]:
        prompt = """
        Detect the following entities:
        1. The Ball (and distinct 'kick point' moment if visible).
        2. The Attacker (involved in the play).
        3. The Second-Last Defender (setting the line).
        
        For the Attacker and Defender, identify the specific body part closest to the goal line (Head, Foot, Knee).
        
        Return JSON:
        {
            "attacker": {"box": [ymin, xmin, ymax, xmax], "label": "Attacker (Body Part)"},
            "defender": {"box": [ymin, xmin, ymax, xmax], "label": "Defender (Body Part)"},
            "ball": {"box": [ymin, xmin, ymax, xmax]}
        }
        """
        return self.think(prompt, [frame])

class RulesAgent(Agent):
    def __init__(self):
        super().__init__("Rules", "You are a FIFA Certified Referee. You interpret Law 11 (Offside) and Law 12 (Fouls/Handball). You do NOT draw lines; you adjudicate based on data.")

    def adjudicate(self, geometry_data: Dict, vision_data: Dict) -> Dict[str, Any]:
        prompt = f"""
        Adjudicate this play based on the provided data.
        
        Geometry Data: {json.dumps(geometry_data)}
        Vision Data: {json.dumps(vision_data)}
        
        Law 11 Basics:
        - Offside if any part of head, body, or feet is nearer to goal line than both ball and second-last opponent.
        - Hands/Arms do not count.
        
        Make a decision.
        
        Return JSON:
        {{
            "decision": "OFFSIDE" | "ONSIDE",
            "reasoning": "cite specific Law clause",
            "confidence": float
        }}
        """
        # No image needed, pure logic on data
        return self.think(prompt)

class MultiAgentOrchestrator:
    def __init__(self):
        # Use Flash for specialized sub-tasks (Higher Quota)
        self.geometry_agent = GeometryAgent()
        self.vision_agent = VisionAgent()
        self.rules_agent = RulesAgent()
        # Use Pro only for final synthesis where high reasoning is critical
        self.manager = Agent("Manager", "You are the VAR Process Coordinator.", model_name="gemini-2.5-flash")
        self.synthesizer = Agent("Synthesizer", "Final VAR Judge.", model_name="gemini-2.5-pro")

    def process_frame(self, frame: Image.Image) -> Dict[str, Any]:
        """
        Map Step: specific agents analyze the frame in parallel.
        """
        with ThreadPoolExecutor() as executor:
            future_geo = executor.submit(self.geometry_agent.analyze_perspective, frame)
            future_vis = executor.submit(self.vision_agent.detect_entities, frame)
            
            geo_result = future_geo.result() or {}
            vis_result = future_vis.result() or {}
            
        # Reduce Step 1: Adjudicate per frame
        rule_result = self.rules_agent.adjudicate(geo_result, vis_result)
        
        return {
            "geometry": geo_result,
            "vision": vis_result,
            "rule_verdict": rule_result or {"decision": "UNCLEAR", "error": "Rule agent failed"}
        }

    def detect_critical_moments(self, frames: List[Image.Image]) -> List[int]:
        """
        Manager Agent scans all frames to find critical ballplays (passes, shots, deflections).
        Returns indices of frames that need deep analysis.
        """
        prompt = """
        Review these video frames. Identify ONLY the "Critical Ballplays" relevant to VAR.
        
        Critical definitions:
        1. The exact moment the ball is played (kicked/headed) by an attacker.
        2. The moment of a potential handball or foul.
        
        Ignore frames where the ball is just traveling or nothing is happening.
        
        Return JSON:
        {
            "critical_frame_indices": [int, int, ...],
            "reasoning": "string"
        }
        """
        # We assume the manager uses a faster model for this scan or we batch it
        # using the manager's default model (Gemini 2.5 Pro) is fine for high quality selection
        result = self.manager.think(prompt, frames)
        indices = result.get("critical_frame_indices", [])
        logger.info(f"Manager detected critical frames: {indices}")
        return indices

    def process_clip(self, frames: List[Image.Image]) -> Dict[str, Any]:
        """
        Coordinator Workflow:
        1. Manager scans video for Critical Moments.
        2. Swarm analyzes ONLY those critical frames.
        3. Synthesis of the final verdict.
        """
        # Step 1: Critical Play Detection
        critical_indices = self.detect_critical_moments(frames)
        
        if not critical_indices:
            logger.warning("No critical moments found. Analyzing middle frame as fallback.")
            critical_indices = [len(frames) // 2]
            
        frame_results = []
        
        # Step 2: Parallel Swarm Execution for Critical Frames ONLY
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Map indices to futures
            futures = {executor.submit(self.process_frame, frames[i]): i for i in critical_indices}
            
            for future in as_completed(futures):
                i = futures[future]
                try:
                    res = future.result()
                    res['frame_index'] = i
                    frame_results.append(res)
                except Exception as e:
                    logger.error(f"Frame {i} failed: {e}")

        # Step 3: Final Synthesis
        synthesis_prompt = f"""
        Review the swarm reports from the CRITICAL moments ID'd by the Manager.
        
        Reports: {json.dumps(frame_results)}
        
        Determine the final VAR decision.
        
        Return JSON:
        {{
            "decision": "OFFSIDE" | "ONSIDE" | "UNCLEAR",
            "confidence": float,
            "explanation": "Summary of the findings",
            "visual_cues": "Description of the key frame",
            "annotated_frames": [] (Placeholder)
        }}
        """
        
        final_verdict = self.synthesizer.think(synthesis_prompt)
        
        # Merge data logic
        # Pick the most impactful frame (e.g. first offside or first critical)
        if not frame_results:
             return {"decision": "UNCLEAR", "error": "No frames analyzed"}

        # Find best frame to show
        best_frame_data = frame_results[0]
        for res in frame_results:
            if res.get('rule_verdict', {}).get('decision') == 'OFFSIDE':
                best_frame_data = res
                break
        
        final_verdict['entities'] = []
        
        vis = best_frame_data.get('vision', {})
        geo = best_frame_data.get('geometry', {})
        
        # Defensive entity mapping
        offside_line = geo.get('offside_line')
        if offside_line and isinstance(offside_line, list) and len(offside_line) >= 4:
             final_verdict['entities'].append({
                 "label": "Offside Line", 
                 "box_2d": offside_line
             })
             
        attacker = vis.get('attacker', {})
        if attacker and attacker.get('box'):
            final_verdict['entities'].append({"label": "Attacker", "box_2d": attacker.get('box')})
            
        defender = vis.get('defender', {})
        if defender and defender.get('box'):
             final_verdict['entities'].append({"label": "Defender", "box_2d": defender.get('box')})

        return final_verdict
