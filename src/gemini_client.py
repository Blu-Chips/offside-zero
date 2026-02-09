import os
import google.generativeai as genai
import json
from typing import List, Dict, Any, Optional
from PIL import Image
from dotenv import load_dotenv
from pathlib import Path
import logging

## Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Load Rules
try:
    BRAIN_DIR = r"C:\Users\JAMES\.gemini\antigravity\brain\b9612e86-52c6-4d3f-829e-0ebcb0fff33c" # Ideally dynamic, but hardcoded for this env based on artifact path
    RULES_PATH = os.path.join(BRAIN_DIR, "fifa_rules.md")
    with open(RULES_PATH, 'r') as f:
        FIFA_RULES = f.read()
except Exception as e:
    logger.warning(f"Could not load FIFA rules: {e}")
    FIFA_RULES = "Standard FIFA Offside and Handball rules apply."

class GeminiClient:
    def __init__(self, model_name: str = None):
        """
        Initialize the Client. Exclusively uses Google Gemini models.
        """
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        
        # Google-Exclusive Model List
        default_models = [
            "gemini-2.5-flash",          # Primary: Fast & Multimodal
            "gemini-3-flash-preview",    # Experimental: Next-gen speed
            "gemini-2.5-pro",            # High Reasoning
        ]

        if model_name:
            if isinstance(model_name, str):
                self.models = [model_name]
            else:
                self.models = model_name
        else:
            env_model = os.environ.get("GEMINI_MODEL")
            if env_model:
                self.models = [env_model] + [m for m in default_models if m != env_model]
            else:
                self.models = default_models

        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)

    def _analyze_with_gemini(self, model_name: str, full_prompt: list) -> dict:
        """Call Gemini API."""
        model = genai.GenerativeModel(model_name)
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1, # Lower temperature for stricter rule adherence
        )
        response = model.generate_content(full_prompt, generation_config=generation_config)
        return json.loads(response.text)

    def chat_with_context(self, message: str, context: Dict[str, Any]) -> str:
        """
        Ask a follow-up question about a specific analysis context.
        Uses a high-reasoning model (Gemini 2.5 Pro) for "Supreme Intelligence".
        """
        model_name = "gemini-2.5-pro" # Hardcoded for supreme intelligence
        
        system_prompt = f"""
        You are an elite FIFA-certified VAR AI Expert. 
        You have just analyzed a football clip.
        
        Here is your previous analysis (Context):
        {json.dumps(context, indent=2)}
        
        The user is asking a follow-up question.
        
        CRITICAL RULES:
        1. Verify the specific action (e.g., was it a header, a kick, a deflection?). Correct the user if they premise their question on the wrong action type.
        2. Cite specific FIFA Laws (Law 11 Offside, Law 12 Handball).
        3. Be concise, authoritative, and precise.
        """
        
        full_prompt = [system_prompt, f"User Question: {message}"]
        
        try:
            logger.info(f"Chatting with {model_name}...")
            # Use generate_content for single turn chat (stateless for now)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return "I apologize, but I cannot answer that right now. System error."

    def analyze_frames(
        self, 
        frames: List[Image.Image], 
        prompt: str = "Analyze this football clip for offside and handball violations."
    ) -> Dict[str, Any]:
        """
        Analyze frames with Gemini models using FIFA context.
        """
        system_instruction = f"""
        You are an elite FIFA-certified VAR AI Assistant.
        Your decisions must be based STRICTLY on the following FIFA Laws of the Game:
        
        {FIFA_RULES}
        
        CRITICAL: To prove your decision, you MUST provide geometric data.
        1. Identify the 'offside_line_y' (normalized 0-1) where the last defender is positioned.
        2. Identify key players (Attacker, Defender) with their 'box_2d' [ymin, xmin, ymax, xmax].
        
        Return JSON schema:
        {{
            "decision": "OFFSIDE" | "ONSIDE" | "HANDBALL" | "NO_INFRACTION" | "UNCLEAR",
            "confidence": float,
            "explanation": "string (cite specific Law 11/12 clauses)",
            "visual_cues": "string (describe the geometry)",
            "entities": [
                {{"label": "Offside Line", "box_2d": [y, 0, y, 1]}}, 
                {{"label": "Attacker"|"Defender"|"Ball", "box_2d": [ymin, xmin, ymax, xmax], "id": "string"}}
            ]
        }}
        """
        
        last_exception = None
        for model_name in self.models:
            try:
                logger.info(f"Attempting analysis with model: {model_name}")
                
                full_prompt = [system_instruction, "Provide the JSON analysis with geometric proof.", prompt] + frames
                return self._analyze_with_gemini(model_name, full_prompt)
                    
            except Exception as e:
                logger.error(f"Error with model {model_name}: {e}")
                last_exception = e
                continue
        
        return {
            "decision": "API_ERROR",
            "confidence": 0.0,
            "explanation": f"All models failed. Last error: {str(last_exception)}"
        }

    def analyze_video_segment(self, frames: List[Image.Image], context: str = "") -> Dict[str, Any]:
        return self.analyze_frames(frames, prompt=context)

def get_analyzer(model_name: str = None) -> GeminiClient:
    return GeminiClient(model_name)
