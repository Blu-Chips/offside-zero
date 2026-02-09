"""
Overlay Engine for Offside Zero
Draws offside lines, zones, and annotations on video frames
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional


class OverlayEngine:
    """Draws VAR-style overlays on football frames."""
    
    # Colors (BGR for OpenCV, RGB for PIL)
    COLORS = {
        "offside_line": (0, 0, 255),      # Red
        "onside_line": (0, 255, 0),        # Green
        "ball": (255, 255, 0),             # Cyan
        "attacker": (255, 0, 0),           # Blue
        "defender": (0, 255, 0),           # Green
        "violation": (0, 0, 255),          # Red
        "text_bg": (0, 0, 0),              # Black
        "text": (255, 255, 255),           # White
    }
    
    def __init__(self, width: int, height: int):
        """Initialize with frame dimensions."""
        self.width = width
        self.height = height
    
    def draw_offside_line(
        self,
        frame: np.ndarray,
        y_normalized: float,
        color: Tuple[int, int, int] = None,
        thickness: int = 3,
        label: str = "OFFSIDE LINE"
    ) -> np.ndarray:
        """
        Draw a horizontal offside line across the frame.
        
        Args:
            frame: Input frame (numpy array)
            y_normalized: Y position (0-1, where 0 is top)
            color: Line color (BGR)
            thickness: Line thickness
            label: Label text
        
        Returns:
            Frame with overlay
        """
        frame = frame.copy()
        color = color or self.COLORS["offside_line"]
        
        y = int(y_normalized * self.height)
        
        # Draw dashed line
        dash_length = 20
        gap_length = 10
        x = 0
        while x < self.width:
            x_end = min(x + dash_length, self.width)
            cv2.line(frame, (x, y), (x_end, y), color, thickness)
            x += dash_length + gap_length
        
        # Add label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, 2)
        
        # Background for text
        cv2.rectangle(
            frame,
            (10, y - text_height - 10),
            (10 + text_width + 10, y + 5),
            self.COLORS["text_bg"],
            -1
        )
        cv2.putText(frame, label, (15, y - 5), font, font_scale, color, 2)
        
        return frame
    
    def draw_player_marker(
        self,
        frame: np.ndarray,
        x_normalized: float,
        y_normalized: float,
        player_id: str = "",
        is_violation: bool = False,
        is_attacker: bool = True
    ) -> np.ndarray:
        """Draw a marker on a player."""
        frame = frame.copy()
        
        x = int(x_normalized * self.width)
        y = int(y_normalized * self.height)
        
        if is_violation:
            color = self.COLORS["violation"]
            radius = 30
        elif is_attacker:
            color = self.COLORS["attacker"]
            radius = 20
        else:
            color = self.COLORS["defender"]
            radius = 20
        
        # Draw circle around player
        cv2.circle(frame, (x, y), radius, color, 3)
        
        # Draw player ID if provided
        if player_id:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, str(player_id), (x - 10, y - radius - 10), 
                       font, 0.6, color, 2)
        
        return frame
    
    def draw_ball_marker(
        self,
        frame: np.ndarray,
        x_normalized: float,
        y_normalized: float
    ) -> np.ndarray:
        """Draw a marker on the ball."""
        frame = frame.copy()
        
        x = int(x_normalized * self.width)
        y = int(y_normalized * self.height)
        
        # Draw crosshair on ball
        color = self.COLORS["ball"]
        size = 15
        thickness = 2
        
        cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
        cv2.line(frame, (x, y - size), (x, y + size), color, thickness)
        cv2.circle(frame, (x, y), size, color, thickness)
        
        return frame
    
    def draw_decision_banner(
        self,
        frame: np.ndarray,
        decision: str,
        confidence: float,
        explanation: str = ""
    ) -> np.ndarray:
        """Draw a VAR-style decision banner."""
        frame = frame.copy()
        
        # Banner background
        banner_height = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, banner_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Decision text
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        if decision == "OFFSIDE":
            color = (0, 0, 255)  # Red
            text = "⚠ OFFSIDE"
        elif decision == "HANDBALL":
            color = (0, 0, 255)  # Red
            text = "⚠ HANDBALL"
        elif decision == "NO_VIOLATION":
            color = (0, 255, 0)  # Green
            text = "✓ NO VIOLATION"
        else:
            color = (255, 255, 0)  # Yellow
            text = "? REVIEWING..."
        
        cv2.putText(frame, text, (20, 45), font, 1.5, color, 3)
        
        # Confidence
        conf_text = f"Confidence: {confidence:.0%}"
        cv2.putText(frame, conf_text, (self.width - 250, 45), font, 0.8, (255, 255, 255), 2)
        
        # Explanation (if fits)
        if explanation and len(explanation) < 80:
            cv2.putText(frame, explanation[:80], (20, 70), font, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def draw_handball_zone(
        self,
        frame: np.ndarray,
        x_normalized: float,
        y_normalized: float,
        radius_normalized: float = 0.05
    ) -> np.ndarray:
        """Highlight handball contact area."""
        frame = frame.copy()
        
        x = int(x_normalized * self.width)
        y = int(y_normalized * self.height)
        radius = int(radius_normalized * min(self.width, self.height))
        
        # Draw attention circle
        color = self.COLORS["violation"]
        cv2.circle(frame, (x, y), radius, color, 3)
        cv2.circle(frame, (x, y), radius + 10, color, 2)
        
        # Add "HANDBALL" label
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "HANDBALL", (x - 50, y - radius - 20), font, 0.7, color, 2)
        
        return frame
    
    def create_annotated_frame(
        self,
        frame: np.ndarray,
        analysis: Dict
    ) -> np.ndarray:
        """
        Create fully annotated frame based on Gemini analysis.
        
        Args:
            frame: Input frame
            analysis: Analysis result from GeminiAnalyzer
        
        Returns:
            Annotated frame
        """
        result = frame.copy()
        
        # Parse 'entities' from Gemini 2.0 schema
        entities = analysis.get("entities", [])
        
        # 1. Draw Offside Line first (background layer)
        for entity in entities:
            if entity.get("label") == "Offside Line":
                # Assuming box_2d [ymin, xmin, ymax, xmax], we take average Y
                box = entity.get("box_2d")
                if box and isinstance(box, list) and len(box) >= 4:
                    y_norm = (box[0] + box[2]) / 2
                    result = self.draw_offside_line(result, y_norm)

        # 2. Draw Players
        for entity in entities:
            label = entity.get("label", "")
            if label in ["Attacker", "Defender"]:
                box = entity.get("box_2d")
                if box and isinstance(box, list) and len(box) >= 4:
                    # Center point
                    y_center = (box[0] + box[2]) / 2
                    x_center = (box[1] + box[3]) / 2
                    
                    is_attacker = (label == "Attacker")
                    is_violation = is_attacker and analysis.get("decision") == "OFFSIDE"
                    
                    result = self.draw_player_marker(
                        result,
                        x_center,
                        y_center,
                        player_id=entity.get("id", ""),
                        is_violation=is_violation,
                        is_attacker=is_attacker
                    )

        # 3. Draw Ball
        for entity in entities:
            if entity.get("label") == "Ball":
                box = entity.get("box_2d")
                if box and isinstance(box, list) and len(box) >= 4:
                    y_center = (box[0] + box[2]) / 2
                    x_center = (box[1] + box[3]) / 2
                    result = self.draw_ball_marker(result, x_center, y_center)
        
        # 5. Draw decision banner
        result = self.draw_decision_banner(
            result,
            analysis.get("decision", "UNCLEAR"),
            analysis.get("confidence", 0.0),
            str(analysis.get("explanation", ""))[:80]
        )
        
        return result


def create_overlay_engine(width: int, height: int) -> OverlayEngine:
    """Factory function."""
    return OverlayEngine(width, height)
