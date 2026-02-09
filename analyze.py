"""
Main analysis script for Offside Zero
Analyzes video clips for offside and handball violations
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.analysis_service import get_analysis_service


def analyze_clip(
    video_path: str,
    timestamp: float = None,
    output_dir: str = "output",
    model: str = None
) -> dict:
    """
    Analyze a video clip using the shared AnalysisService.
    """
    service = get_analysis_service(output_dir)
    return service.analyze_clip(video_path, timestamp, model_name=model)


def main():
    parser = argparse.ArgumentParser(
        description="Offside Zero - AI-powered football officiating assistant"
    )
    parser.add_argument(
        "--video", "-v",
        required=True,
        help="Path to video file to analyze"
    )
    parser.add_argument(
        "--timestamp", "-t",
        type=float,
        default=None,
        help="Specific timestamp (seconds) to analyze"
    )
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Model to use (defaults to GEMINI_MODEL env var or gemini-2.0-flash)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)
    
    # Allow env override when CLI not provided
    model_name = args.model or os.environ.get("GEMINI_MODEL")

    result = analyze_clip(
        args.video,
        timestamp=args.timestamp,
        output_dir=args.output,
        model=model_name
    )
    
    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)
    
    print("\nAnalysis complete!")
    print(f"   Decision: {result.get('decision')}")
    print(f"   Confidence: {result.get('confidence', 0):.0%}")
    print(f"   Explanation: {result.get('explanation', 'N/A')[:100]}...")
    
    if "slowmo_video" in result:
         print(f"   Slow-motion replay saved: {result['slowmo_video']}")


if __name__ == "__main__":
    main()
