# Offside Zero

**Realtime AI-assisted officiating that eliminates VAR controversy using Gemini 3.**

## Overview

Offside Zero uses the **Gemini 3 API** as the sole AI engine to analyze recorded football matches and make **explainable** decisions for **offside** and **handball** incidents. It overlays the correct lines/zones on the pitch, generates a slow-motion replay, and sends a decision to a referee/fan dashboard.

## MVP Scope

- **Input:** Recorded match footage
- **Detect:** Offside + Handball
- **Output:** Slow-mo annotated replay + live decision dashboard overlay

## Architecture

```
Video Input → Gemini 3 Vision + Reasoning → Overlay Engine → Dashboard Output
```

Gemini 3 performs:
- Spatial-temporal reasoning (player/ball positions, lines, zones)
- Rule-aware decisioning (offside/handball logic)
- Tool-calling to draw overlays, generate slow-mo, update dashboard
- Explainable decisions + confidence scoring

## Tech Stack

- **AI Engine:** Gemini 3 API (multimodal vision + reasoning)
- **Backend:** Python / Node.js
- **Frontend Dashboard:** Web-based (HTML/JS)
- **Video Processing:** OpenCV / FFmpeg

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set Gemini API key
export GEMINI_API_KEY="your-key"

# Run analysis on a video clip
python analyze.py --video clips/offside_example.mp4
```

## Project Structure

```
offside-zero/
├── README.md
├── requirements.txt
├── analyze.py           # Main analysis script
├── src/
│   ├── gemini_client.py # Gemini API wrapper
│   ├── video_processor.py # Frame extraction
│   ├── overlay.py       # Draw lines/zones
│   └── dashboard.py     # Decision output
├── clips/               # Sample match footage
└── output/              # Annotated replays
```

## Demo

[Demo video link - TBD]

## Team

- James (Blu-Chips)

## License

MIT
