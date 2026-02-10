import os
import glob
import uuid
import threading
import queue
import time
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
from typing import List, Dict, Any
import asyncio

# Import our core modules
from src.analysis_service import get_analysis_service

app = FastAPI(title="Offside Zero Dashboard")

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
CLIPS_DIR = os.path.join(ROOT_DIR, "clips")
TEMPLATES_DIR = os.path.join(ROOT_DIR, "templates")
STATIC_DIR = os.path.join(ROOT_DIR, "static")
LIVE_BUFFER_DIR = os.path.join(ROOT_DIR, "live_buffer")

# Ensure dirs exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(LIVE_BUFFER_DIR, exist_ok=True)

# Mounts
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# --- Async Processing Setup ---
# Simulating Cloud Pub/Sub and Firestore
TASK_QUEUE = queue.Queue()
TASKS: Dict[str, Dict[str, Any]] = {}

def worker():
    """Background worker to process analysis tasks."""
    print("Worker failed to start? No, it's running.")
    while True:
        try:
            task_id, clip_name = TASK_QUEUE.get()
            print(f"Processing task {task_id} for {clip_name}")
            
            TASKS[task_id]["status"] = "PROCESSING"
            
            video_path = os.path.join(CLIPS_DIR, clip_name)
            if not os.path.exists(video_path):
                 TASKS[task_id]["status"] = "FAILED"
                 TASKS[task_id]["error"] = "Video not found"
                 TASK_QUEUE.task_done()
                 continue

            # Run Analysis
            service = get_analysis_service(OUTPUT_DIR)
            result = service.analyze_clip(video_path)
            
            if "error" in result:
                TASKS[task_id]["status"] = "FAILED"
                TASKS[task_id]["error"] = result["error"]
            else:
                # Fixup paths for web display
                if "annotated_frames" in result:
                    result["annotated_frames"] = [f"/output/{os.path.basename(p)}" for p in result["annotated_frames"]]
                if "slowmo_video" in result:
                    result["slowmo_video"] = f"/output/{os.path.basename(result['slowmo_video'])}"
                
                TASKS[task_id]["status"] = "COMPLETED"
                TASKS[task_id]["result"] = result
            
            print(f"Task {task_id} completed.")
            TASK_QUEUE.task_done()
            
        except Exception as e:
            print(f"Worker crashed: {e}")
            # In production, we'd want to restart the worker or handle this better
            
# Start worker thread
threading.Thread(target=worker, daemon=True).start()

# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the main dashboard."""
    clips = [os.path.basename(p) for p in glob.glob(os.path.join(CLIPS_DIR, "*.mp4"))]
    return templates.TemplateResponse("index.html", {"request": request, "clips": clips})

@app.post("/analyze")
async def submit_analysis(clip_name: str = Form(...)):
    """Submit a clip for asynchronous analysis."""
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {
        "status": "PENDING",
        "clip_name": clip_name,
        "submitted_at": time.time()
    }
    TASK_QUEUE.put((task_id, clip_name))
    return {"task_id": task_id, "status": "PENDING"}

@app.post("/upload")
async def upload_clip(file: UploadFile = File(...)):
    """Upload a new video clip."""
    try:
        # Sanitize filename (basic)
        filename = os.path.basename(file.filename)
        if not filename.endswith(('.mp4', '.mov', '.avi')):
             return JSONResponse({"error": "Invalid file type"}, status_code=400)
             
        file_path = os.path.join(CLIPS_DIR, filename)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            
        return {"status": "uploaded", "filename": filename}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/ingest")
async def ingest_frame(file: UploadFile = File(...)):
    """Receive a live video frame."""
    try:
        # Save frame to buffer (simulating live processing pipeline)
        # In a real system, this would push to Pub/Sub or a streaming pipe
        timestamp = int(time.time() * 1000)
        filename = f"frame_{timestamp}.jpg"
        filepath = os.path.join(LIVE_BUFFER_DIR, filename)
        
        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # Cleanup old frames (keep buffer small - max 100 frames)
        try:
            all_frames = sorted(
                glob.glob(os.path.join(LIVE_BUFFER_DIR, "*.jpg")),
                key=os.path.getmtime
            )
            if len(all_frames) > 100:
                # Delete oldest frames
                for old_frame in all_frames[:-100]:
                    try:
                        os.remove(old_frame)
                    except OSError:
                        pass  # Ignore deletion errors
        except Exception:
            pass  # Don't fail if cleanup fails
        
        return {"status": "received", "file": filename}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

import glob
import asyncio
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

# ... imports ...

class ChatRequest(BaseModel):
    message: str
    context: Dict[str, Any]

class ROIRequest(BaseModel):
    clip_name: str
    points: List[float] # [x, y] normalized

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Ask the AI a question about the analysis."""
    service = get_analysis_service()
    response = service.analyzer.chat_with_context(request.message, request.context)
    return {"response": response}

@app.get("/live_feed")
async def live_feed():
    """Stream MJPEG feed from live buffer."""
    def frame_generator():
        while True:
            # Find latest frame in buffer
            files = sorted(glob.glob(os.path.join(LIVE_BUFFER_DIR, "*.jpg")))
            if files:
                latest_frame = files[-1]
                with open(latest_frame, "rb") as f:
                    frame_data = f.read()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            time.sleep(0.05) # 20 FPS cap for viewer

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/analyze_roi")
async def analyze_roi(request: ROIRequest):
    """Analyze specific region of interest."""
    # Add high priority task
    task_id = f"roi_{int(time.time())}"
    # In a real system, we'd pass the ROI points to the analyzer
    # For now, we'll re-analyze the whole clip but prioritize it
    TASK_QUEUE.put((task_id, request.clip_name)) 
    return {"task_id": task_id, "status": "ROI_ANALYZING"}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """Check the status of a task."""
    task = TASKS.get(task_id)
    if not task:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return task

if __name__ == "__main__":
    import webbrowser
    
    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://localhost:8000")

    threading.Thread(target=open_browser).start()
    # Use 127.0.0.1 and app object directly to avoid Windows firewall/reloader issues
    uvicorn.run(app, host="127.0.0.1", port=8000)
