import cv2
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict
import os
import time
import threading
from flask import Flask, Response, jsonify, send_file, render_template, request
from flask_cors import CORS
from io import BytesIO
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for sharing data between threads
current_detections = {
    "total_vehicles": 0,
    "cars": 0,
    "trucks": 0,
    "buses": 0,
    "motorbikes": 0,
    "bicycles": 0,
    "persons": 0,
    "frame_no": 0,
    "second": 0,
    "current_video": ""
}
frame_data = []
detection_active = False
video_thread = None
cap = None
current_video = ""

# Video configuration
VIDEO_FOLDER = "videos"
os.makedirs(VIDEO_FOLDER, exist_ok=True)

# Load YOLOv8 Medium model
try:
    logger.info("Loading YOLOv8 model...")
    model = YOLO("yolov8m.pt")
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Vehicle types to track
vehicle_types = ["car", "truck", "bus", "motorbike", "bicycle", "person"]

def get_available_videos():
    """Get list of available video files"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    available_videos = []
    
    for file in os.listdir(VIDEO_FOLDER):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            available_videos.append(file)
    
    logger.info(f"Found {len(available_videos)} videos: {available_videos}")
    return available_videos

def generate_frames():
    """Generate frames with detection boxes for video streaming"""
    global current_detections, detection_active, cap, frame_data, current_video
    
    if not current_video:
        logger.error("No video selected!")
        return
    
    video_path = os.path.join(VIDEO_FOLDER, current_video)
    
    # Check if video file exists
    if not os.path.exists(video_path):
        logger.error(f"Video file {video_path} not found!")
        # Create error frame
        while detection_active:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Video not found: {current_video}", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Place video files in 'videos' folder", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)
        return
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Error opening video file")
            return
        
        frame_no = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / fps if fps > 0 else 0.033
        
        logger.info(f"Started processing: {current_video}")
        
        while detection_active and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # Loop video when ended
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_no = 0
                continue
            
            frame_no += 1
            
            # Perform YOLO detection if model is loaded
            if model is not None:
                results = model(frame, verbose=False, conf=0.25, imgsz=640)
                
                counts = dict.fromkeys(vehicle_types, 0)
                total = 0
                
                if results and len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes:
                        cls = int(box.cls[0])
                        name = model.names[cls]
                        
                        if name in vehicle_types:
                            counts[name] += 1
                            total += 1
                            
                            # Draw bounding box
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, name, (x1, y1 - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                # Use dummy data if model isn't loaded
                counts = {"car": 2, "truck": 1, "bus": 0, "motorbike": 1, "bicycle": 0, "person": 1}
                total = 4
            
            # Update global detection data
            current_time = int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            current_detections.update({
                "total_vehicles": total,
                "cars": counts["car"],
                "trucks": counts["truck"],
                "buses": counts["bus"],
                "motorbikes": counts["motorbike"],
                "bicycles": counts["bicycle"],
                "persons": counts["person"],
                "frame_no": frame_no,
                "second": current_time,
                "current_video": current_video
            })
            
            # Save frame data (limit to last 1000 frames to prevent memory issues)
            frame_data.append({
                "video": current_video,
                "frame": frame_no,
                "second": current_time,
                "total_vehicles": total,
                "cars": counts["car"],
                "trucks": counts["truck"],
                "buses": counts["bus"],
                "motorbikes": counts["motorbike"],
                "bicycles": counts["bicycle"],
                "persons": counts["person"]
            })
            
            # Keep only last 1000 frames
            if len(frame_data) > 1000:
                frame_data.pop(0)
            
            # Add statistics overlay
            cv2.putText(frame, f"Video: {current_video}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Total: {total}", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Frame: {frame_no}", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Time: {current_time}s", (20, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Encode frame for streaming
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Control frame rate
            time.sleep(frame_delay)
    
    except Exception as e:
        logger.error(f"Error in video processing: {e}")
    finally:
        if cap:
            cap.release()

def start_detection():
    """Start the vehicle detection in a separate thread"""
    global detection_active, video_thread
    if not detection_active and current_video:
        detection_active = True
        video_thread = threading.Thread(target=generate_frames)
        video_thread.daemon = True
        video_thread.start()
        logger.info(f"Vehicle detection started for: {current_video}")
        return True
    return False

def stop_detection():
    """Stop the vehicle detection"""
    global detection_active, frame_data, cap
    detection_active = False
    
    if cap:
        cap.release()
        cap = None
    
    # Save data to CSV when detection stops
    if frame_data:
        try:
            df = pd.DataFrame(frame_data)
            csv_filename = f"vehicle_report_{current_video.split('.')[0]}_{int(time.time())}.csv"
            df.to_csv(csv_filename, index=False)
            logger.info(f"✅ Vehicle report saved to {csv_filename}")
        except Exception as e:
            logger.error(f"Error saving CSV: {e}")
    
    logger.info("Vehicle detection stopped")

# API Routes
@app.route('/')
def index():
    """Serve the frontend from Frontend folder"""
    try:
        # Get the absolute path to the frontend index.html
        base_dir = os.path.dirname(os.path.abspath(__file__))
        frontend_path = os.path.join(base_dir, '..', 'Frontend', 'index.html')
        frontend_path = os.path.normpath(frontend_path)
        
        logger.info(f"Looking for frontend at: {frontend_path}")
        
        if os.path.exists(frontend_path):
            with open(frontend_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise FileNotFoundError(f"Frontend file not found at: {frontend_path}")
            
    except Exception as e:
        logger.error(f"Error loading frontend: {e}")
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Vehicle Detection System</title>
            <style>
                body {{ 
                    background: linear-gradient(135deg, #1a2a6c, #2a3a7c);
                    color: white; 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    padding: 50px; 
                    text-align: center;
                    margin: 0;
                    min-height: 100vh;
                }}
                .container {{ 
                    max-width: 800px; 
                    margin: 0 auto; 
                    background: rgba(255,255,255,0.1);
                    padding: 40px;
                    border-radius: 15px;
                    backdrop-filter: blur(10px);
                }}
                h1 {{ 
                    background: linear-gradient(to right, #4facfe, #00f2fe);
                    -webkit-background-clip: text;
                    background-clip: text;
                    color: transparent;
                    font-size: 2.5rem;
                    margin-bottom: 20px;
                }}
                .btn {{ 
                    background: linear-gradient(to right, #4facfe, #00f2fe);
                    color: white; 
                    padding: 12px 25px; 
                    border: none; 
                    border-radius: 8px; 
                    margin: 10px; 
                    cursor: pointer; 
                    font-size: 16px;
                    font-weight: 600;
                }}
                .links {{ 
                    margin: 30px 0; 
                }}
                a {{ 
                    color: #00f2fe; 
                    text-decoration: none; 
                    margin: 0 15px; 
                    font-size: 1.1rem;
                }}
                a:hover {{ color: #4facfe; }}
                .status {{ 
                    margin-top: 20px; 
                    padding: 15px; 
                    background: rgba(255,255,255,0.1); 
                    border-radius: 8px; 
                    font-family: monospace;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚗 Vehicle Detection System</h1>
                <p>Backend is running successfully! Frontend file issue: {str(e)}</p>
                
                <div class="links">
                    <h3>Test API Endpoints:</h3>
                    <a href="/videos" target="_blank">Available Videos</a>
                    <a href="/status" target="_blank">System Status</a>
                    <a href="/detections" target="_blank">Detection Data</a>
                    <a href="/video_feed" target="_blank">Video Feed</a>
                </div>
                
                <div>
                    <button class="btn" onclick="startDetection()">Start Detection</button>
                    <button class="btn" onclick="stopDetection()">Stop Detection</button>
                </div>
                
                <div id="status" class="status">Loading status...</div>
            </div>

            <script>
                async function startDetection() {{
                    try {{
                        const response = await fetch('/start', {{ method: 'POST' }});
                        const data = await response.json();
                        document.getElementById('status').innerHTML = '<span style="color: #4caf50">' + data.message + '</span>';
                    }} catch (error) {{
                        document.getElementById('status').innerHTML = '<span style="color: #f44336">Error: ' + error + '</span>';
                    }}
                }}
                
                async function stopDetection() {{
                    try {{
                        const response = await fetch('/stop', {{ method: 'POST' }});
                        const data = await response.json();
                        document.getElementById('status').innerHTML = '<span style="color: #4caf50">' + data.message + '</span>';
                    }} catch (error) {{
                        document.getElementById('status').innerHTML = '<span style="color: #f44336">Error: ' + error + '</span>';
                    }}
                }}
                
                // Load system status on page load
                async function loadStatus() {{
                    try {{
                        const response = await fetch('/status');
                        const data = await response.json();
                        document.getElementById('status').innerHTML = 
                            'Model Loaded: ' + data.model_loaded + ' | ' +
                            'Videos Available: ' + data.available_videos.length + ' | ' +
                            'Detection Active: ' + data.detection_active;
                    }} catch (error) {{
                        document.getElementById('status').innerHTML = 'Could not load status';
                    }}
                }}
                
                loadStatus();
            </script>
        </body>
        </html>
        """

@app.route('/videos')
def get_available_videos_list():
    """Get list of available videos"""
    videos = get_available_videos()
    return jsonify({"videos": videos})

@app.route('/video_feed')
def video_feed():
    """Stream video with real-time detection"""
    if detection_active:
        return Response(generate_frames(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Detection not active. Start detection first.", 400

@app.route('/detections')
def get_detections():
    """Get current detection data"""
    return jsonify(current_detections)

@app.route('/start', methods=['POST'])
def start_detection_route():
    """Start vehicle detection with specific video"""
    try:
        data = request.get_json()
        video_filename = data.get('video_filename')
        
        if not video_filename:
            return jsonify({
                "status": "error",
                "message": "No video filename provided"
            }), 400
        
        # Set current video
        global current_video, frame_data
        current_video = video_filename
        frame_data = []  # Clear previous data
        
        if start_detection():
            return jsonify({
                "status": "success", 
                "message": f"Vehicle detection started for {current_video}",
                "detection_active": True,
                "current_video": current_video
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Detection already active or no video selected"
            }), 400
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to start detection: {str(e)}"
        }), 500

@app.route('/stop', methods=['POST'])
def stop_detection_route():
    """Stop vehicle detection"""
    try:
        stop_detection()
        return jsonify({
            "status": "success", 
            "message": "Vehicle detection stopped",
            "detection_active": False
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to stop detection: {str(e)}"
        }), 500

@app.route('/export')
def export_data():
    """Export detection data as CSV"""
    try:
        if frame_data:
            df = pd.DataFrame(frame_data)
            csv_buffer = BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            filename = f"vehicle_report_{current_video.split('.')[0] if current_video else 'all'}.csv"
            
            return Response(
                csv_buffer.getvalue(),
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment;filename={filename}'}
            )
        else:
            return jsonify({"error": "No data available for export"}), 404
    except Exception as e:
        return jsonify({"error": f"Export failed: {str(e)}"}), 500

@app.route('/stats')
def get_stats():
    """Get summary statistics"""
    try:
        if frame_data:
            df = pd.DataFrame(frame_data)
            stats = {
                "total_frames": len(frame_data),
                "max_vehicles_per_frame": int(df['total_vehicles'].max()),
                "average_vehicles_per_frame": round(df['total_vehicles'].mean(), 2),
                "total_cars": int(df['cars'].sum()),
                "total_trucks": int(df['trucks'].sum()),
                "total_buses": int(df['buses'].sum()),
                "total_motorbikes": int(df['motorbikes'].sum()),
                "total_bicycles": int(df['bicycles'].sum()),
                "total_persons": int(df['persons'].sum()),
                "current_video": current_video,
                "status": "success"
            }
            return jsonify(stats)
        else:
            return jsonify({"message": "No data collected yet", "status": "empty"})
    except Exception as e:
        return jsonify({"error": f"Stats calculation failed: {str(e)}", "status": "error"}), 500

@app.route('/frame_data')
def get_frame_data():
    """Get all collected frame data"""
    return jsonify(frame_data[-50:])  # Return last 50 frames for performance

@app.route('/status')
def get_status():
    """Get system status"""
    return jsonify({
        "detection_active": detection_active,
        "frames_processed": len(frame_data),
        "current_detections": current_detections,
        "current_video": current_video,
        "available_videos": get_available_videos(),
        "model_loaded": model is not None,
        "status": "success"
    })

@app.route('/upload', methods=['POST'])
def upload_video():
    """Upload a new video file"""
    try:
        if 'video' not in request.files:
            return jsonify({"status": "error", "message": "No video file provided"}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No file selected"}), 400
        
        # Save the file
        filename = file.filename
        file_path = os.path.join(VIDEO_FOLDER, filename)
        file.save(file_path)
        
        return jsonify({
            "status": "success",
            "message": f"Video {filename} uploaded successfully",
            "filename": filename
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Upload failed: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🚗 Starting Multi-Video Vehicle Detection System Server...")
    print("📊 Available endpoints:")
    print("  http://localhost:5000/ - Web Interface")
    print("  http://localhost:5000/videos - List available videos")
    print("  http://localhost:5000/video_feed - Live video stream")
    print("  http://localhost:5000/start - Start detection (POST)")
    print("  http://localhost:5000/stop - Stop detection (POST)")
    print("  http://localhost:5000/export - Export data as CSV")
    print("  http://localhost:5000/upload - Upload new video (POST)")
    print("\n📹 Supported formats: MP4, AVI, MOV, MKV, WMV, FLV")
    print(f"📁 Video folder: {os.path.abspath(VIDEO_FOLDER)}")
    print(f"🎬 Available videos: {get_available_videos()}")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)