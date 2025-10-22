from flask import Flask, request, jsonify, render_template_string, send_from_directory
import os
import json
import numpy as np
from PIL import Image, ImageFile # Import ImageFile
import base64
import io
import cv2
from datetime import datetime
import uuid
import time

# Set Pillow's maximum image pixel limit to a higher value
# Default is 178,956,970 pixels (approx 178.9 MP)
# Be cautious when setting this too high, as it consumes RAM.
# 671,452,800 pixels is ~0.67 GP. Let's set it to something like 1 Gigapixel (1,000,000,000 pixels)
Image.MAX_IMAGE_PIXELS = 1_000_000_000 # 1 Gigapixel, or adjust as needed.
# You can also use None to disable the limit entirely, but that's not recommended for production.

# Only import SAM2 components (no need for build_sam imports)
try:
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    SAM2_AVAILABLE = True
except ImportError:
    print("SAM2 not installed. Install with: pip install git+https://github.com/facebookresearch/sam2.git")
    SAM2_AVAILABLE = False

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Set to 100MB

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('segmentations', exist_ok=True)

# SAM2 model configuration - Using Hugging Face models
# Available models: "facebook/sam2-hiera-tiny", "facebook/sam2-hiera-small",
#                  "facebook/sam2-hiera-base-plus", "facebook/sam2-hiera-large"
sam2_model_id = "facebook/sam2-hiera-large"  # You can change this to smaller models if needed

# Global variables for SAM2
predictor = None
current_image = None
current_session = None


def initialize_sam2():
    """Initialize SAM2 model using Hugging Face"""
    global predictor
    if not SAM2_AVAILABLE:
        return False

    try:
        print(f"Loading SAM2 model: {sam2_model_id}")
        # Initialize SAM2 image predictor with Hugging Face model
        predictor = SAM2ImagePredictor.from_pretrained(sam2_model_id)
        print("SAM2 model loaded successfully!")
        return True
    except Exception as e:
        print(f"Failed to initialize SAM2: {e}")
        return False


class SegmentationSession:
    def __init__(self, image_path):
        self.image_path = image_path
        self.points = []
        self.labels = []
        self.masks = []
        self.segmentations = []
        self.session_id = str(uuid.uuid4())

    def add_point(self, x, y, is_positive):
        """Add a point and label for segmentation"""
        self.points.append([x, y])
        self.labels.append(1 if is_positive else 0)

    def clear_points(self):
        """Clear all points and labels"""
        self.points = []
        self.labels = []

    def get_points_array(self):
        """Get points as numpy array"""
        if not self.points:
            return None
        return np.array(self.points)

    def get_labels_array(self):
        """Get labels as numpy array"""
        if not self.labels:
            return None
        return np.array(self.labels)


@app.route('/')
def index():
    """Main page with upload and segmentation interface"""
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>SAM2 Image Segmentation</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container { 
            background: white; 
            padding: 20px; 
            border-radius: 8px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .upload-area {
            border: 2px dashed #ccc;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            transition: border-color 0.3s;
        }
        .upload-area:hover { border-color: #007bff; }
        .upload-area.dragover { border-color: #007bff; background-color: #f8f9fa; }
        .canvas-container { 
            position: relative; 
            display: inline-block; 
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
            overflow: hidden;
            max-width: 100%;
            max-height: 80vh;
        }
        canvas { 
            cursor: crosshair;
            display: block;
            transition: transform 0.1s ease;
        }
        .controls {
            margin: 20px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }
        .btn {
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            margin: 5px;
            font-size: 14px;
        }
        .btn:hover { background: #0056b3; }
        .btn-secondary { background: #6c757d; }
        .btn-secondary:hover { background: #545b62; }
        .btn-success { background: #28a745; }
        .btn-success:hover { background: #1e7e34; }
        .btn-danger { background: #dc3545; }
        .btn-danger:hover { background: #c82333; }
        .status {
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .status.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .status.info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        .points-list {
            max-height: 200px;
            overflow-y: auto;
            border: 1px solid #ddd;
            padding: 10px;
            border-radius: 4px;
            background: white;
        }
        .point-item {
            padding: 5px;
            margin: 2px 0;
            border-radius: 3px;
        }
        .point-positive { background: #d4edda; }
        .point-negative { background: #f8d7da; }
        .instructions {
            background: #e9ecef;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        .segmentation-list {
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 4px;
        }
        .segmentation-item {
            background: white;
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>SAM2 Interactive Image Segmentation</h1>

        <div class="instructions">
            <h3>Instructions:</h3>
            <ul>
                <li><strong>Left click:</strong> Add positive point (inside structure)</li>
                <li><strong>Right click:</strong> Add negative point (outside structure)</li>
                <li><strong>Mouse wheel:</strong> Zoom in/out</li>
                <li><strong>Middle mouse:</strong> Pan image when zoomed</li>
                <li>Segmentation runs automatically after each click</li>
                <li>Mask overlay is shown by default</li>
            </ul>
        </div>

        <div class="upload-area" id="uploadArea">
            <p>Drag and drop an image here or click to select</p>
            <input type="file" id="imageInput" accept="image/*" style="display: none;">
        </div>

        <div id="status"></div>

        <div id="imageContainer" style="display: none;">
            <div class="canvas-container">
                <canvas id="imageCanvas"></canvas>
            </div>

            <div class="controls">
                <h3>Current Points:</h3>
                <div id="pointsList" class="points-list">
                    No points added yet
                </div>

                <div style="margin-top: 15px;">
                    <button class="btn btn-secondary" onclick="clearPoints()">Clear Points</button>
                    <button class="btn btn-success" onclick="saveSegmentation()" id="saveBtn" disabled>Save Segmentation</button>
                    <button class="btn btn-danger" onclick="resetSession()">Reset Session</button>
                </div>

                <div>
                    <label>
                        <input type="checkbox" id="showMask" checked> Show mask overlay
                    </label>
                    <span style="margin-left: 20px;">Zoom: <span id="zoomLevel">100%</span></span>
                    <button class="btn btn-secondary" onclick="resetZoom()" style="margin-left: 10px;">Reset Zoom</button>
                </div>
            </div>

            <div class="segmentation-list">
                <h3>Saved Segmentations:</h3>
                <div id="segmentationsList">
                    No segmentations saved yet
                </div>
                <button class="btn btn-success" onclick="downloadSegmentations()" id="downloadBtn" disabled>
                    Download All Segmentations (JSON)
                </button>
            </div>
        </div>
    </div>

    <script>
        let canvas, ctx;
        let image = null;
        let currentMask = null;
        let points = [];
        let sessionId = null;
        let savedSegmentations = [];
        let zoomLevel = 1;
        let panX = 0, panY = 0;
        let isDragging = false;
        let lastMouseX = 0, lastMouseY = 0;

        document.addEventListener('DOMContentLoaded', function() {
            const uploadArea = document.getElementById('uploadArea');
            const imageInput = document.getElementById('imageInput');

            uploadArea.addEventListener('click', () => imageInput.click());
            uploadArea.addEventListener('dragover', handleDragOver);
            uploadArea.addEventListener('drop', handleDrop);
            imageInput.addEventListener('change', handleFileSelect);

            canvas = document.getElementById('imageCanvas');
            ctx = canvas.getContext('2d');

            canvas.addEventListener('click', handleCanvasClick);
            canvas.addEventListener('contextmenu', handleCanvasRightClick);
            canvas.addEventListener('wheel', handleWheel);
            canvas.addEventListener('mousedown', handleMouseDown);
            canvas.addEventListener('mousemove', handleMouseMove);
            canvas.addEventListener('mouseup', handleMouseUp);
            canvas.addEventListener('mouseleave', handleMouseUp);

            document.getElementById('showMask').addEventListener('change', toggleMaskOverlay);
        });

        function handleDragOver(e) {
            e.preventDefault();
            e.currentTarget.classList.add('dragover');
        }

        function handleDrop(e) {
            e.preventDefault();
            e.currentTarget.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                uploadImage(files[0]);
            }
        }

        function handleFileSelect(e) {
            if (e.target.files.length > 0) {
                uploadImage(e.target.files[0]);
            }
        }

        function uploadImage(file) {
            // Add client-side file size check for better user experience
            const MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024; // 100MB
            if (file.size > MAX_FILE_SIZE_BYTES) {
                showStatus('Error: Image size exceeds 100MB limit.', 'error');
                return; 
            }

            const formData = new FormData();
            formData.append('image', file);

            showStatus('Uploading image...', 'info');

            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                // Check for non-OK HTTP status codes (e.g., 4xx, 5xx)
                if (!response.ok) {
                    // Try to get more detailed error from server if it's not a Flask MAX_CONTENT_LENGTH error
                    return response.text().then(text => {
                        let errorMsg = `Server error: ${response.status} ${response.statusText}`;
                        try {
                            const jsonError = JSON.parse(text);
                            if (jsonError.error) {
                                errorMsg = jsonError.error;
                            }
                        } catch (e) {
                            // If not JSON, use the raw text
                            errorMsg = text;
                        }
                        throw new Error(errorMsg);
                    });
                }
                // If response is OK, parse as JSON
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    sessionId = data.session_id;
                    loadImage(data.image_url);
                    showStatus('Image uploaded successfully!', 'success');
                    document.getElementById('imageContainer').style.display = 'block';
                } else {
                    showStatus('Error uploading image: ' + data.error, 'error');
                }
            })
            .catch(error => {
                console.error('Fetch error during image upload:', error); // Log the full error
                showStatus('Error uploading image: ' + error.message, 'error');
            });
        }

        function loadImage(imageUrl) {
            image = new Image();
            image.onload = function() {
                console.log('Image loaded into JS Image object. Dimensions:', this.width, this.height, 'Src:', this.src);
                // Ensure dimensions are valid before setting canvas size
                if (this.width > 0 && this.height > 0) {
                    canvas.width = this.width;
                    canvas.height = this.height;
                    resetZoom();
                    redrawCanvas();
                } else {
                    console.error('Image loaded but has invalid dimensions (0 width or height). Image URL:', this.src);
                    showStatus('Error: Loaded image has invalid dimensions.', 'error');
                }
            };
            image.onerror = function() {
                console.error('Failed to load image from URL:', this.src);
                showStatus('Failed to load image into browser: ' + this.src, 'error');
            };
            image.src = imageUrl;
            console.log('Attempting to load image from URL:', imageUrl);
        }

        function handleCanvasClick(e) {
            if (!image || isDragging) return;

            const coords = getCanvasCoordinates(e);
            addPoint(coords.x, coords.y, true); // Left click = positive point
        }

        function handleCanvasRightClick(e) {
            e.preventDefault();
            if (!image || isDragging) return;

            const coords = getCanvasCoordinates(e);
            addPoint(coords.x, coords.y, false); // Right click = negative point
        }

        function getCanvasCoordinates(e) {
            const rect = canvas.getBoundingClientRect();
            const clientX = e.clientX - rect.left;
            const clientY = e.clientY - rect.top;

            // Convert screen coordinates to canvas coordinates accounting for zoom and pan
            const x = Math.round((clientX / zoomLevel) - panX);
            const y = Math.round((clientY / zoomLevel) - panY);

            return { x, y };
        }

        function handleWheel(e) {
            e.preventDefault();

            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            const newZoom = Math.max(0.1, Math.min(5, zoomLevel * delta));

            // Adjust pan to zoom towards mouse position
            const zoomRatio = newZoom / zoomLevel;
            panX = mouseX / newZoom - (mouseX / zoomLevel - panX) * zoomRatio;
            panY = mouseY / newZoom - (mouseY / zoomLevel - panY) * zoomRatio;

            zoomLevel = newZoom;
            updateZoomDisplay();
            redrawCanvas();
        }

        function handleMouseDown(e) {
            if (e.button === 1) { // Middle mouse button for panning
                e.preventDefault();
                isDragging = true;
                lastMouseX = e.clientX;
                lastMouseY = e.clientY;
                canvas.style.cursor = 'grabbing';
            }
        }

        function handleMouseMove(e) {
            if (isDragging) {
                const deltaX = e.clientX - lastMouseX;
                const deltaY = e.clientY - lastMouseY;

                panX += deltaX / zoomLevel;
                panY += deltaY / zoomLevel;

                lastMouseX = e.clientX;
                lastMouseY = e.clientY;

                redrawCanvas();
            }
        }

        function handleMouseUp(e) {
            isDragging = false;
            canvas.style.cursor = 'crosshair';
        }

        function resetZoom() {
            zoomLevel = 1;
            panX = 0;
            panY = 0;
            updateZoomDisplay();
            redrawCanvas();
        }

        function updateZoomDisplay() {
            document.getElementById('zoomLevel').textContent = Math.round(zoomLevel * 100) + '%';
        }

        function addPoint(x, y, isPositive) {
            points.push({x, y, isPositive});

            fetch('/add_point', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    session_id: sessionId,
                    x: x,
                    y: y,
                    is_positive: isPositive
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    updatePointsList();
                    redrawCanvas();
                    // Automatically run segmentation after adding point
                    runSegmentation();
                } else {
                    showStatus('Error adding point: ' + data.error, 'error');
                }
            })
            .catch(error => {
                console.error('Fetch error during add_point:', error);
                showStatus('Error adding point: ' + error.message, 'error');
            });
        }

        function runSegmentation() {
            if (points.length === 0) {
                return; // Don't show error for automatic segmentation
            }

            fetch('/segment', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({session_id: sessionId})
            })
            .then(response => {
                if (!response.ok) {
                    return response.text().then(text => {
                        let errorMsg = `Server error during segmentation: ${response.status} ${response.statusText}`;
                        try {
                            const jsonError = JSON.parse(text);
                            if (jsonError.error) {
                                errorMsg = jsonError.error;
                            }
                        } catch (e) {
                            errorMsg = text;
                        }
                        throw new Error(errorMsg);
                    });
                }
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    currentMask = data.mask;
                    document.getElementById('saveBtn').disabled = false;
                    redrawCanvas();
                } else {
                    showStatus('Error running segmentation: ' + data.error, 'error');
                }
            })
            .catch(error => {
                console.error('Fetch error during segmentation:', error);
                showStatus('Error running segmentation: ' + error.message, 'error');
            });
        }

        function saveSegmentation() {
            if (!currentMask) {
                showStatus('No segmentation to save', 'error');
                return;
            }

            fetch('/save_segmentation', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({session_id: sessionId})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    savedSegmentations.push({
                        id: data.segmentation_id,
                        points: [...points],
                        timestamp: new Date().toLocaleString()
                    });
                    updateSegmentationsList();
                    clearPoints();
                    currentMask = null;
                    document.getElementById('saveBtn').disabled = true;
                    document.getElementById('downloadBtn').disabled = false;
                    redrawCanvas();
                    showStatus('Segmentation saved!', 'success');
                } else {
                    showStatus('Error saving segmentation: ' + data.error, 'error');
                }
            })
            .catch(error => {
                console.error('Fetch error during save_segmentation:', error);
                showStatus('Error saving segmentation: ' + error.message, 'error');
            });
        }

        function clearPoints() {
            points = [];
            currentMask = null;
            document.getElementById('saveBtn').disabled = true;

            fetch('/clear_points', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({session_id: sessionId})
            })
            .then(response => response.json())
            .then(data => {
                updatePointsList();
                redrawCanvas();
                if (data.success) {
                    showStatus('Points cleared', 'info');
                } else {
                    showStatus('Error clearing points: ' + data.error, 'error');
                }
            })
            .catch(error => {
                console.error('Fetch error during clear_points:', error);
                showStatus('Error clearing points: ' + error.message, 'error');
            });
        }

        function resetSession() {
            points = [];
            currentMask = null;
            savedSegmentations = [];
            document.getElementById('saveBtn').disabled = true;
            document.getElementById('downloadBtn').disabled = true;
            resetZoom();

            fetch('/reset_session', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({session_id: sessionId})
            })
            .then(response => response.json())
            .then(data => {
                updatePointsList();
                updateSegmentationsList();
                redrawCanvas();
                if (data.success) {
                    showStatus('Session reset', 'info');
                } else {
                    showStatus('Error resetting session: ' + data.error, 'error');
                }
            })
            .catch(error => {
                console.error('Fetch error during reset_session:', error);
                showStatus('Error resetting session: ' + error.message, 'error');
            });
        }

        function downloadSegmentations() {
            if (savedSegmentations.length === 0) {
                showStatus('No segmentations to download', 'error');
                return;
            }

            fetch('/download_segmentations', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({session_id: sessionId})
            })
            .then(response => {
                if (!response.ok) {
                    return response.text().then(text => {
                        let errorMsg = `Server error during download: ${response.status} ${response.statusText}`;
                        try {
                            const jsonError = JSON.parse(text);
                            if (jsonError.error) {
                                errorMsg = jsonError.error;
                            }
                        } catch (e) {
                            errorMsg = text;
                        }
                        throw new Error(errorMsg);
                    });
                }
                return response.blob();
            })
            .then(blob => {
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = `segmentations_${sessionId}.json`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                showStatus('Segmentations downloaded!', 'success');
            })
            .catch(error => {
                console.error('Fetch error during download_segmentations:', error);
                showStatus('Error downloading segmentations: ' + error.message, 'error');
            });
        }

        function redrawCanvas() {
            if (!image) {
                console.warn('redrawCanvas called but image is null. Cannot draw.');
                return;
            }
            if (image.width === 0 || image.height === 0) {
                console.warn('Image has 0 dimensions. Cannot draw. Current image:', image);
                return;
            }

            ctx.save();
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Apply zoom and pan transformation
            ctx.scale(zoomLevel, zoomLevel);
            ctx.translate(panX, panY);

            // Draw image
            console.log('Drawing image on canvas. Image dimensions:', image.width, image.height);
            ctx.drawImage(image, 0, 0);

            // Draw mask overlay if enabled and available
            if (currentMask && document.getElementById('showMask').checked) {
                drawMaskOverlay();
            }

            // Draw points
            points.forEach(point => {
                ctx.beginPath();
                ctx.arc(point.x, point.y, 3, 0, 2 * Math.PI); // Smaller radius (3 instead of 8)
                ctx.fillStyle = point.isPositive ? '#00ff00' : '#ff0000';
                ctx.fill();
                ctx.strokeStyle = '#000000';
                ctx.lineWidth = 1; // Thinner stroke
                ctx.stroke();
            });

            ctx.restore();
        }

        function drawMaskOverlay() {
            if (!currentMask) return;

            const mask = JSON.parse(currentMask);
            // Verify mask dimensions match canvas
            if (mask.length !== canvas.height || (mask[0] && mask[0].length !== canvas.width)) {
                console.error('Mask dimensions do not match canvas dimensions!', 'Mask H:', mask.length, 'Mask W:', mask[0] ? mask[0].length : 'N/A', 'Canvas H:', canvas.height, 'Canvas W:', canvas.width);
                showStatus('Error: Mask dimensions mismatch.', 'error');
                return;
            }

            const maskCanvas = document.createElement('canvas');
            maskCanvas.width = canvas.width;
            maskCanvas.height = canvas.height;
            const maskCtx = maskCanvas.getContext('2d');

            const imageData = maskCtx.createImageData(canvas.width, canvas.height);

            for (let y = 0; y < canvas.height; y++) {
                for (let x = 0; x < canvas.width; x++) {
                    const idx = (y * canvas.width + x) * 4;
                    // Ensure mask[y] and mask[y][x] exist to prevent errors on malformed masks
                    if (mask[y] && typeof mask[y][x] !== 'undefined' && mask[y][x]) {
                        imageData.data[idx] = 0;     // R
                        imageData.data[idx + 1] = 255; // G
                        imageData.data[idx + 2] = 0;   // B
                        imageData.data[idx + 3] = 120; // A (semi-transparent)
                    } else {
                        imageData.data[idx + 3] = 0; // Transparent
                    }
                }
            }

            maskCtx.putImageData(imageData, 0, 0);
            ctx.drawImage(maskCanvas, 0, 0);
        }

        function toggleMaskOverlay() {
            redrawCanvas();
        }

        function updatePointsList() {
            const pointsList = document.getElementById('pointsList');
            if (points.length === 0) {
                pointsList.innerHTML = 'No points added yet';
                return;
            }

            pointsList.innerHTML = points.map((point, index) => `
                <div class="point-item ${point.isPositive ? 'point-positive' : 'point-negative'}">
                    Point ${index + 1}: (${point.x}, ${point.y}) - ${point.isPositive ? 'Positive' : 'Negative'}
                </div>
            `).join('');
        }

        function updateSegmentationsList() {
            const segList = document.getElementById('segmentationsList');
            if (savedSegmentations.length === 0) {
                segList.innerHTML = 'No segmentations saved yet';
                return;
            }

            segList.innerHTML = savedSegmentations.map((seg, index) => `
                <div class="segmentation-item">
                    <strong>Segmentation ${index + 1}</strong><br>
                    Points: ${seg.points.length}<br>
                    Saved: ${seg.timestamp}
                </div>
            `).join('');
        }

        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.innerHTML = `<div class="status ${type}">${message}</div>`;
            // Clear status after 5 seconds, but only if it's not an error
            if (type !== 'error') {
                setTimeout(() => {
                    status.innerHTML = '';
                }, 5000);
            }
        }
    </script>
</body>
</html>
    ''')


@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload"""
    global current_session, current_image

    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    try:
        # Validate image file type and header if necessary (optional but good practice)
        # For simplicity, we'll rely on browser 'accept="image/*"' for now

        # Save uploaded image
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Ensure file is completely written before verification
        time.sleep(0.1)  # Small delay to ensure file is fully written

        # IMPORTANT: Verify the saved file can be read as an image
        # This is a crucial step for debugging "no errors but no display"
        try:
            # First check if file exists and has content
            if not os.path.exists(filepath):
                print(f"Server: Error - saved file '{filepath}' does not exist")
                return jsonify({'success': False, 'error': 'File was not saved properly'}), 500
            
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                os.remove(filepath) # Clean up empty file
                print(f"Server: Error - saved file '{filename}' is empty")
                return jsonify({'success': False, 'error': 'Uploaded file is empty'}), 422
            
            # Use PIL for robust image check as cv2.imread can sometimes fail silently
            img = Image.open(filepath)
            if img is None:
                os.remove(filepath) # Clean up invalid file
                print(f"Server: Error - PIL returned None for file '{filename}'")
                return jsonify({'success': False, 'error': 'Failed to open image file'}), 422
            
            try:
                img.verify() # Verify that it is an image
                # Reopen the image since verify() closes it
                img = Image.open(filepath)
                # Convert to RGB if it's not (e.g., CMYK JPEGs can cause issues)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Get dimensions
                width, height = img.size
                print(f"Server: Saved image '{filename}' verified. Dimensions: {width}x{height}, Size: {file_size} bytes")
                img.close() # Close the image properly
            except Exception as verify_err:
                if img:
                    img.close()
                os.remove(filepath) # Clean up invalid file
                print(f"Server: Error verifying image '{filename}': {verify_err}")
                return jsonify({'success': False, 'error': f'Uploaded file is not a valid image: {verify_err}'}), 422
                
        except Exception as img_err:
            # Clean up file if it exists
            if os.path.exists(filepath):
                os.remove(filepath)
            print(f"Server: Error processing uploaded image '{filename}': {img_err}")
            return jsonify({'success': False, 'error': f'Error processing uploaded file: {img_err}'}), 422

        # Initialize new session
        current_session = SegmentationSession(filepath)

        # Load image for SAM2 (if SAM2 is available and initialized)
        if predictor and SAM2_AVAILABLE:
            image_cv2 = cv2.imread(filepath)
            if image_cv2 is None:
                print(f"Server: OpenCV failed to read image at '{filepath}'.")
                return jsonify({'success': False, 'error': 'Server failed to read image with OpenCV. It might be corrupted or an unsupported format.'}), 500
            image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
            predictor.set_image(image_rgb)
            current_image = image_rgb
            print(f"Server: Image set for SAM2 predictor.")
        else:
            print("Server: SAM2 predictor not available or initialized. Skipping set_image.")


        return jsonify({
            'success': True,
            'session_id': current_session.session_id,
            'image_url': f'/uploads/{filename}'
        })

    except Exception as e:
        print(f"Server: Unhandled error during image upload: {e}")
        return jsonify({'success': False, 'error': f'An unexpected error occurred during upload: {str(e)}'}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded images"""
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        print(f"Server: Error serving file '{filename}': {e}")
        return jsonify({'error': 'File not found or access denied.'}), 404 # Or 500 if permissions


@app.route('/add_point', methods=['POST'])
def add_point():
    """Add a point for segmentation"""
    global current_session

    if not current_session:
        return jsonify({'success': False, 'error': 'No active session'}), 400

    data = request.json
    x = data.get('x')
    y = data.get('y')
    is_positive = data.get('is_positive', True)

    if x is None or y is None:
        return jsonify({'success': False, 'error': 'Coordinates (x, y) are required.'}), 400

    current_session.add_point(x, y, is_positive)

    return jsonify({'success': True})


@app.route('/segment', methods=['POST'])
def run_segmentation():
    """Run SAM2 segmentation"""
    global current_session, predictor, current_image

    if not current_session:
        return jsonify({'success': False, 'error': 'No active session'}), 400

    if not SAM2_AVAILABLE or not predictor:
        return jsonify({'success': False, 'error': 'SAM2 not available or not initialized on the server.'}), 503 # Service Unavailable

    try:
        points = current_session.get_points_array()
        labels = current_session.get_labels_array()

        if points is None or len(points) == 0:
            return jsonify({'success': False, 'error': 'No points added to perform segmentation.'}), 400
        
        if labels is None or len(labels) == 0:
            return jsonify({'success': False, 'error': 'No labels available for segmentation.'}), 400
        
        # Ensure points and labels are numpy arrays with correct shapes
        points = np.array(points)
        labels = np.array(labels)
        
        if points.ndim != 2 or points.shape[1] != 2:
            return jsonify({'success': False, 'error': 'Invalid points format. Expected 2D array with 2 columns (x, y).'}), 400
        
        if labels.ndim != 1 or len(labels) != len(points):
            return jsonify({'success': False, 'error': 'Invalid labels format. Expected 1D array with same length as points.'}), 400

        # Run SAM2 prediction
        masks, scores, logits = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True
        )

        # Validate SAM2 output
        if masks is None or len(masks) == 0:
            return jsonify({'success': False, 'error': 'SAM2 did not generate any masks for the given points.'}), 400
        
        if scores is None or len(scores) == 0:
            return jsonify({'success': False, 'error': 'SAM2 did not generate valid scores.'}), 400
        
        # Ensure we have valid numpy arrays
        masks = np.array(masks)
        scores = np.array(scores)
        
        if masks.size == 0 or scores.size == 0:
            return jsonify({'success': False, 'error': 'SAM2 generated empty arrays.'}), 400

        # Use the mask with highest score
        best_mask = masks[np.argmax(scores)]

        # Convert mask to JSON serializable format
        # IMPORTANT: Ensure mask dimensions match the original image dimensions for the frontend
        # This implicitly relies on SAM2 outputting masks with same dims as input image
        mask_json = json.dumps(best_mask.astype(int).tolist())

        # Store mask in session
        current_session.masks = [best_mask]

        return jsonify({
            'success': True,
            'mask': mask_json,
            'score': float(scores[np.argmax(scores)])
        })

    except Exception as e:
        print(f"Server: Error during SAM2 segmentation: {e}")
        # Consider logging full traceback for debugging server-side SAM2 issues
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'An error occurred during segmentation: {str(e)}'}), 500


@app.route('/save_segmentation', methods=['POST'])
def save_segmentation():
    """Save current segmentation"""
    global current_session

    if not current_session:
        return jsonify({'success': False, 'error': 'No active session'}), 400

    if not current_session.masks:
        return jsonify({'success': False, 'error': 'No segmentation to save. Run segmentation first.'}), 400

    try:
        # Create segmentation data
        segmentation_data = {
            'id': len(current_session.segmentations),
            'timestamp': datetime.now().isoformat(),
            'points': current_session.points.copy(),
            'labels': current_session.labels.copy(),
            'mask': current_session.masks[0].astype(int).tolist()
        }

        current_session.segmentations.append(segmentation_data)

        # Clear current points and mask for next segmentation
        current_session.clear_points()
        current_session.masks = []

        return jsonify({
            'success': True,
            'segmentation_id': segmentation_data['id']
        })

    except Exception as e:
        print(f"Server: Error saving segmentation: {e}")
        return jsonify({'success': False, 'error': f'An error occurred saving segmentation: {str(e)}'}), 500


@app.route('/clear_points', methods=['POST'])
def clear_points():
    """Clear current points"""
    global current_session

    if current_session:
        current_session.clear_points()
        current_session.masks = []

    return jsonify({'success': True})


@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Reset current session"""
    global current_session

    if current_session:
        current_session.clear_points()
        current_session.masks = []
        current_session.segmentations = []

    return jsonify({'success': True})


@app.route('/download_segmentations', methods=['POST'])
def download_segmentations():
    """Download all segmentations as JSON"""
    global current_session

    if not current_session or not current_session.segmentations:
        return jsonify({'success': False, 'error': 'No segmentations to download'}), 404

    try:
        # Prepare segmentation data
        export_data = {
            'session_id': current_session.session_id,
            'image_path': current_session.image_path,
            'export_timestamp': datetime.now().isoformat(),
            'segmentations': current_session.segmentations
        }

        # Create JSON file
        filename = f"segmentations_{current_session.session_id}.json"
        filepath = os.path.join('segmentations', filename)

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        # Return file for download
        return send_from_directory('segmentations', filename, as_attachment=True)

    except Exception as e:
        print(f"Server: Error during segmentation download: {e}")
        return jsonify({'success': False, 'error': f'An error occurred during download: {str(e)}'}), 500


# Initialize SAM2 when the module is imported (for gunicorn compatibility)
print("Initializing SAM2...")
if initialize_sam2():
    print("SAM2 initialized successfully!")
else:
    print("Warning: SAM2 not available. Some features may not work.")
    print("To use SAM2, please install with: pip install git+https://github.com/facebookresearch/sam2.git")

if __name__ == '__main__':
    print("Starting Flask server...")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)