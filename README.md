# PathAI - SAM2 Image Segmentation Tool

A web-based interactive image segmentation tool powered by Meta's SAM2 (Segment Anything Model 2). This application allows users to upload images and perform precise segmentation by clicking on objects of interest.

## Features

- **Interactive Segmentation**: Click on objects to segment them using SAM2
- **Real-time Processing**: Instant segmentation results as you add points
- **Multiple Point Support**: Add positive and negative points for better segmentation
- **Zoom and Pan**: Navigate large images with mouse controls
- **Save Segmentations**: Save and download segmentation results as JSON
- **Modern Web Interface**: Clean, responsive UI with drag-and-drop upload

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/recaiyilmaz/pathAI.git
cd pathAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install SAM2:
```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

4. Run the application:
```bash
python codeR.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Usage

1. **Upload Image**: Drag and drop an image or click to select one
2. **Add Points**: 
   - Left click to add positive points (inside the object you want to segment)
   - Right click to add negative points (outside the object)
3. **View Results**: The segmentation mask will appear automatically
4. **Save**: Click "Save Segmentation" to store your results
5. **Download**: Download all segmentations as a JSON file

## Controls

- **Left Click**: Add positive point
- **Right Click**: Add negative point  
- **Mouse Wheel**: Zoom in/out
- **Middle Mouse**: Pan when zoomed
- **Reset Zoom**: Return to original view

## API Endpoints

- `POST /upload` - Upload an image
- `POST /add_point` - Add a segmentation point
- `POST /segment` - Run segmentation
- `POST /save_segmentation` - Save current segmentation
- `POST /clear_points` - Clear all points
- `POST /reset_session` - Reset the entire session
- `POST /download_segmentations` - Download all segmentations

## Deployment

This application is configured to run on Render. See the deployment section below for setup instructions.

## Requirements

- Flask
- Pillow (PIL)
- OpenCV
- NumPy
- SAM2 (Segment Anything Model 2)

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
