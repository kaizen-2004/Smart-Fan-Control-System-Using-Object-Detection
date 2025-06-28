# Smart Fan Control System Using Object Detection

A real-time object detection system that controls a virtual fan based on the presence of chair and minifan objects.

## Features

- Real-time webcam object detection
- Image upload for detection
- Smart fan control logic
- Visual fan simulation
- Multiple camera support
- Confidence threshold adjustment

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run streamlit_app.py
```

## Deployment Options

### Option 1: Streamlit Community Cloud (Recommended)

1. **Push your code to GitHub**
   - Make sure your `best.pt` model file is in the `Deploy/` folder
   - Commit and push all files

2. **Deploy on Streamlit Cloud**
   - Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set the app file path to: `Deploy/streamlit_app.py`
   - Click "Deploy"

### Option 2: Render.com

1. **Create a new Web Service**
2. **Connect your GitHub repository**
3. **Set build command:**
   ```bash
   pip install -r Deploy/requirements.txt
   ```
4. **Set start command:**
   ```bash
   streamlit run Deploy/streamlit_app.py --server.port $PORT --server.address 0.0.0.0
   ```

### Option 3: Railway

1. **Connect your GitHub repository**
2. **Railway will auto-detect it's a Python app**
3. **Add environment variable:**
   - `PORT=8501`

## File Structure

```
Deploy/
├── streamlit_app.py      # Main Streamlit application
├── best.pt              # Your trained YOLO model
├── requirements.txt     # Python dependencies
├── .streamlit/
│   └── config.toml     # Streamlit configuration
└── README.md           # This file
```

## Model Information

- **Model**: Custom YOLO model (`best.pt`)
- **Classes**: chair, minifan
- **Input**: Images or webcam feed
- **Output**: Object detection with bounding boxes and fan control

## Troubleshooting

- **Webcam not working**: Check camera permissions and try different camera indices
- **Model not loading**: Ensure `best.pt` is in the correct location
- **Deployment issues**: Check that all files are committed to GitHub 