import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import time
import gc

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(page_title="Smart Fan Control System Using Object Detection", layout="wide")

# ============================================================================
# MODEL MANAGEMENT
# ============================================================================
@st.cache_resource
def load_model():
    """Load YOLO model once and cache it"""
    return YOLO("best.pt")

# ============================================================================
# CAMERA MANAGEMENT
# ============================================================================
class CameraManager:
    """Manages camera resources to prevent conflicts"""
    
    def __init__(self):
        self.current_camera = None
        self.current_mode = None
    
    def release_camera(self):
        """Release current camera if active"""
        if self.current_camera is not None:
            self.current_camera.release()
            self.current_camera = None
            self.current_mode = None
            gc.collect()  # Force garbage collection
    
    def get_camera(self, camera_index, mode):
        """Get camera instance, releasing any existing camera"""
        if self.current_mode != mode:
            self.release_camera()
        
        if self.current_camera is None:
            self.current_camera = cv2.VideoCapture(camera_index)
            self.current_mode = mode
            
        return self.current_camera
    
    def is_camera_available(self, camera_index):
        """Check if camera is available"""
        temp_cap = cv2.VideoCapture(camera_index)
        if temp_cap.isOpened():
            ret, frame = temp_cap.read()
            temp_cap.release()
            return ret
        return False

# Initialize camera manager
if 'camera_manager' not in st.session_state:
    st.session_state.camera_manager = CameraManager()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def detect_cameras():
    """Detect available cameras"""
    available_cameras = []
    for i in range(10):
        if st.session_state.camera_manager.is_camera_available(i):
            available_cameras.append(i)
    return available_cameras

def check_detection_status(results):
    """Check detection status and control fan logic"""
    if len(results) == 0 or len(results[0].boxes) == 0:
        return [], "No objects detected", "fan_off"
    
    detected_classes = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        class_name = model.names[cls]
        detected_classes.append(class_name)
    
    detected_classes = list(set(detected_classes))
    
    # Check for chair and minifan
    chair_detected = "chair" in detected_classes
    fan_detected = "minifan" in detected_classes
    
    if chair_detected and fan_detected:
        status = "✅ Chair detected, ✅ Fan detected - FAN TURNED ON"
        fan_state = "fan_on"
    elif chair_detected and not fan_detected:
        status = "✅ Chair detected, ❌ Fan not detected - FAN TURNED OFF"
        fan_state = "fan_off"
    elif not chair_detected and fan_detected:
        status = "❌ Chair not detected, ✅ Fan detected - FAN TURNED OFF"
        fan_state = "fan_off"
    else:
        status = "❌ Chair not detected, ❌ Fan not detected - FAN TURNED OFF"
        fan_state = "fan_off"
    
    return detected_classes, status, fan_state

def display_fan_animation(fan_state):
    """Display fan animation based on state"""
    if fan_state == "fan_on":
        st.markdown("### 🌀 Spinning Fan Simulation")
        st.markdown("""
        ```
        ╭─────────────────╮
        │    🌀 FAN ON 🌀    │
        │   ╭─────────╮   │
        │   │  ⚡⚡⚡   │   │
        │   │ ⚡🌀🌀🌀⚡  │   │
        │   │⚡🌀🌀🌀🌀🌀⚡ │   │
        │   │ ⚡🌀🌀🌀⚡  │   │
        │   │  ⚡⚡⚡   │   │
        │   ╰─────────╯   │
        │                 │
        ╰─────────────────╯
        ```
        """)
    else:
        st.markdown("### 💤 Fan Status: OFF")
        st.markdown("""
        ```
        ╭─────────────────╮
        │   💤 FAN OFF 💤   │
        │   ╭─────────╮   │
        │   │  ⚪⚪⚪   │   │
        │   │ ⚪⚪⚪⚪⚪  │   │
        │   │⚪⚪⚪⚪⚪⚪⚪ │   │
        │   │ ⚪⚪⚪⚪⚪  │   │
        │   │  ⚪⚪⚪   │   │
        │   ╰─────────╯   │
        │                 │
        ╰─────────────────╯
        ```
        """)

def display_detection_results(image, results, confidence_threshold=0.5):
    """Display detection results with side-by-side comparison"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Original Image")
        if isinstance(image, np.ndarray):
            st.image(image, channels="BGR", use_container_width=True)
        else:
            st.image(image, use_container_width=True)
    
    with col2:
        st.write("### Detection Result")
        result_img = results[0].plot()
        st.image(result_img, channels="BGR", use_container_width=True)
    
    # Check detection status and control fan
    detected_classes, status, fan_state = check_detection_status(results)
    
    # Display status
    if fan_state == "fan_on":
        st.success(status)
    else:
        st.warning(status)
    
    # Display fan animation
    display_fan_animation(fan_state)
    
    # Display detection info
    if detected_classes:
        st.write("### 📊 Detection Details:")
        for class_name in detected_classes:
            st.write(f"• {class_name}")
    
    return detected_classes, status, fan_state

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    st.title("Smart Fan Control System Using Object Detection")
    
    # Load model
    global model
    model = load_model()
    
    # Sidebar controls
    st.sidebar.title("Controls")
    st.sidebar.info("📱 **Mobile Tip**: Use Image Upload for best mobile experience!")
    
    # Main option selection
    option = st.sidebar.radio("Choose input source:", ("Image Upload", "Webcam"), index=0)
    
    if option == "Image Upload":
        image_upload_mode()
    elif option == "Webcam":
        webcam_mode()

def image_upload_mode():
    """Handle image upload mode"""
    st.write("### Image Upload Detection")
    
    st.info("📱 **Mobile Tip**: You can take a photo with your camera and upload it for detection!")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        results = model(img_array)
        display_detection_results(image, results)

def webcam_mode():
    """Handle webcam mode"""
    st.write("### Webcam Detection")
    
    # Webcam mode selection
    webcam_mode = st.radio("Choose webcam mode:", ("Capture Image", "Real-Time Detection"), index=0)
    
    # Confidence threshold
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.1)
    
    if webcam_mode == "Capture Image":
        webcam_capture_mode(confidence_threshold)
    elif webcam_mode == "Real-Time Detection":
        webcam_realtime_mode(confidence_threshold)

def webcam_capture_mode(confidence_threshold):
    """Handle webcam capture mode"""
    st.info("""
    📷 **Webcam Access Guide**
    
    To enable webcam access, follow these steps:
    
    **Step 1: Check Browser Address Bar**
    - Look for a camera icon (📷) or lock icon (🔒) in the address bar
    - Click on it to manage permissions
    
    **Step 2: Allow Camera Access**
    - Select "Allow" when prompted for camera access
    - If you see "Block", change it to "Allow"
    
    **Step 3: Refresh the Page**
    - After allowing permissions, refresh the page
    - The camera should now work
    
    **Step 4: Alternative Solution**
    - If webcam still doesn't work, use Image Upload instead
    - Take a photo with your phone/computer camera and upload it
    """)
    
    try:
        if st.button("🔧 Test Camera Access"):
            st.info("Testing camera access... Please allow camera permissions if prompted.")
        
        webcam_image = st.camera_input("📷 Click here to take a picture for detection")
        
        if webcam_image is not None:
            image = Image.open(webcam_image)
            img_array = np.array(image)
            results = model(img_array, conf=confidence_threshold)
            display_detection_results(image, results, confidence_threshold)
        
        else:
            st.write("📷 **Click the camera button above to take a picture for detection**")
            st.warning("""
            ⚠️ **Camera Not Working?**
            
            If you can't see the camera button or it's not working:
            
            1. **Check browser permissions** (see guide above)
            2. **Try a different browser** (Chrome, Firefox, Safari)
            3. **Use Image Upload** instead (works on all devices)
            4. **Check if camera is being used by another app**
            """)
            
    except Exception as e:
        st.error(f"Webcam error: {str(e)}")
        st.info("""
        🔧 **Troubleshooting Tips:**
        
        - **Close other apps** that might be using the camera
        - **Try a different browser**
        - **Use Image Upload** as an alternative
        - **Check your camera drivers** are working
        """)

def webcam_realtime_mode(confidence_threshold):
    """Handle webcam real-time detection mode"""
    st.info("""
    🔄 **Real-Time Detection Mode**
    
    This mode provides continuous detection from your webcam.
    Note: This may not work on all mobile devices due to browser limitations.
    """)
    
    # Camera detection and selection
    st.sidebar.markdown("### 📹 Camera Detection")
    
    if st.sidebar.button("🔍 Detect Available Cameras"):
        with st.spinner("Detecting cameras..."):
            available_cameras = detect_cameras()
            if available_cameras:
                st.sidebar.success(f"Found cameras: {available_cameras}")
                st.session_state.available_cameras = available_cameras
            else:
                st.sidebar.error("No cameras detected!")
                st.sidebar.info("""
                **Troubleshooting:**
                1. Check if camera is connected
                2. Close other apps using camera
                3. Try Image Upload instead
                """)
    
    # Camera source selection
    if 'available_cameras' in st.session_state and st.session_state.available_cameras:
        camera_index = st.sidebar.selectbox(
            "Select Camera:", 
            st.session_state.available_cameras,
            help="Choose from detected cameras"
        )
    else:
        camera_index = st.sidebar.number_input(
            "Camera Index", 
            min_value=0, 
            max_value=10, 
            value=0, 
            step=1, 
            help="0=default camera, 1=second camera, etc."
        )
    
    # Real-time detection controls
    col1, col2, col3 = st.columns(3)
    with col1:
        start_button = st.button("▶️ Start Real-Time Detection", key="start_realtime")
    with col2:
        stop_button = st.button("⏹️ Stop Detection", key="stop_realtime")
    with col3:
        if 'running_realtime' in st.session_state and st.session_state.running_realtime:
            st.write("**Status:** Running")
        else:
            st.write("**Status:** Ready")
    
    # Create placeholders for video streams
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Original Feed")
        original_placeholder = st.empty()
    
    with col2:
        st.write("### Detection Results")
        detection_placeholder = st.empty()
    
    # Status and fan animation area
    status_placeholder = st.empty()
    fan_animation_placeholder = st.empty()
    
    # Initialize real-time detection state
    if 'running_realtime' not in st.session_state:
        st.session_state.running_realtime = False
    
    if start_button:
        st.session_state.running_realtime = True
        
        try:
            # Get camera instance
            cap = st.session_state.camera_manager.get_camera(camera_index, "realtime")
            
            if not cap.isOpened():
                st.error(f"Could not open camera {camera_index}. Please check if your camera is connected and not being used by another application.")
                st.info("""
                **Troubleshooting:**
                1. **Click 'Detect Available Cameras'** to find working cameras
                2. **Close other apps** that might be using the camera
                3. **Try a different camera index**
                4. **Use Image Upload** as an alternative
                """)
                st.session_state.running_realtime = False
                return
            
            # Main real-time detection loop
            while st.session_state.running_realtime:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to grab frame from webcam")
                    break
                
                # Show original frame
                original_placeholder.image(frame, channels="BGR", use_container_width=True)
                
                # Run detection
                results = model(frame, conf=confidence_threshold)
                
                # Get detection result
                if len(results) > 0:
                    result_img = results[0].plot()
                    detection_placeholder.image(result_img, channels="BGR", use_container_width=True)
                    
                    # Check detection status and control fan
                    detected_classes, status, fan_state = check_detection_status(results)
                    
                    # Display status
                    if fan_state == "fan_on":
                        status_placeholder.success(status)
                    else:
                        status_placeholder.warning(status)
                    
                    # Display fan animation
                    if fan_state == "fan_on":
                        fan_animation_placeholder.markdown("### 🌀 Spinning Fan Simulation")
                        fan_animation_placeholder.markdown("""
                        ```
                        ╭─────────────────╮
                        │    🌀 FAN ON 🌀    │
                        │   ╭─────────╮   │
                        │   │  ⚡⚡⚡   │   │
                        │   │ ⚡🌀🌀🌀⚡  │   │
                        │   │⚡🌀🌀🌀🌀🌀⚡ │   │
                        │   │ ⚡🌀🌀🌀⚡  │   │
                        │   │  ⚡⚡⚡   │   │
                        │   ╰─────────╯   │
                        │                 │
                        ╰─────────────────╯
                        ```
                        """)
                    else:
                        fan_animation_placeholder.markdown("### 💤 Fan Status: OFF")
                        fan_animation_placeholder.markdown("""
                        ```
                        ╭─────────────────╮
                        │   💤 FAN OFF 💤   │
                        │   ╭─────────╮   │
                        │   │  ⚪⚪⚪   │   │
                        │   │ ⚪⚪⚪⚪⚪  │   │
                        │   │⚪⚪⚪⚪⚪⚪⚪ │   │
                        │   │ ⚪⚪⚪⚪⚪  │   │
                        │   │  ⚪⚪⚪   │   │
                        │   ╰─────────╯   │
                        │                 │
                        ╰─────────────────╯
                        ```
                        """)
                    
                    # Display detection info in sidebar
                    if detected_classes:
                        st.sidebar.markdown("### 📊 Detection Summary:")
                        for class_name in detected_classes:
                            st.sidebar.write(f"• {class_name}")
                
                time.sleep(0.1)  # Small delay to prevent overwhelming the UI
                
        except Exception as e:
            st.error(f"Error in real-time detection: {str(e)}")
            st.session_state.running_realtime = False
    
    if stop_button:
        st.session_state.running_realtime = False
        st.session_state.camera_manager.release_camera()

# ============================================================================
# SIDEBAR INFORMATION
# ============================================================================
def display_sidebar_info():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Instructions:")
    st.sidebar.markdown("1. **Image Upload**: Upload an image to detect objects (Recommended)")
    st.sidebar.markdown("2. **Webcam - Capture**: Take a single photo for detection")
    st.sidebar.markdown("3. **Webcam - Real-Time**: Continuous detection from webcam")
    st.sidebar.markdown("4. **Fan Control**: Fan turns ON when both chair and minifan are detected")
    st.sidebar.markdown("5. **Mobile**: Use Image Upload for best mobile experience")
    st.sidebar.markdown("6. **Camera Issues**: Use 'Detect Available Cameras' button")

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
    display_sidebar_info() 