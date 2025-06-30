import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import time
import gc
from functools import lru_cache
import threading
import platform
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(page_title="Group 7: Smart Fan Control System Using Object Detection", layout="wide")

# Performance settings
MAX_FRAME_WIDTH = 640
MAX_FRAME_HEIGHT = 480
TARGET_FPS = 15
FRAME_DELAY = 1.0 / TARGET_FPS

# ============================================================================
# BROWSER COMPATIBILITY
# ============================================================================
def get_browser_info():
    """Get browser information for compatibility"""
    try:
        # This is a simplified approach - in real deployment, you'd use JavaScript
        user_agent = st.get_option("server.userAgent") if hasattr(st, 'get_option') else "Unknown"
        
        # Detect browser type
        if "Chrome" in user_agent:
            return "Chrome", "Good camera support"
        elif "Firefox" in user_agent:
            return "Firefox", "Good camera support"
        elif "Safari" in user_agent:
            return "Safari", "Limited camera support"
        elif "Edge" in user_agent:
            return "Edge", "Good camera support"
        else:
            return "Unknown", "Unknown camera support"
    except:
        return "Unknown", "Unknown camera support"

def get_camera_troubleshooting_guide(browser_name):
    """Get browser-specific camera troubleshooting guide"""
    guides = {
        "Chrome": """
        **Chrome Camera Setup:**
        1. Click the camera icon (📷) in the address bar
        2. Select "Allow" for camera access
        3. If blocked, click "Site Settings" → "Camera" → "Allow"
        4. Refresh the page after allowing permissions
        """,
        "Firefox": """
        **Firefox Camera Setup:**
        1. Look for camera icon in address bar
        2. Click "Allow" when prompted
        3. If blocked, click the shield icon → "Permissions" → "Camera" → "Allow"
        4. Refresh the page after allowing permissions
        """,
        "Safari": """
        **Safari Camera Setup:**
        1. Safari may block camera access by default
        2. Go to Safari → Preferences → Websites → Camera
        3. Set to "Allow" for this site
        4. Refresh the page
        **Note:** Safari has limited webcam support
        """,
        "Edge": """
        **Edge Camera Setup:**
        1. Click the camera icon in address bar
        2. Select "Allow" for camera access
        3. If blocked, click "Site permissions" → "Camera" → "Allow"
        4. Refresh the page after allowing permissions
        """,
        "Unknown": """
        **General Camera Setup:**
        1. Look for camera/microphone icons in address bar
        2. Click and allow camera access
        3. Check browser settings for camera permissions
        4. Try refreshing the page
        5. Use Image Upload as alternative
        """
    }
    return guides.get(browser_name, guides["Unknown"])

# ============================================================================
# MODEL MANAGEMENT
# ============================================================================
@st.cache_resource
def load_model():
    """Load YOLO model once and cache it"""
    model = YOLO("best.pt")
    # Set model to evaluation mode for better performance
    model.eval()
    return model

# ============================================================================
# CAMERA MANAGEMENT
# ============================================================================
class CameraManager:
    """Manages camera resources to prevent conflicts with performance optimizations"""
    
    def __init__(self):
        self.current_camera = None
        self.current_mode = None
        self.frame_buffer = None
        self.last_frame_time = 0
        self.camera_errors = []
        self.camera_in_use = False
    
    def release_camera(self):
        """Release current camera if active"""
        if self.current_camera is not None:
            try:
                self.current_camera.release()
                time.sleep(0.1)  # Small delay to ensure proper release
            except:
                pass
            self.current_camera = None
            self.current_mode = None
            self.frame_buffer = None
            self.camera_in_use = False
            gc.collect()  # Force garbage collection
    
    def get_camera(self, camera_index, mode):
        """Get camera instance, releasing any existing camera"""
        # Always release camera when switching modes
        if self.current_mode != mode or self.current_camera is None:
            self.release_camera()
        
        if self.current_camera is None:
            try:
                self.current_camera = cv2.VideoCapture(camera_index)
                self.current_mode = mode
                self.camera_in_use = True
                
                # Set camera properties for better performance
                self.current_camera.set(cv2.CAP_PROP_FRAME_WIDTH, MAX_FRAME_WIDTH)
                self.current_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, MAX_FRAME_HEIGHT)
                self.current_camera.set(cv2.CAP_PROP_FPS, TARGET_FPS)
                
            except Exception as e:
                self.camera_errors.append(f"Camera {camera_index} error: {str(e)}")
                return None
            
        return self.current_camera
    
    def is_camera_available(self, camera_index):
        """Check if camera is available with better error handling"""
        try:
            temp_cap = cv2.VideoCapture(camera_index)
            if temp_cap.isOpened():
                ret, frame = temp_cap.read()
                temp_cap.release()
                return ret
            return False
        except Exception as e:
            self.camera_errors.append(f"Camera {camera_index} check failed: {str(e)}")
            return False
    
    def get_frame_with_rate_limit(self):
        """Get frame with rate limiting for consistent FPS"""
        current_time = time.time()
        if current_time - self.last_frame_time < FRAME_DELAY:
            return None, False
        
        if self.current_camera is None:
            return None, False
        
        try:
            ret, frame = self.current_camera.read()
            if ret:
                self.last_frame_time = current_time
                # Resize frame for better performance
                frame = self.resize_frame(frame)
            
            return frame, ret
        except Exception as e:
            self.camera_errors.append(f"Frame capture error: {str(e)}")
            return None, False
    
    def resize_frame(self, frame):
        """Resize frame to target dimensions for better performance"""
        try:
            height, width = frame.shape[:2]
            
            if width > MAX_FRAME_WIDTH or height > MAX_FRAME_HEIGHT:
                # Calculate aspect ratio
                aspect_ratio = width / height
                if width > height:
                    new_width = MAX_FRAME_WIDTH
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = MAX_FRAME_HEIGHT
                    new_width = int(new_height * aspect_ratio)
                
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            return frame
        except Exception as e:
            self.camera_errors.append(f"Frame resize error: {str(e)}")
            return frame
    
    def force_release_all_cameras(self):
        """Force release all camera resources - useful for troubleshooting"""
        self.release_camera()
        # Try to release common camera indices
        for i in range(5):
            try:
                temp_cap = cv2.VideoCapture(i)
                if temp_cap.isOpened():
                    temp_cap.release()
                    time.sleep(0.05)  # Small delay between releases
            except:
                pass
        gc.collect()

# Initialize camera manager
if 'camera_manager' not in st.session_state:
    st.session_state.camera_manager = CameraManager()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
@lru_cache(maxsize=128)
def detect_cameras():
    """Detect available cameras with caching and better error handling"""
    available_cameras = []
    for i in range(10):
        if st.session_state.camera_manager.is_camera_available(i):
            available_cameras.append(i)
    return tuple(available_cameras)  # Convert to tuple for caching

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
            st.image(image, channels="BGR")
        else:
            st.image(image)
    
    with col2:
        st.write("### Detection Result")
        result_img = results[0].plot()
        st.image(result_img, channels="BGR")
    
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
# PERFORMANCE MONITORING
# ============================================================================
class PerformanceMonitor:
    """Monitor and display performance metrics"""
    
    def __init__(self):
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.detection_times = []
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        elapsed = current_time - self.fps_start_time
        
        if elapsed >= 1.0:  # Update every second
            fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = current_time
            return fps
        return None
    
    def add_detection_time(self, detection_time):
        """Add detection time for averaging"""
        self.detection_times.append(detection_time)
        if len(self.detection_times) > 10:  # Keep last 10 measurements
            self.detection_times.pop(0)
    
    def get_avg_detection_time(self):
        """Get average detection time"""
        if self.detection_times:
            return sum(self.detection_times) / len(self.detection_times)
        return 0

# Initialize performance monitor
if 'performance_monitor' not in st.session_state:
    st.session_state.performance_monitor = PerformanceMonitor()

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    st.title("Group 7: Smart Fan Control System Using Object Detection")
    
    # Load model
    global model
    model = load_model()
    
    # Get browser information
    browser_name, browser_support = get_browser_info()
    
    # Sidebar controls
    st.sidebar.title("Controls")
    st.sidebar.info("📱 **Mobile Tip**: Use Image Upload for best mobile experience!")
    
    # Browser compatibility info
    st.sidebar.markdown("### 🌐 Browser Info")
    st.sidebar.info(f"**Browser**: {browser_name}\n**Camera Support**: {browser_support}")
    
    # Performance settings in sidebar
    st.sidebar.markdown("### ⚡ Performance Settings")
    global TARGET_FPS, FRAME_DELAY
    TARGET_FPS = st.sidebar.slider("Target FPS", 5, 30, 15, 1)
    FRAME_DELAY = 1.0 / TARGET_FPS
    
    # Main option selection
    option = st.sidebar.radio("Choose input source:", ("Image Upload", "Webcam"), index=0)
    
    # Track last option to release camera when switching
    if 'last_input_option' not in st.session_state:
        st.session_state.last_input_option = None
    
    # Release camera when switching from webcam to image upload
    if st.session_state.last_input_option == "Webcam" and option == "Image Upload":
        st.session_state.camera_manager.release_camera()
        st.session_state.running_realtime = False
        st.info("Switched to Image Upload mode. Camera resources have been released.")
    
    st.session_state.last_input_option = option
    
    if option == "Image Upload":
        image_upload_mode()
    elif option == "Webcam":
        webcam_mode(browser_name)

def image_upload_mode():
    """Handle image upload mode"""
    st.write("### Image Upload Detection")
    
    st.info("📱 **Mobile Tip**: You can take a photo with your camera and upload it for detection!")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Optimize image loading
        image = Image.open(uploaded_file)
        
        # Resize large images for better performance
        if max(image.size) > 1024:
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        
        img_array = np.array(image)
        
        # Time the detection
        start_time = time.time()
        results = model(img_array)
        detection_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Update performance metrics
        st.session_state.performance_monitor.add_detection_time(detection_time)
        
        # Display results
        display_detection_results(image, results)
        
        # Show performance info
        st.sidebar.markdown("### 📊 Performance")
        st.sidebar.metric("Detection Time", f"{detection_time:.1f}ms")

def webcam_mode(browser_name):
    """Handle webcam mode with browser-specific handling"""
    st.write("### Webcam Detection")
    
    # Show browser-specific camera guide
    st.info(f"""
    📷 **Camera Access Guide for {browser_name}**
    
    {get_camera_troubleshooting_guide(browser_name)}
    
    **Alternative**: If webcam doesn't work, use Image Upload mode instead.
    """)
    
    # Add camera reset button for troubleshooting
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔧 Reset Camera Resources"):
            st.session_state.camera_manager.force_release_all_cameras()
            st.success("Camera resources released! Try switching modes now.")
    
    with col2:
        if st.button("🔄 Refresh Page"):
            st.rerun()
    
    # Webcam mode selection
    webcam_mode = st.radio("Choose webcam mode:", ("Capture Image", "Real-Time Detection"), index=0)
    
    # Release camera when switching modes
    if 'last_webcam_mode' not in st.session_state:
        st.session_state.last_webcam_mode = None
    
    if st.session_state.last_webcam_mode != webcam_mode:
        st.session_state.camera_manager.release_camera()
        st.session_state.last_webcam_mode = webcam_mode
        st.info(f"Switched to {webcam_mode} mode. Camera resources have been released.")
    
    # Confidence threshold
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.1)
    
    if webcam_mode == "Capture Image":
        webcam_capture_mode(confidence_threshold, browser_name)
    elif webcam_mode == "Real-Time Detection":
        webcam_realtime_mode(confidence_threshold, browser_name)

def webcam_capture_mode(confidence_threshold, browser_name):
    """Handle webcam capture mode with browser compatibility"""
    try:
        # Release any existing camera resources for capture mode
        if st.session_state.camera_manager.current_mode != "capture":
            st.session_state.camera_manager.release_camera()
        
        if st.button("🔧 Test Camera Access"):
            st.info(f"Testing camera access for {browser_name}... Please allow camera permissions if prompted.")
        
        webcam_image = st.camera_input("📷 Click here to take a picture for detection")
        
        if webcam_image is not None:
            image = Image.open(webcam_image)
            img_array = np.array(image)
            
            # Time the detection
            start_time = time.time()
            results = model(img_array, conf=confidence_threshold)
            detection_time = (time.time() - start_time) * 1000
            
            # Update performance metrics
            st.session_state.performance_monitor.add_detection_time(detection_time)
            
            display_detection_results(image, results, confidence_threshold)
            
            # Show performance info
            st.sidebar.markdown("### 📊 Performance")
            st.sidebar.metric("Detection Time", f"{detection_time:.1f}ms")
            
            # Release camera after capture
            st.session_state.camera_manager.release_camera()
        
        else:
            st.write("📷 **Click the camera button above to take a picture for detection**")
            st.warning(f"""
            ⚠️ **Camera Not Working in {browser_name}?**
            
            **Try these steps:**
            1. **Check browser permissions** (see guide above)
            2. **Close other apps** using the camera (Zoom, Teams, etc.)
            3. **Try a different browser** (Chrome, Firefox, Edge)
            4. **Use Image Upload** instead (works on all browsers)
            5. **Check camera drivers** are working
            6. **Click 'Reset Camera Resources'** button above
            
            **Browser-specific tips:**
            {get_camera_troubleshooting_guide(browser_name)}
            """)
            
    except Exception as e:
        st.error(f"Webcam error in {browser_name}: {str(e)}")
        # Release camera on error
        st.session_state.camera_manager.release_camera()
        st.info(f"""
        🔧 **Troubleshooting for {browser_name}:**
        
        - **Follow the browser guide** above
        - **Close other apps** that might be using the camera
        - **Try a different browser** if this one doesn't work
        - **Use Image Upload** as an alternative
        - **Check your camera drivers** are working
        - **Click 'Reset Camera Resources'** button above
        """)

def webcam_realtime_mode(confidence_threshold, browser_name):
    """Handle webcam real-time detection mode with browser compatibility"""
    st.info(f"""
    🔄 **Real-Time Detection Mode for {browser_name}**
    
    This mode provides continuous detection from your webcam.
    **Note**: Real-time mode may not work on all browsers/devices.
    If it doesn't work, try the Capture Image mode or Image Upload.
    """)
    
    # Release any existing camera resources for realtime mode
    if st.session_state.camera_manager.current_mode != "realtime":
        st.session_state.camera_manager.release_camera()
    
    # Camera detection and selection
    st.sidebar.markdown("### 📹 Camera Detection")
    
    if st.sidebar.button("🔍 Detect Available Cameras"):
        with st.spinner("Detecting cameras..."):
            available_cameras = detect_cameras()
            if available_cameras:
                st.sidebar.success(f"Found cameras: {list(available_cameras)}")
                st.session_state.available_cameras = available_cameras
            else:
                st.sidebar.error("No cameras detected!")
                st.sidebar.info(f"""
                **Troubleshooting for {browser_name}:**
                1. Check if camera is connected
                2. Close other apps using camera
                3. Try Image Upload instead
                4. Check browser permissions
                5. Click 'Reset Camera Resources' button
                """)
    
    # Camera source selection
    if 'available_cameras' in st.session_state and st.session_state.available_cameras:
        camera_index = st.sidebar.selectbox(
            "Select Camera:", 
            list(st.session_state.available_cameras),
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
    
    # Performance metrics area
    performance_placeholder = st.empty()
    
    # Initialize real-time detection state
    if 'running_realtime' not in st.session_state:
        st.session_state.running_realtime = False
    
    if start_button:
        st.session_state.running_realtime = True
        
        try:
            # Get camera instance
            cap = st.session_state.camera_manager.get_camera(camera_index, "realtime")
            
            if cap is None or not cap.isOpened():
                st.error(f"Could not open camera {camera_index} in {browser_name}. Please check if your camera is connected and not being used by another application.")
                st.info(f"""
                **Troubleshooting for {browser_name}:**
                1. **Click 'Detect Available Cameras'** to find working cameras
                2. **Close other apps** that might be using the camera
                3. **Try a different camera index**
                4. **Use Image Upload** as an alternative
                5. **Check browser permissions** (see guide above)
                6. **Click 'Reset Camera Resources'** button above
                """)
                st.session_state.running_realtime = False
                return
            
            # Main real-time detection loop with performance optimizations
            while st.session_state.running_realtime:
                # Get frame with rate limiting
                frame, ret = st.session_state.camera_manager.get_frame_with_rate_limit()
                
                if not ret or frame is None:
                    continue
                
                # Show original frame
                original_placeholder.image(frame, channels="BGR")
                
                # Time the detection
                start_time = time.time()
                results = model(frame, conf=confidence_threshold)
                detection_time = (time.time() - start_time) * 1000
                
                # Update performance metrics
                st.session_state.performance_monitor.add_detection_time(detection_time)
                fps = st.session_state.performance_monitor.update_fps()
                
                # Get detection result
                if len(results) > 0:
                    result_img = results[0].plot()
                    detection_placeholder.image(result_img, channels="BGR")
                    
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
                
                # Display performance metrics
                if fps is not None:
                    avg_detection_time = st.session_state.performance_monitor.get_avg_detection_time()
                    performance_placeholder.markdown(f"""
                    ### ⚡ Performance Metrics
                    - **FPS**: {fps:.1f}
                    - **Avg Detection Time**: {avg_detection_time:.1f}ms
                    - **Current Detection**: {detection_time:.1f}ms
                    """)
                
        except Exception as e:
            st.error(f"Error in real-time detection for {browser_name}: {str(e)}")
            st.session_state.running_realtime = False
            # Release camera on error
            st.session_state.camera_manager.release_camera()
    
    if stop_button:
        st.session_state.running_realtime = False
        st.session_state.camera_manager.release_camera()
        st.success("Real-time detection stopped. Camera resources released.")

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
    st.sidebar.markdown("7. **Performance**: Adjust FPS slider for optimal performance")
    st.sidebar.markdown("8. **Browser Issues**: Check browser-specific camera guide")
    
    st.sidebar.markdown("### 🔧 Camera Troubleshooting:")
    st.sidebar.markdown("• **Can't access camera?** Click 'Reset Camera Resources'")
    st.sidebar.markdown("• **Switching modes?** Camera is automatically released")
    st.sidebar.markdown("• **Still having issues?** Try Image Upload mode")
    st.sidebar.markdown("• **Browser problems?** Check camera permissions")

def cleanup_session_state():
    """Clean up session state when app is closed or refreshed"""
    if 'camera_manager' in st.session_state:
        st.session_state.camera_manager.release_camera()
    if 'running_realtime' in st.session_state:
        st.session_state.running_realtime = False

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    try:
        main()
        display_sidebar_info()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        # Ensure camera is released on error
        if 'camera_manager' in st.session_state:
            st.session_state.camera_manager.release_camera()
    finally:
        # Always cleanup resources
        cleanup_session_state() 