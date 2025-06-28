import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import time

st.set_page_config(page_title="Smart Fan Control System Using Object Detection", layout="wide")
st.title("Smart Fan Control System Using Object Detection")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Detect mobile device
def is_mobile():
    user_agent = st.get_option("server.userAgent")
    mobile_keywords = ['Mobile', 'Android', 'iPhone', 'iPad', 'Windows Phone']
    return any(keyword in user_agent for keyword in mobile_keywords)

# Sidebar for controls
st.sidebar.title("Controls")

# Show mobile warning if detected
if is_mobile():
    st.sidebar.warning("📱 Mobile detected: Webcam may not work. Use Image Upload instead.")
    default_option = "Image Upload"
else:
    default_option = "Webcam"

option = st.sidebar.radio("Choose input source:", ("Webcam", "Image Upload"), index=0 if default_option == "Webcam" else 1)

# Function to check detection status and control fan
def check_detection_status(results):
    if len(results) == 0 or len(results[0].boxes) == 0:
        return [], "No objects detected", "fan_off"
    
    detected_classes = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        class_name = model.names[cls]
        detected_classes.append(class_name)
    
    detected_classes = list(set(detected_classes))  # Remove duplicates
    
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

if option == "Webcam":
    st.write("### Real-Time Webcam Detection")
    
    # Mobile warning for webcam
    if is_mobile():
        st.warning("""
        📱 **Mobile Device Detected**
        
        Webcam access on mobile devices may not work due to browser restrictions. 
        If webcam doesn't work, please use the **Image Upload** option instead.
        """)
    
    # Webcam controls
    col1, col2, col3 = st.columns(3)
    with col1:
        start_button = st.button("Start Webcam", key="start")
    with col2:
        stop_button = st.button("Stop Webcam", key="stop")
    with col3:
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.1)
    
    # Camera source selection (only show on desktop)
    if not is_mobile():
        camera_index = st.sidebar.number_input("Camera Index", min_value=0, max_value=10, value=0, step=1, help="0=default camera, 1=second camera, etc.")
    else:
        camera_index = 0  # Default camera for mobile
    
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
    
    # Initialize webcam
    if 'cap' not in st.session_state:
        st.session_state.cap = None
        st.session_state.running = False
    
    if start_button:
        st.session_state.running = True
        if st.session_state.cap is None:
            st.session_state.cap = cv2.VideoCapture(camera_index)
            if not st.session_state.cap.isOpened():
                st.error("Could not open webcam. Please check if your webcam is connected.")
                st.session_state.running = False
    
    if stop_button:
        st.session_state.running = False
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
    
    # Main webcam loop
    if st.session_state.running and st.session_state.cap is not None:
        try:
            while st.session_state.running:
                ret, frame = st.session_state.cap.read()
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
                    
                    # Display fan animation if both objects detected
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
            st.error(f"Error in webcam stream: {str(e)}")
            st.session_state.running = False
    
    # Cleanup when app stops
    if st.session_state.cap is not None:
        st.session_state.cap.release()

elif option == "Image Upload":
    st.write("### Image Upload Detection")
    
    # Mobile-friendly upload instructions
    if is_mobile():
        st.info("📱 **Mobile Tip**: You can take a photo with your camera and upload it for detection!")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Create two columns for side-by-side display
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Original Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.write("### Detection Result")
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            results = model(img_array)
            result_img = results[0].plot()
            st.image(result_img, caption="Detection Result", use_container_width=True)
        
        # Check detection status and control fan
        detected_classes, status, fan_state = check_detection_status(results)
        
        # Display status
        if fan_state == "fan_on":
            st.success(status)
        else:
            st.warning(status)
        
        # Display fan animation if both objects detected
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
        
        # Display detection info
        if detected_classes:
            st.write("### 📊 Detection Details:")
            for class_name in detected_classes:
                st.write(f"• {class_name}")

# Add some helpful information
st.sidebar.markdown("---")
st.sidebar.markdown("### Instructions:")
st.sidebar.markdown("1. **Webcam**: Click 'Start Webcam' to begin real-time detection")
st.sidebar.markdown("2. **Image Upload**: Upload an image to detect objects")
st.sidebar.markdown("3. **Fan Control**: Fan turns ON when both chair and minifan are detected")
st.sidebar.markdown("4. **Mobile**: Use Image Upload for best mobile experience")
st.sidebar.markdown("5. Adjust confidence threshold to filter detections") 