import streamlit as st
import torch
from PIL import Image
import cv2
import numpy as np

# Load the two models
xray_model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/xray_model.pt')  # YOLOv9 X-ray model
camera_model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/camera_model.pt')  # YOLO model for camera images

# Function to draw bounding boxes
def draw_boxes(image, results):
    detections = results.xyxy[0].cpu().numpy()  # Convert GPU tensor to CPU numpy
    classes = results.names
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        label = f"{classes[int(cls)]} {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Streamlit App
st.title("Dental Disease Detection App")
st.write("Upload an X-ray or Camera image to detect dental problems.")

# Step 1: Select image type
image_type = st.radio("Select Image Type", ["X-ray", "Camera"])

# Step 2: Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    np_image = np.array(image)

    # Step 3: Process the image
    if st.button("Detect Diseases"):
        with st.spinner("Processing..."):
            # Convert image to required format for the model
            model_input = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

            # Choose the model based on image type
            if image_type == "X-ray":
                results = xray_model(model_input)
            else:
                results = camera_model(model_input)

            # Draw bounding boxes
            output_image = draw_boxes(np_image.copy(), results)

        # Step 4: Display results
        st.image(output_image, caption="Detected Diseases", use_column_width=True)
