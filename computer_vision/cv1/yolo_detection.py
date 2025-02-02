import streamlit as st
from ultralytics import YOLO
import io, os, json
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import cv2
import numpy as np

# Create a draw object
try:
    font = ImageFont.truetype("arial.ttf", 20)  # Use a font file if available
except IOError:
    font = ImageFont.load_default()  # Default font if arial.ttf is missing

def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")

# This is the root directory for YOLO Turkish Licence Plate Detection.
cv_root_dir = "[SAME-PATH-WHERE-INTERFACE.PY-IS]\\computer_vision\\"

menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""  # Hide main menu style
# Main title of streamlit application
main_title_cfg = """<div>
                        <h1 style="color:#00000; text-align:center; font-size:40px; margin-top:-50px;
                                                font-family: 'Archivo', sans-serif; margin-bottom:20px;">
                            YOLOv9 Licence Plate Detection
                        </h1>
                    </div>"""
st.markdown(menu_style_cfg, unsafe_allow_html=True)
st.markdown(main_title_cfg, unsafe_allow_html=True)

## User Configuration for frame source
st.title("User Configuration")
source = st.selectbox(
            "Source of the Frame(s)",
            ("PICTURE","VIDEO"),
        )

## Model Selection
selected_model = f"{cv_root_dir}models\\yolov9.pt"
model = YOLO(selected_model)
## Loading class names from model
class_names = list(model.names.values())  # Convert dictionary to list of class names
st.success(f"Model selected successfully!\n {selected_model}")
## Display class names
st.info("Detected Classes: " + ", ".join(class_names))

## Source Uploading
source_file_name = ""
if source == "VIDEO":
    vid_file = st.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])

    if vid_file is not None:
        g = io.BytesIO(vid_file.read())  # Read file into memory
        with open("ultralytics.mp4", "wb") as out:
            out.write(g.read())  # Save uploaded file locally
        source_file_name = "ultralytics.mp4"

        # Generate timestamped output folder
        now = datetime.now().strftime("%m%d%Y_%H%M%S")
        root_dir = f"{cv_root_dir}cv1\\runs\\detect\\{now}"
        ensure_folder_exists(root_dir)

        out_json_path = f"{root_dir}\\detections.json"
        out_video_path = f"{root_dir}\\detections.mp4"

        # Open video file
        cap = cv2.VideoCapture(source_file_name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format
        out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

        # Store detection results
        detections = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame from OpenCV to PIL Image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image)

            # Run inference
            results = model(image)

            # Process results
            frame_detections = []
            for result in results:
                for box in result.boxes:
                    class_name = result.names[int(box.cls)]
                    confidence = float(box.conf)
                    bbox = box.xyxy.tolist()[0]  # [x1, y1, x2, y2]

                    # Store detection data
                    frame_detections.append({
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": bbox
                    })

                    # Draw bounding box
                    draw.rectangle(bbox, outline="blue", width=5)

                    # Draw label
                    text = f"{class_name} {confidence:.2f}"
                    text_position = (bbox[0], bbox[1] - 10)
                    draw.text(text_position, text, fill="red")

            detections.append(frame_detections)

            # Convert back to OpenCV format
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            out_video.write(frame)

        # Save detection results as JSON
        with open(out_json_path, "w") as f:
            json.dump(detections, f, indent=4)

        # Release resources
        cap.release()
        out_video.release()

        st.info(f"Detection results are saved in: cv1\\runs\\detect\\{now}")

        # Display video in Streamlit
        st.video(out_video_path)

elif source == "PICTURE":
    im_file = st.file_uploader("Upload Image File", type = ["JPG", "JPEC"])
    if im_file is not None:
        im1 = Image.open(im_file)
        draw = ImageDraw.Draw(im1)  

        results = model(im1)

        now = datetime.now().strftime("%m%d%Y_%H%M%S")
        root_dir = f"{cv_root_dir}cv1\\runs\\detect\\{now}"
        ensure_folder_exists(root_dir)

        out_path = f"{root_dir}\\detections.json"
        # results.save(root_dir)
        output = []
        for result in results:
            for box in result.boxes:
                class_name = result.names[int(box.cls)]
                confidence = float(box.conf)
                bbox = box.xyxy.tolist()[0]  # [x1, y1, x2, y2]

                # Append to output list
                output.append({
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": bbox
                })

                # Draw bounding box
                draw.rectangle(bbox, outline="blue", width=5)

                # Draw label
                text = f"{class_name} {confidence:.2f}"
                text_position = (bbox[0], bbox[1] - 10)
                draw.text(text_position, text, fill="red", font=font)
        
        with open(out_path, "w") as f:
            json.dump(output, f, indent=4)

        im1 = im1.convert('RGB')
        im1.save(f"{root_dir}\\detections.jpg")

        st.info(f"Detection results are saved in: cv1\\runs\\detect\\{now}")
        st.image(im1)