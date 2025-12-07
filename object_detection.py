import cv2
import gradio as gr
from ultralytics import YOLO
import tempfile
import os
import sys

# ------------ Handle model path for PyInstaller ------------
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ------------ Load YOLOv10 Model ------------
model_path = resource_path("yolov10x.pt")
model = YOLO(model_path)

# ------------ IMAGE DETECTION ------------
def detect_image(image):
    results = model.predict(image, conf=0.5)
    annotated = results[0].plot()
    return annotated

# ------------ VIDEO DETECTION ------------
def detect_video(video_file):
    cap = cv2.VideoCapture(video_file)
    temp_dir = tempfile.mkdtemp()
    temp_output = os.path.join(temp_dir, "output.mp4")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=0.5)
        annotated = results[0].plot()
        out.write(annotated)

    cap.release()
    out.release()
    return temp_output

# ------------ WEBCAM DETECTION ------------
def detect_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=0.5)
        annotated = results[0].plot()
        annotated = cv2.resize(annotated, (800, 600))
        cv2.imshow("YOLOv10 Webcam Detection - Press Q to quit", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return "Webcam session ended."

# ------------ GRADIO FRONTEND ------------
def launch_app():
    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown("## 🧠 YOLOv10 Object Detection")
        gr.Markdown("Upload an **image**, **video**, or use your **webcam** for real-time detection!")

        with gr.Tab("📷 Image Detection"):
            image_input = gr.Image(type="numpy", label="Upload Image")
            image_output = gr.Image(label="Detected Output")
            image_btn = gr.Button("Detect Objects")
            image_btn.click(fn=detect_image, inputs=image_input, outputs=image_output)

        with gr.Tab("🎞️ Video Detection"):
            video_input = gr.Video(label="Upload MP4 Video")
            video_output = gr.Video(label="Detected Video")
            video_btn = gr.Button("Detect in Video")
            video_btn.click(fn=detect_video, inputs=video_input, outputs=video_output)

        with gr.Tab("🎥 Webcam Detection"):
            webcam_btn = gr.Button("Start Webcam Detection")
            webcam_output = gr.Textbox(label="Status")
            webcam_btn.click(fn=detect_webcam, outputs=webcam_output)

    # ✅ Run locally and auto-open in Chrome
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)

# ------------ MAIN CALL ------------
if __name__ == "__main__":
    launch_app()
