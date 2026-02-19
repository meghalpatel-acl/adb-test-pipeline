import cv2
from ultralytics import YOLO
import os

if "DISPLAY" not in os.environ:
    os.environ["DISPLAY"] = ":0"

MODEL_PATH = "./models/yolo11n.engine"
# 1. Load the exported TensorRT engine model
# Ensure 'yolo11n.engine' is in your current directory
model = YOLO(MODEL_PATH, task="detect")

# 2. Open the camera feed
cap = cv2.VideoCapture(0)

print("Starting TensorRT inference. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Run inference on the current frame
    # 'stream=True' is more memory-efficient for video processing
    results = model.predict(source=frame, stream=True, conf=0.25, verbose=False)

    for result in results:
        # 4. Use Ultralytics' built-in plotting to draw boxes/labels on the frame
        annotated_frame = result.plot()

        # 5. Display the frame using OpenCV
        cv2.imshow("YOLO11 TensorRT Feed", annotated_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()

