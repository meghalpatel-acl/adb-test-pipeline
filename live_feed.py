import cv2
import os

if "DISPLAY" not in os.environ:
    os.environ["DISPLAY"] = ":0"

# Open connection to the default webcam
cap = cv2.VideoCapture(0)

print("Streaming... Press 'ESC' or 'q' to close the window.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Display the live stream
    cv2.imshow('Live Video Stream', frame)

    # Stop the stream if 'q' or 'ESC' (27) is pressed
    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

# Clean up
cap.release()
cv2.destroyAllWindows()

