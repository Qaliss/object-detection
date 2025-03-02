import cv2
from ultralytics import YOLO

# Load trained YOLO model 
model = YOLO(r'C:\Users\naras\Desktop\Trash Object Detection Model\best3.pt') 

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame)

    # Get the bounding boxes and labels from the results
    boxes = results[0].boxes 
    labels = boxes.cls  # Class labels
    confs = boxes.conf  # Confidence scores
    coords = boxes.xyxy  # Coordinates of the bounding boxes (x1, y1, x2, y2)

    # Draw the bounding boxes on the frame
    for coord, label, conf in zip(coords, labels, confs):
        x1, y1, x2, y2 = map(int, coord)  # Convert to integers
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw rectangle
        cv2.putText(frame, f'{model.names[int(label)]} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Add label

    # Show the processed frame (live feed)
    cv2.imshow("YOLO Object Detection - Live Feed", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
