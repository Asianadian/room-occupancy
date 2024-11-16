import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

video_capture = cv2.VideoCapture(0)

while True:
  ret, frame = video_capture.read()
  if not ret:
    break

  results = model(frame, conf=0.5)

  detections = results[0].boxes.data.cpu().numpy()

  seats, people = 0, 0

  for detection in detections:
    class_id = int(detection[5])

    if class_id == 0:
      people += 1
    elif class_id == 56:
      seats += 1

    # Draw bounding boxes
    x1, y1, x2, y2 = map(int, detection[:4])
    label = "Person" if class_id == 0 else "Chair"
    color = (0, 255, 0) if class_id == 0 else (255, 0, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  # Calculate occupancy percentage
  occupancy = (people / seats * 100) if seats > 0 else 0

  # Display information
  cv2.putText(frame, f"People: {people}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
  cv2.putText(frame, f"Seats: {seats}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
  cv2.putText(frame, f"Occupancy: {occupancy:.2f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

  # Show the frame
  cv2.imshow("Room Occupancy", frame)

  # Exit on pressing 'q'
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

video_capture.release()
cv2.destroyAllWindows()