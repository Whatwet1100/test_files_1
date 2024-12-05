import cv2

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

model = YOLO("yolo11n.pt")
names = model.names

cap = cv2.VideoCapture(0)  # Open the default camera (0 is usually the default webcam)
assert cap.isOpened(), "Error opening camera"

# Get video properties (frame width, height, and FPS)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Blur ratio
blur_ratio = 100

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = model.predict(im0, show=False,classes=[0,62,63],show_boxes=False)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()
    annotator = Annotator(im0, line_width=0, example=names)

    if boxes is not None:
        for box, cls in zip(boxes, clss):
            annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

            obj = im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
            blur_obj = cv2.blur(obj, (blur_ratio, blur_ratio))

            im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] = blur_obj

    cv2.imshow("ultralytics", im0)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()