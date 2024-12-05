import cv2

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

model = YOLO("yolo11n-seg.pt")  # segmentation model
cap = cv2.VideoCapture(0)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

while True:
    ret, im0 = cap.read()
    if not ret:
        break

    results = model.predict(im0,classes=[0])
    annotator = Annotator(im0, line_width=2)

    if results[0].masks is not None:
        clss = results[0].boxes.cls.cpu().tolist()
        masks = results[0].masks.xy
        for mask, cls in zip(masks, clss):
            annotator.seg_bbox(mask=mask, mask_color=colors(int(cls), True))

    out.write(im0)
    cv2.imshow("instance-segmentation", im0)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()