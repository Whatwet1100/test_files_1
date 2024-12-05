import cv2

from ultralytics import solutions

cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))


# In case you want to apply object counting + heatmaps, you can pass region points.
region_points = [(20, 400), (1080, 400)]  # Define line points
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # Define region points
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]  # Define polygon points

# Init heatmap
heatmap = solutions.Heatmap(
    show=True,  # Display the output
    model="yolo11n.pt",  # Path to the YOLO11 model file
    colormap=cv2.COLORMAP_PARULA,  # Colormap of heatmap
    # region=region_points,  # If you want to do object counting with heatmaps, you can pass region_points
    classes=[0],  # If you want to generate heatmap for specific classes i.e person and car.
    show_in=True,  # Display in counts
    show_out=True,  # Display out counts
    line_width=2,  # Adjust the line width for bounding boxes and text display
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    im0 = heatmap.generate_heatmap(im0)


cap.release()

cv2.destroyAllWindows()