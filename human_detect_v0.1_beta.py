import cv2
from ultralytics import solutions


def count_specific_classes_from_camera(model_path, classes_to_count):
    """Count specific classes of objects in real-time video from the camera."""
    cap = cv2.VideoCapture(0)  # Open the default camera (0 is usually the default webcam)

    # Get video properties (frame width, height, and FPS)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Initialize the ObjectCounter with the region of interest (if needed)
    line_points = [(300, 0), (350, 0), (350,1000), (300,1000)]  # Define a line for object counting if necessary

    counter = solutions.ObjectCounter(show=True, region=line_points, model=model_path, line_width=1,
                                      classes=classes_to_count)

    while True:
        success, im0 = cap.read()
        if not success:
            print("Failed to grab frame.")
            break

        # Perform object counting on the current frame
        im0 = counter.count(im0)

        # Optionally, print the class-wise counts
        print(counter.classwise_counts)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage: Start counting from camera
count_specific_classes_from_camera("yolo11n.pt", [0])  # For example, class 0 (person)
