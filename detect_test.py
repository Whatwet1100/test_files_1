from tkinter import Tk, Label, Button
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2

# 全局变量，用于控制截图功能
is_saving = False
model = YOLO("yolo11n.pt")


def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results


def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results

def update_frame():
    global is_saving

    # 从摄像头捕获一帧
    ret, frame = cap.read()
    if ret:
        # 转换颜色从BGR到RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 将numpy数组转换为Image
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)

        # 更新标签中的图像
        label.imgtk = img_tk
        label.configure(image=img_tk)

        # 如果正在保存，保存当前帧为1.png
        if is_saving:
            cv2.imwrite('1.png', frame)
            image = cv2.imread("1.png")
            result_img, _ = predict_and_detect(model, image, classes=[], conf=0.5)

            cv2.imshow("Image", result_img)
            cv2.imwrite("2.png", result_img)

    # 每隔10毫秒更新一次图像
    label.after(10, update_frame)


def toggle_save():
    global is_saving
    is_saving = not is_saving  # 切换状态
    if is_saving:
        button.config(text="停止识别")
    else:
        button.config(text="开始识别")


# 初始化Tkinter窗口
root = Tk()
root.title("Camera Feed")

# 创建一个标签用于显示视频
label = Label(root)
label.pack()

# 创建一个按钮用于开始/停止识别
button = Button(root, text="开始识别", command=toggle_save)
button.pack()

# 打开摄像头
cap = cv2.VideoCapture(0)

# 启动更新视频帧的循环
update_frame()

# 运行主循环
root.mainloop()

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()
