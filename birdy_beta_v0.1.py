import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import cv2
import threading
import json
from ultralytics import solutions
import os
from datetime import datetime
import shutil

class birdy:
    def __init__(self, root):# 初始化
        self.root = root
        self.root.title("Birdy(not yet) v0.1 beta")
        self.root.geometry("300x200")

        self.file_path = ''  # 视频路径
        self.model_path = 'yolo11n.pt'  # 模型路径
        # self.classes_to_count = [0]  # 要计数的目标类

        # 窗口布局
        self.create_widgets()

        # 时间获取
        self.current_time = datetime.now()
        self.current_time_str = self.current_time.strftime("%Y%m%d_%H%M%S")

        # 文件完整性检查
        self.files()

    def create_widgets(self):# 窗口布局
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        # 导入视频按钮
        self.b1 = tk.Button(button_frame, text="导入视频", command=self.upload)
        self.b1.pack(side="left", padx=5)

        # 处理视频按钮
        self.b2 = tk.Button(button_frame, text="开始处理", command=self.dealing_video)
        self.b2.pack(side="left", padx=5)

        # 待添加
        self.b3 = tk.Button(button_frame, text="待添加功能", command=self.info_message)
        self.b3.pack(side="left", padx=5)

        # 文件位置提示文本
        self.l1 = tk.Label(self.root, text="未添加文件")
        self.l1.pack(pady=5)

        # 进度条
        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=250, mode="determinate")
        self.progress.pack(pady=10)

        # 文字提示文本
        self.l2 = tk.Label(self.root, text='')
        self.l2.pack(pady=5)

        # 菜单栏
        self.top = tk.Menu(self.root)

        self.start = tk.Menu(self.top)
        self.deal = tk.Menu(self.top)
        self.edit = tk.Menu(self.top)

        # 分区1 开始
        self.top.add_cascade(label='开始', menu=self.start)
        self.start.add_command(label='打开目录',command=self.open_file)
        self.start.add_command(label='补全文件', command=self.files)
        self.start.add_command(label='退出',command=root.quit)

        # 分区2 处理
        self.top.add_cascade(label='处理',menu=self.deal)
        self.deal.add_command(label='打开mp4文件',command=self.upload)
        self.deal.add_command(label='处理视频',command=self.dealing_video)

        #分区3 表格
        self.top.add_cascade(label="表格",menu = self.edit)
        self.edit.add_command(label='单日图（待开发）')
        self.edit.add_command(label='多日图（待开发）')

        self.root.config(menu = self.top)

    def upload(self):# 上传视频文件
        self.file_path = filedialog.askopenfilename(
            title="选择一个MP4文件",
            filetypes=[("MP4 文件", "*.mp4")]  # 限定为MP4文件
        )
        self.l1.config(text=self.file_path if self.file_path else "未选中文件")

    def count_specific_classes(self, video_path, output_video_path, progress_callback):# 处理视频与计数
        if os.path.exists('output_specific_classes.mp4'):
            os.remove('output_specific_classes.mp4')


        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        line_points = [(0, 300), (10000, 300)]  # 定义计数线
        counter = solutions.ObjectCounter(show=True,
                                          region=line_points,
                                          model=self.model_path,
                                          line_width=1,
                                          classes=[0]
                                          )

        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数
        fps_over = 0

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                break

            im0 = counter.count(im0)  # 计数
            video_writer.write(im0)  # 写入输出视频

            # 更新进度
            fps_over += 1
            progress_percent = (fps_over / frames) * 100
            self.l2.config(text=str(counter.classwise_counts))  # 更新计数信息

            # 更新进度条
            progress_callback(progress_percent)
            self.count_stats = counter.classwise_counts

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

        # 复制原视频文件
        shutil.copy2(self.file_path, './input/' + self.current_time_str + '.mp4')

        # 导出结果视频
        output_dir = './output/'
        shutil.move('output_specific_classes.mp4', output_dir)
        new_file_name = f"{self.current_time_str}.mp4"
        os.rename(os.path.join(output_dir, 'output_specific_classes.mp4'), os.path.join(output_dir, new_file_name))

        # 导出结果文本
        file_path = "./text/" + str(self.current_time_str) + ".txt"
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(self.count_stats, file, ensure_ascii=False, indent=4)

    def update_progress(self, value):# 更新进度条
        self.progress["value"] = value
        self.root.update_idletasks()  # 强制更新UI

    def dealing_video(self):# 处理视频
        if not self.file_path:
            self.l2.config(text="请先选择视频文件")
            return

        # 单独线程视频处理与输出结果
        threading.Thread(target=self.count_specific_classes, args=(self.file_path, "output_specific_classes.mp4", self.update_progress), daemon=True).start()

    def info_message(self):# 待添加功能
        self.l2.config(text="待添加")

    def open_file(self):# 打开文件位置
        directory_path = os.path.dirname(os.path.abspath(__file__))
        os.startfile(directory_path)

    def files(self):# 检查文件夹完整性
        folders = ['output', 'text', 'chart', 'input']

        # 检查每个文件夹是否存在，如果不存在，则创建它
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)
                self.l2.config(text="文件夹缺失，已补全")

            else:
                self.l2.config(text="文件夹完整")

# 主程序
if __name__ == "__main__":
    root = tk.Tk()
    app = birdy(root)
    root.mainloop()
