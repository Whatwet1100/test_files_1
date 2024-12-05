import pygame
import cv2
from ultralytics import solutions
import threading
import sys
import random
import time

class CatchItems:
    # 初始化
    def __init__(self):
        """初始化"""
        # pygame初始化
        pygame.init()


        # 获取cv2视频
        self.cap = cv2.VideoCapture(0)
        self.w, self.h, self.fps = (int(self.cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # 初始化高度分析
        self.line_points = []
        self.counter = solutions.ObjectCounter(show=True, region=self.line_points, model="yolo11n.pt", line_width=1, classes=[0], half=True)

        # 启动获取Y坐标的线程
        self.x_thread = threading.Thread(target=self.getx)
        self.x_thread.daemon = True  # 设置为守护线程，程序退出时线程也会自动退出
        self.x_thread.start()

        # 设置屏幕尺寸
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("铃木千束特训")

        # 方块参数
        self.block_size = 20
        self.x = self.w / 2  # 方块的x坐标
        self.y = self.h / 2 + 150  # 方块的y坐标

        # 游戏参数
        self.hp = 10  # 血量
        self.font = pygame.font.SysFont('Comicsans', 24)  # 设置字体
        self.down_speed=0.03 # 下落频率，最高为1
        self.fps_game = 120

        # 游戏主循环
        self.clock = pygame.time.Clock()
        self.running = True

        # 子弹掉落列表
        self.bullets = []

        # 初始位置
        self.counter.track_positions = {1:[self.w/2,0]}

    def getx(self):
        """镜像翻转视频并获取x轴坐标"""
        while True:
            success, im0 = self.cap.read()

            if not success:
                print("NO CAMERA OR BLOCKED")
                continue

            im0 = cv2.flip(im0, 1)  # 水平翻转
            im0 = self.counter.count(im0)

            self.x = int(self.counter.track_positions[1][0])


    def drop_bullet(self):
        """创建多个子弹，每次可以生成多个子弹"""
        # 控制生成多少子弹
        num_bullets = random.randint(1, 5)  # 每次生成 1 到 3 个子弹
        for _ in range(num_bullets):
            bullet_x = random.randint(0, self.w - 20)  # 随机生成子弹的 x 位置
            bullet_y = 0  # 子弹从屏幕顶部掉落
            speed = random.randint(10, 15)  # 子弹的下落速度
            self.bullets.append([bullet_x, bullet_y, speed])  # 将子弹加入到子弹列表中

    def update_bullets(self):
        """更新所有子弹的位置，并检查与方块的碰撞"""
        for bullet in self.bullets[:]:
            bullet[1] += bullet[2]  # 子弹沿y轴下落

            # 检查子弹是否与方块碰撞
            if self.x < bullet[0] < self.x + self.block_size and self.y < bullet[1] < self.y + self.block_size:
                # 方块与子弹碰撞，血量减少
                self.hp -= 1
                self.bullets.remove(bullet)  # 删除掉落的子弹

            # 如果子弹掉出屏幕，则删除子弹
            if bullet[1] > self.h:
                self.bullets.remove(bullet)

    def game_page(self):
        """游戏界面"""
        # 每次刷新屏幕时，先绘制背景（清除之前的方块）
        self.screen.fill((255, 255, 255))  # 使用白色背景，覆盖前面的绘制内容


        pygame.draw.rect(self.screen, (0, 0, 0), (self.x, self.y, self.block_size, self.block_size))

        # 绘制掉落的子弹
        for bullet in self.bullets:
            pygame.draw.rect(self.screen, (255, 0, 0), (bullet[0], bullet[1], 10, 10))  # 绘制红色方块作为子弹

        # 显示分数等信息
        score_text = self.font.render(f"Health: {self.hp}", True, (0, 0, 0))
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()  # 更新显示
        self.clock.tick(self.fps_game)  # 控制帧率

    # 游戏主循环
    def run(self):
        """主循环"""
        for i in range(7):
            print("game will start in" + str(7 - i))
            time.sleep(1)

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            # 每隔一段时间掉落一个子弹
            if random.random() < self.down_speed:  # 设定子弹掉落的概率，控制频率
                self.drop_bullet()

            # 更新子弹位置及检查碰撞
            self.update_bullets()

            # 刷新游戏界面
            self.game_page()

        pygame.quit()
        self.cap.release()
        sys.exit()


if __name__ == "__main__":
    app = CatchItems()
    app.run()
