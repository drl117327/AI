# motion_controller.py (V3 - 纯净驱动版)

import requests
import time
import cv2


# ==============================================================================
# ======================== PART 1: 配置区域 ==================================
# ==============================================================================

class Config:
    SERVER_URL = "http://127.0.0.1:8082/MyWcfService/getstring"
    SERIAL_PORT = 'COM10'
    CAMERA_INDEX = 1


# ==============================================================================
# =================== PART 2: 功能实现区域 ======================
# ==============================================================================

class MotionController:
    def __init__(self, config):
        """
        初始化运动控制器。
        这个类只负责与硬件通信和执行基本动作。
        """
        self.config = config
        self.resource_handle = None
        self.camera = None

        # 连接硬件和摄像头
        self._connect_to_service()
        self._init_camera()

    def _init_camera(self):
        """初始化摄像头"""
        self.camera = cv2.VideoCapture(self.config.CAMERA_INDEX)
        if not self.camera.isOpened():
            raise RuntimeError("ERROR: 无法打开摄像头！")


    def _connect_to_service(self):
        """连接到硬件控制服务并获取资源号，增加了重试机制。"""
        print(f"INFO: 正在尝试连接到端口 {self.config.SERIAL_PORT}...")

        max_retries = 5  # 最多重试5次
        for attempt in range(max_retries):
            params = {"duankou": self.config.SERIAL_PORT, "hco": 0, "daima": 0}
            try:
                response = requests.get(self.config.SERVER_URL, params=params, timeout=5)
                response.raise_for_status()
                response_text = response.text.strip()

                import json
                try:
                    # 优先尝试解析JSON
                    data = json.loads(response_text)
                    resource_id = int(data)
                except (json.JSONDecodeError, TypeError):
                    # 如果失败，再尝试用eval
                    resource_id = int(eval(response_text))

                if resource_id > 0:
                    self.resource_handle = resource_id
                    print(f"SUCCESS: (在第 {attempt + 1} 次尝试后) 成功获取到资源号: {self.resource_handle}")
                    time.sleep(2)
                    return  # 成功了就直接退出函数

                else:  # resource_id <= 0
                    print(f"  - WARNING (尝试 {attempt + 1}/{max_retries}): 获取资源号失败 (ID <= 0)。将在2秒后重试...")

            except requests.exceptions.RequestException as e:
                print(f"  - WARNING (尝试 {attempt + 1}/{max_retries}): 无法连接到本地服务: {e}。将在2秒后重试...")
            except Exception as e:
                print(f"  - WARNING (尝试 {attempt + 1}/{max_retries}): 处理响应时出错: {e}。将在2秒后重试...")

            time.sleep(2)  # 每次重试前都等待2秒

        # 如果循环结束了还没成功，就抛出最终的错误
        raise ConnectionError(
            f"在 {max_retries} 次尝试后，仍未能从本地服务获取到有效的资源号。请检查硬件服务是否正在运行且稳定。")

    def _send_command(self, command_code: str):
        """向硬件发送一个指令字符串。"""
        if self.resource_handle is None:
            print("ERROR: 资源号无效，无法发送指令。")
            return
        params = {"duankou": "0", "hco": self.resource_handle, "daima": command_code}
        # print(f"DEBUG: 发送指令代码: {command_code}") # 可以改为DEBUG级别
        try:
            requests.get(self.config.SERVER_URL, params=params, timeout=2)
            time.sleep(0.2)  # 留出硬件响应时间
        except requests.exceptions.RequestException as e:
            print(f"WARNING: 发送指令 '{command_code}' 时出错: {e}")


    def move_to(self, x: float, y: float):
        """移动到指定的【物理】坐标。"""
        print(f"INFO: 移动到物理坐标 ({x:.2f}, {y:.2f})")
        move_command = f"x{int(x)}y{int(y)}"
        self._send_command(move_command)

    def click(self):
        """在当前位置执行一次点击（下降后抬起）。"""
        print("INFO: 执行点击...")
        self._send_command("z9")  # 下降
        time.sleep(0.1)
        self._send_command("z-8")  # 抬起

    def swipe(self, x1: float, y1: float, x2: float, y2: float, duration: float = 1.0):
        """执行一次从物理坐标A到物理坐标B的滑动。"""
        print(f"INFO: 从({x1:.2f},{y1:.2f}) 滑动到 ({x2:.2f},{y2:.2f})")
        self.move_to(x1, y1)
        self._send_command("z3")  # 下降到滑动高度
        self.move_to(x2, y2)
        time.sleep(duration)
        self._send_command("z-8")  # 抬起

    def move_and_click(self, x: float, y: float):
        """组合操作：移动到指定物理坐标然后点击。"""
        self.move_to(x, y)
        time.sleep(0.1)
        self.click()

    def capture_image(self):
        """从控制器关联的摄像头捕获一帧图像。"""
        if not self.camera or not self.camera.isOpened():
            print("ERROR: 摄像头未初始化或已关闭。")
            return None
        ret, frame = self.camera.read()
        if not ret:
            print("WARNING: 无法从摄像头读取画面。")
            return None
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return rotated_frame

    def shutdown(self):
        """安全关闭与硬件的连接并释放摄像头。"""
        print("INFO: 正在关闭控制器...")
        if self.resource_handle:
            self._send_command("0")  # 发送关闭指令
            self.resource_handle = None
            print("  - 硬件资源已释放。")
        if self.camera:
            self.camera.release()
            self.camera = None
            print("  - 摄像头已释放。")
        print("控制器已关闭。")


# ==============================================================================
# ======================== PART 3: 主程序入口=======================
# ==============================================================================

if __name__ == "__main__":
    # 这里的代码只用于【单独测试】这个驱动文件是否正常。
    print("--- 正在单独测试 MotionController 驱动 ---")
    controller = None
    try:
        my_config = Config()
        controller = MotionController(my_config)

        # 测试1: 移动到物理坐标 (10, 10)
        controller.move_to(10, 10)
        time.sleep(1)

        # 测试2: 在 (50, 50) 的位置点击
        controller.move_and_click(50, 50)
        time.sleep(1)

        # 测试3: 捕获一张图片并保存
        img = controller.capture_image()
        if img is not None:
            cv2.imwrite("driver_test_capture.png", img)
            print("INFO: 已捕获测试图片到 driver_test_capture.png")

        # 测试4: 返回原点
        controller.move_to(0, 0)

    except Exception as e:
        print(f"\nCRITICAL: 驱动测试失败: {e}")
    finally:
        if controller:
            controller.shutdown()
        print("\n--- 驱动测试结束 ---")