# local_hardware_gateway.py

from flask import Flask, request, jsonify, Response
import cv2
import motion_controller
import time
import threading

print("--- 本地硬件网关启动程序 ---")

# --- 全局变量 ---
app = Flask(__name__)
arm_controller = None


# --- 摄像头相关 ---
def gen_frames():
    """摄像头视频流生成器"""
    while True:
        if arm_controller is None:
            time.sleep(1)
            continue

        frame = arm_controller.capture_image()
        if frame is None:
            time.sleep(0.1)
            continue

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """摄像头直播的网址"""

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# --- 机械臂相关 ---
@app.route('/click', methods=['POST'])
def click_at_coordinate():
    """机械臂点击的网址"""
    if arm_controller is None:
        return jsonify({"status": "error", "message": "Hardware not ready."}), 503

    data = request.get_json()
    x = data.get('x')
    y = data.get('y')
    if x is None or y is None:
        return jsonify({"status": "error", "message": "Missing coordinates."}), 400

    print(f"收到网络指令: 点击物理坐标 ({x}, {y})")
    try:
        # 直接使用全局的、已经连接好的控制器实例
        arm_controller.move_and_click(float(x), float(y))
        return jsonify({"status": "success"})
    except Exception as e:
        print(f"ERROR: 执行点击时出错: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# --- 主启动流程 ---
def initialize_hardware():
    """
    这是核心初始化函数，它会在一个独立的线程中运行，
    以避免阻塞 Flask 主服务。
    """
    global arm_controller
    print("\n[硬件线程]: 正在初始化硬件...")
    try:
        config = motion_controller.Config()
        arm_controller = motion_controller.MotionController(config)
        print("[硬件线程]: 硬件已就绪！网关现在可以接收指令。")
    except Exception as e:
        print(f"[硬件线程]: CRITICAL! 硬件初始化失败: {e}")


if __name__ == '__main__':
    # 1. 启动一个独立的后台线程来处理耗时且可能失败的硬件初始化
    init_thread = threading.Thread(target=initialize_hardware, daemon=True)
    init_thread.start()

    # 2. Flask Web 服务器立刻在前台启动，不会被硬件初始化卡住
    print("\n--- 本地硬件网关已启动 ---")
    print("硬件正在后台初始化，请稍候...")
    print("您可以随时通过下面的地址访问服务：")
    print("  - 摄像头: http://<您的IP>:5000/video_feed")
    print("  - 机械臂: POST http://<您的IP>:5001/click")
    print("--------------------------\n")


    def run_camera_app():
        app.run(host='0.0.0.0', port=5000)


    def run_robot_app():
        app.run(host='0.0.0.0', port=5001)


    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)