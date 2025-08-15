# set_camera_resolution.py
# 功能：尝试设置一个指定的分辨率，并验证摄像头是否接受了该设置。

import cv2
import time

# --- 配置区域 ---
CAMERA_INDEX = 1


DESIRED_WIDTH = 640
DESIRED_HEIGHT = 480


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def run_resolution_test():
    """主测试流程"""
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"CRITICAL ERROR: 无法打开摄像头 {CAMERA_INDEX}！")
        return

    print(f"--- 正在尝试设置分辨率为 {DESIRED_WIDTH} x {DESIRED_HEIGHT} ---")

    # --- 步骤1: 主动设置分辨率 ---
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)

    # --- 步骤2: 验证设置是否成功 ---
    # 摄像头可能不会精确接受你的设置，而是选择一个最接近的支持值。
    # 所以我们必须读取回来，看看实际生效的分辨率是多少。
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("\n--- 验证结果 ---")
    print(f"期望设置的分辨率: {DESIRED_WIDTH} x {DESIRED_HEIGHT}")
    print(f"摄像头实际生效的分辨率: {actual_width} x {actual_height}")

    if actual_width == DESIRED_WIDTH and actual_height == DESIRED_HEIGHT:
        print("SUCCESS: 分辨率设置成功！")
    else:
        print("WARNING: 分辨率设置未完全匹配。摄像头选择了最接近的支持值。")
    print("--------------------")

    print("\n正在显示实时画面... 按 'ESC' 或 'q' 键退出。")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 在画面上显示当前分辨率
        info_text = f"Actual Resolution: {actual_width} x {actual_height}"
        cv2.putText(frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(f"Resolution Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_resolution_test()