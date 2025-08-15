# final_calibrate.py (The Simplest Correct Version)
import cv2
import motion_controller
import time
import json
import numpy as np

# --- 全局变量和鼠标回调 ---
mouse_click_pos = None


def mouse_callback(event, x, y, flags, param):
    global mouse_click_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_click_pos = (x, y)
        print(f"  - 记录到鼠标点击: {mouse_click_pos}")


def get_user_click(arm, prompt_text):
    global mouse_click_pos
    mouse_click_pos = None
    print(f"\n>>> {prompt_text}")
    while mouse_click_pos is None:
        frame = arm.capture_image()
        if frame is None:
            time.sleep(0.1)
            continue

        cv2.putText(frame, "Please LEFT-CLICK on the center of the cross mark.",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Final Calibration", frame)

        if cv2.waitKey(100) == 27:
            raise KeyboardInterrupt("用户中断了标定。")
    return mouse_click_pos


def run_final_calibration():
    """主标定流程"""
    arm = None
    try:
        # --- 1. 初始化 ---
        config = motion_controller.Config()
        arm = motion_controller.MotionController(config)
        cv2.namedWindow("Final Calibration")
        cv2.setMouseCallback("Final Calibration", mouse_callback)

        print("\n--- 最终版手眼标定程序 ---")

        # --- 步骤1: 标定 mm/pixel ---
        print("\n【步骤1】标定像素-毫米转换比。")
        input("  - 请将白纸上的十字标记移动到摄像头视野内，然后按回车键继续...")

        # 获取第一个点
        p1 = get_user_click(arm, "首先，请在窗口中点击【十字的中心】。")

        # 移动机械臂
        known_move_dist = 50.0
        print(f"  - 机械臂将移动 {known_move_dist}mm。")
        arm.move_to(known_move_dist, 0)
        time.sleep(2)

        # 获取第二个点
        p2 = get_user_click(arm, "机械臂已移动。请再次在窗口中点击【十字的中心】。")

        # 计算转换比
        pixel_dist = np.linalg.norm(np.array(p1) - np.array(p2))
        if pixel_dist < 5:
            raise ValueError("像素位移过小(<5)，无法计算。")
        mm_per_pixel = known_move_dist / pixel_dist
        print(f"  - 计算出的转换比: {mm_per_pixel:.4f} mm/pixel")

        # 标定完比例后，让机械臂回到原点
        arm.move_to(0, 0)
        time.sleep(2)

        # --- 步骤2: 标定偏移量 ---
        print("\n【步骤2】标定摄像头中心与点击头之间的偏移。")
        print("  - 请通过发送指令的方式（或者用手轻轻拨动工作台上的纸），将【黄色点击头】精确地对准十字中心。")
        input("  - 对准后按回车键...")

        # 获取点击头在摄像头中的位置
        clicker_pos_in_camera = get_user_click(arm, "对准已完成。请在窗口中点击【十字的中心】（也就是点击头的位置）。")

        # 获取摄像头中心
        # vvvvvvvvvvvv 这是修复你问题的关键 vvvvvvvvvvvv
        # 我们需要在【同一帧】画面上，既知道点击头的位置，也知道画面的中心
        frame_for_size = arm.capture_image()
        h, w, _ = frame_for_size.shape
        camera_center_px = (w // 2, h // 2)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        # 计算像素偏移
        offset_px = (clicker_pos_in_camera[0] - camera_center_px[0],
                     clicker_pos_in_camera[1] - camera_center_px[1])

        print(f"  - 计算出的像素偏移 (点击头 - 摄像头中心): {offset_px}")

        # --- 保存 ---
        calibration_data = {"mm_per_pixel": mm_per_pixel, "offset_px": offset_px}
        with open('final_config.json', 'w') as f:
            json.dump(calibration_data, f, indent=4)
        print("\n标定成功！配置已保存到 final_config.json")

    except (KeyboardInterrupt, ValueError) as e:
        print(f"\n操作被中断或出错: {e}")
    finally:
        if arm: arm.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    run_final_calibration()