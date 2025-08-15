# run_agent_physical.py

import json
import os
import time
import requests
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import cv2

# === MODIFIED VOICE RECOGNIZER  ===
import sounddevice as sd
import numpy as np
import torch

try:
    import whisper

    VOICE_ENABLED = True
except ImportError:
    print("WARNING: 'whisper' 库未安装，语音输入功能将不可用。")
    print("请运行: pip install openai-whisper")
    VOICE_ENABLED = False

try:
    # 检查是否有可用的麦克风
    sd.query_devices()
except Exception as e:
    print(f"WARNING: sounddevice 库无法正常工作: {e}")
    print("语音输入功能可能不可用。请确保已安装 sounddevice 并且有可用的麦克风。")
    print("Linux 用户可能需要安装: sudo apt-get install libportaudio2")
    VOICE_ENABLED = False


class VoiceRecognizer:
    def __init__(self, model_size="base"):
        if not VOICE_ENABLED:
            raise ImportError("语音识别所需库未安装或麦克风不可用，无法初始化。")
        print(f"  - 正在加载 Whisper 模型 ({model_size})...")
        try:
            self.model = whisper.load_model(model_size)
            print("  - Whisper 模型加载成功！")
        except Exception as e:
            print(f"CRITICAL: 加载Whisper模型失败: {e}")
            raise

    def recognize_speech(self, audio_data):
        print("  - Whisper 正在识别录制的语音...")
        # 将音频数据展平为一维数组
        audio_data = audio_data.flatten().astype(np.float32)
        result = self.model.transcribe(audio_data, fp16=torch.cuda.is_available())
        recognized_text = result["text"].strip()
        print(f"  - 识别结果: '{recognized_text}'")
        return recognized_text

    def get_instruction_from_voice(self):
        """
        直接从默认麦克风录制音频，并进行识别。
        """
        SAMPLE_RATE = 16000  # Whisper 期望的采样率
        DURATION = 6  # 录音时长（秒）可以根据需要调整

        print("\n--- 语音输入模式 ---")
        input(f"  - >>> 请按 Enter 键开始录音（将录制 {DURATION} 秒）...")

        print(f"  - 开始录音，请说话...")

        # 录制音频
        audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()  # 等待录音完成

        print("  - 录音结束。")

        # 调用识别函数
        instruction = self.recognize_speech(audio_data=audio_data)
        return instruction




# === PART 1: 全局配置区域 ===
CAMERA_NGROK_URL = "https://03f9a4bd524e.ngrok-free.app"
ROBOT_ARM_NGROK_URL = "https://03f9a4bd524e.ngrok-free.app"
MODEL_PATH = "model/AgentCPM-GUI"
NGROK_VIP_HEADERS = {'ngrok-skip-browser-warning': 'true'}


# === PART 2: 远程硬件的 Python 接口  ===
def get_image_from_camera_stream():
    full_url = CAMERA_NGROK_URL + "/video_feed"
    try:
        response = requests.get(full_url, stream=True, timeout=10, headers=NGROK_VIP_HEADERS)
        if response.status_code == 200:
            bytes_data = b''
            for chunk in response.iter_content(chunk_size=1024):
                bytes_data += chunk
                a = bytes_data.find(b'\xff\xd8')
                b = bytes_data.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b + 2]
                    bytes_data = bytes_data[b + 2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return None
    except requests.exceptions.RequestException as e:
        return None


def command_robot_arm_click(x, y):
    full_url = ROBOT_ARM_NGROK_URL + "/click"
    try:
        payload = {"x": float(x), "y": float(y)}
        response = requests.post(full_url, json=payload, timeout=10, headers=NGROK_VIP_HEADERS)
        return response.status_code == 200 and response.json().get('status') == 'success'
    except requests.exceptions.RequestException as e:
        return False


def command_robot_arm_move(x, y):
    return command_robot_arm_click(x, y)


# === PART 3: 坐标系转换的核心逻辑  ===
def convert_px_to_mm_relative(target_px, current_robot_pos_mm, camera_resolution, calibration_data):
    mm_per_pixel = calibration_data['mm_per_pixel']
    offset_px = tuple(calibration_data['offset_px'])
    cam_w, cam_h = camera_resolution
    camera_center_px = (cam_w / 2, cam_h / 2)
    clicker_pos_px = (camera_center_px[0] + offset_px[0], camera_center_px[1] + offset_px[1])
    move_vector_px = (target_px[0] - clicker_pos_px[0], target_px[1] - clicker_pos_px[1])
    move_mm_x = move_vector_px[0] * mm_per_pixel
    move_mm_y = move_vector_px[1] * mm_per_pixel
    target_robot_mm = (current_robot_pos_mm[0] + move_mm_x, current_robot_pos_mm[1] + move_mm_y)
    print(
        f"  - [相对坐标转换] 目标像素:{target_px}, 当前物理位置:{current_robot_pos_mm} -> 最终目标物理坐标:{target_robot_mm}")
    return target_robot_mm


# ======================== PART 4: 主执行流程  ===============================
def run_main_agent_task():
    recognizer = None
    try:
        # --- 步骤 1: 初始化 ---
        print("\n--- 步骤 1: 正在初始化所有模块 ---")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        ai_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                                        torch_dtype=torch.bfloat16).to("cuda:0")
        print("  - AI 模型加载完成！")
        print("  - 正在加载手眼标定配置...")
        if not os.path.exists('final_config.json'):
            raise FileNotFoundError("CRITICAL: 未找到手眼标定文件 'final_config.json'！")
        with open('final_config.json', 'r') as f:
            calibration_data = json.load(f)
        if 'offset_px' not in calibration_data:
            raise ValueError("CRITICAL: 标定文件格式不正确！这个脚本需要包含 'offset_px' 的旧版配置文件。")
        print(f"  - 标定数据加载成功: {calibration_data}")

        # --- 选择指令输入方式 ---
        instruction = ""
        while True:
            print("\n--- 请选择指令输入方式 ---")
            print("  1. 手动输入文本指令")
            if VOICE_ENABLED:
                print("  2. 语音输入指令")
            choice = input("请输入选项 (1 或 2): ")
            if choice == '1':
                instruction = input("请输入你的任务指令: ")
                break
            elif choice == '2' and VOICE_ENABLED:
                if recognizer is None: recognizer = VoiceRecognizer()
                instruction = recognizer.get_instruction_from_voice()
                break
            else:
                print("无效的输入，请重试。")
        if not instruction:
            print("\n未获取到有效指令，程序退出。")
            return

        # --- 步骤 2: 定义任务并开始循环 ---
        MAX_STEPS = 10
        print(f"\n--- 任务开始 ---")
        print(f"最终执行的指令是: \"{instruction}\"")
        print("--------------------")

        for i in range(MAX_STEPS):
            print(f"\n--- 第 {i + 1} 步 ---")
            print("  - [归位] 正在移动到原点(0,0)以获得全局视野...")
            if not command_robot_arm_move(0, 0):
                print("  - [失败] 无法移动到原点，任务终止。")
                break
            time.sleep(2)
            print("  - [观察] 正在从原点位置进行观察...")
            image = get_image_from_camera_stream()
            if image is None:
                print("  - [失败] 观察失败，跳过此步。")
                continue
            camera_resolution = image.size
            prompt = f"""
你是一个能够控制机械臂的智能代理。你的任务是根据我的【总指令】和【当前屏幕截图】，一步一步地完成任务。
你必须以严格的 JSON 格式进行响应。
---
**总指令**: <Question>{instruction}</Question>
---
**你的思考过程**:
1.  **观察**: 描述一下你当前在屏幕上看到了什么关键元素。
2.  **判断**: 对比你观察到的，和总指令的要求，判断任务是否已经【完全】完成。
3.  **决策**:
    *   如果任务【未完成】，请给出下一步要点击的坐标: {{"POINT": [x, y], "REASON": "简要说明你为什么要点这里"}}
    *   只有当总指令中的【所有要求】都已满足时，才能返回: {{"STATUS": "finish", "REASON": "简要说明任务是如何完成的"}}
---
**当前屏幕截图**：
"""
            messages = [{"role": "user", "content": [prompt, image]}]
            outputs_str = ai_model.chat(image=None, msgs=messages, tokenizer=tokenizer)
            try:
                json_part = outputs_str[outputs_str.find('{'):outputs_str.rfind('}') + 1]
                action = json.loads(json_part)
                print(f"  - AI 决策: {action}")
            except (json.JSONDecodeError, IndexError):
                print(f"  - WARNING: AI 输出不是有效的 JSON: {outputs_str}，跳过此步。")
                continue
            if "POINT" in action:
                rel_x, rel_y = action["POINT"]
                target_camera_px = (int(rel_x / 1000 * camera_resolution[0]), int(rel_y / 1000 * camera_resolution[1]))
                target_robot_mm = convert_px_to_mm_relative(target_camera_px, [0.0, 0.0], camera_resolution,
                                                            calibration_data)
                print(f"  - [行动] 正在从原点移动到目标点并点击...")
                if not command_robot_arm_click(target_robot_mm[0], target_robot_mm[1]):
                    print("  - [失败] 机械臂点击指令执行失败，任务终止。")
                    break
            elif "STATUS" in action and action["STATUS"] == "finish":
                print("  - [完成] AI 认为任务已全部完成，流程结束。")
                break
            else:
                print(f"  - [未知/未发现] AI未给出有效点击点: {action}。将在原点重新观察。")
            print("  - 等待 3 秒，让操作生效...")
            time.sleep(3)
        print("\n--- 任务流程已结束 ---")
    except Exception as e:
        print(f"\nCRITICAL: 主代理程序运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n--- 正在关闭所有设备 ---")
        try:
            command_robot_arm_move(0, 0)
        except Exception:
            pass
        print("--- 程序已完全结束 ---")


if __name__ == "__main__":
    run_main_agent_task()
