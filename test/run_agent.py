# -*- coding: utf-8 -*-
import time
from adb_utils import setup_device
import logging
from agent_wrapper import MiniCPMWrapper
import numpy as np
import speech_recognition as sr
import whisper
import os
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

def get_audio_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for command...")
        audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio, language='zh-CN')
            print(f"Recognized command: {command}")
            return command
        except sr.UnknownValueError:
            print("Could not understand the audio.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None

def run_task(query):
    device = setup_device()
    minicpm = MiniCPMWrapper(model_name='AgentCPM-GUI', temperature=1, use_history=True, history_size=2)

    is_finish = False
    while not is_finish:
        text_prompt = query
        screenshot = device.screenshot(1120)
        response = minicpm.predict_mm(text_prompt, [np.array(screenshot)])
        action = response[3]
        print(action)
        is_finish = device.step(action)
        time.sleep(2.5)
    return is_finish





if __name__ == "__main__":
    command = get_audio_input()
    if command:
        run_task(command)
