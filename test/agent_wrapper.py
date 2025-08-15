import abc
import base64
import io
import os
import time
from typing import Any, Optional
import google.generativeai as genai
from google.generativeai import types
from google.generativeai.types import answer_types
from google.generativeai.types import content_types
from google.generativeai.types import generation_types
from google.generativeai.types import safety_types
import numpy as np
from PIL import Image
import requests
import json
from jsonschema import Draft7Validator

ERROR_CALLING_LLM = "Error calling LLM"
END_POINT = "http://localhost:8000/v1/chat/completions"

# 鑾峰彇褰撳墠鏂囦欢鐨勭粷瀵硅矾寰�
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)


def compact_json_dumps(obj):
    return json.dumps(obj, indent=None, separators=(",", ":"), ensure_ascii=False)


ACTION_SCHEMA = json.load(
    open(os.path.join(current_dir, "schema_thought.json"), encoding="utf-8")
)
items = list(ACTION_SCHEMA.items())
insert_index = 3  # 鍋囪瑕佹彃鍏ュ埌绱㈠紩1鐨勪綅缃�
items.insert(insert_index, ("required", ["thought"]))
# items.insert(insert_index, ("optional", ["thought"]))
ACTION_SCHEMA = dict(items)
SYSTEM_PROMPT = f"""# Role
浣犳槸涓€鍚嶇啛鎮夊畨鍗撶郴缁熻Е灞廏UI鎿嶄綔鐨勬櫤鑳戒綋锛屽皢鏍规嵁鐢ㄦ埛鐨勯棶棰橈紝鍒嗘瀽褰撳墠鐣岄潰鐨凣UI鍏冪礌鍜屽竷灞€锛岀敓鎴愮浉搴旂殑鎿嶄綔銆�

# Task
閽堝鐢ㄦ埛闂锛屾牴鎹緭鍏ョ殑褰撳墠灞忓箷鎴浘锛岃緭鍑轰笅涓€姝ョ殑鎿嶄綔銆�

# Rule
- 浠ョ揣鍑慗SON鏍煎紡杈撳嚭
- 杈撳嚭鎿嶄綔蹇呴』閬靛惊Schema绾︽潫

# Schema
{json.dumps(ACTION_SCHEMA, indent=None, ensure_ascii=False, separators=(',', ':'))}"""

EXTRACT_SCHEMA = json.load(
    open(os.path.join(current_dir, "schema_for_extraction.json"), encoding="utf-8")
)
validator = Draft7Validator(EXTRACT_SCHEMA)


def array_to_jpeg_bytes(image: np.ndarray) -> bytes:
    """Converts a numpy array into a byte string for a JPEG image."""
    image = Image.fromarray(image)
    return image_to_jpeg_bytes(image)


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
    in_mem_file = io.BytesIO()
    image.save(in_mem_file, format="PNG")
    # Reset file pointer to start
    in_mem_file.seek(0)
    img_bytes = in_mem_file.read()
    return img_bytes


class LlmWrapper(abc.ABC):
    """Abstract interface for (text only) LLM."""

    @abc.abstractmethod
    def predict(
        self,
        text_prompt: str,
    ) -> tuple[str, Optional[bool], Any]:
        """Calling multimodal LLM with a prompt and a list of images.

        Args:
          text_prompt: Text prompt.

        Returns:
          Text output, is_safe, and raw output.
        """


class MultimodalLlmWrapper(abc.ABC):
    """Abstract interface for Multimodal LLM."""

    @abc.abstractmethod
    def predict_mm(
        self, text_prompt: str, images: list[np.ndarray]
    ) -> tuple[str, Optional[bool], Any]:
        """Calling multimodal LLM with a prompt and a list of images.

        Args:
          text_prompt: Text prompt.
          images: List of images as numpy ndarray.

        Returns:
          Text output and raw output.
        """


SAFETY_SETTINGS_BLOCK_NONE = {
    types.HarmCategory.HARM_CATEGORY_HARASSMENT: (types.HarmBlockThreshold.BLOCK_NONE),
    types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: (types.HarmBlockThreshold.BLOCK_NONE),
    types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
}


class MiniCPMWrapper(LlmWrapper, MultimodalLlmWrapper):

    RETRY_WAITING_SECONDS = 20

    def __init__(
        self,
        model_name: str,
        max_retry: int = 3,
        temperature: float = 0.1,
        use_history: bool = False,
        history_size: int = 10,  # 鏈€澶氫繚鐣欐渶杩� history_size 杞�
    ):
        if max_retry <= 0:
            max_retry = 3
            print("Max_retry must be positive. Reset it to 3")
        self.max_retry = min(max_retry, 5)
        self.temperature = temperature
        self.model = model_name

        # ---------- 鏂板 ----------
        self.use_history  = use_history
        self.history_size = max(history_size, 1)
        # history 浠ャ€屽崟鏉℃秷鎭€嶄负绮掑害锛� [{'role': .., 'content': ..}, ...]
        self.history: list[dict] = []

    @classmethod
    def encode_image(cls, image: np.ndarray) -> str:
        return base64.b64encode(array_to_jpeg_bytes(image)).decode("utf-8")

    def _push_history(self, role: str, content: Any):
        """鎶婁竴鏉℃秷鎭啓鍏ュ巻鍙诧紝骞惰嚜鍔ㄨ鍓暱搴︺€�"""
        if not self.use_history:
            return
        self.history.append({"role": role, "content": content})
        # 姣忚疆瀵硅瘽鍖呭惈 user + assistant 涓ゆ潯娑堟伅
        max_msgs = self.history_size * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

    def clear_history(self):
        """澶栭儴鍙墜鍔ㄦ竻绌鸿蹇嗐€�"""
        self.history.clear()


    def extract_and_validate_json(self, input_string):
        try:
            json_obj = json.loads(input_string)
            validator.validate(json_obj, EXTRACT_SCHEMA)
            return json_obj
        except json.JSONDecodeError as e:
            print("Error, JSON is NOT valid.")
            return input_string
        except Exception as e:
            print(f"Error, JSON is NOT valid according to the schema.{input_string}", e)
            return input_string

    def predict(
        self,
        text_prompt: str,
    ) -> tuple[str, Optional[bool], Any]:
        return self.predict_mm(text_prompt, [])

    def predict_mm(
        self, text_prompt: str, images: list[np.ndarray]
    ) -> tuple[str, Optional[bool], Any]:
        assert len(images) == 1

        # -------- 鏋勯€� messages --------
        messages: list[dict] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            }
        ]

        # 1) 鎻掑叆鍘嗗彶
        if self.use_history and self.history:
            messages.extend(self.history)

        # 2) 褰撳墠 user 娑堟伅
        user_content = [
            {
                "type": "text",
                "text": f"<Question>{text_prompt}</Question>\n褰撳墠灞忓箷鎴浘锛�(<image>./</image>)",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.encode_image(images[0])}"
                },
            },
        ]
        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
            "max_tokens": 2048,
        }

        headers = {
            "Content-Type": "application/json",
        }

        counter = self.max_retry
        wait_seconds = self.RETRY_WAITING_SECONDS
        while counter > 0:
            try:
                response = requests.post(
                    END_POINT,
                    headers=headers,
                    json=payload,
                )
                if response.ok and "choices" in response.json():
                    assistant_msg = response.json()["choices"][0]["message"]
                    assistant_text = assistant_msg["content"]
                    action = self.extract_and_validate_json(assistant_text)

                    # -------- 鍐欏洖鍘嗗彶 --------
                    self._push_history("user",  user_content)
                    self._push_history("assistant", assistant_msg["content"])

                    return assistant_text, None, response, action
                print(
                    "Error calling OpenAI API with error message: "
                    + response.json()["error"]["message"]
                )
                time.sleep(wait_seconds)
                wait_seconds *= 2
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Want to catch all exceptions happened during LLM calls.
                time.sleep(wait_seconds)
                wait_seconds *= 2
                counter -= 1
                print("Error calling LLM, will retry soon...")
                print(e)
        return ERROR_CALLING_LLM, None, None