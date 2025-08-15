import requests

# 1. 定义远程硬件网关的公网地址 (由Ngrok提供)
REMOTE_GATEWAY_URL = "your url"

# 2. AI大脑经过思考，决策出下一步的目标物理坐标
target_physical_coords = {"x": 55.8, "y": -30.2}

# 3. 将决策打包成JSON，并通过HTTP POST请求发送
try:
    print(f"正在向 {REMOTE_GATEWAY_URL}/click 发送指令...")

    response = requests.post(
        url=REMOTE_GATEWAY_URL + "/click",
        json=target_physical_coords,
        headers={'ngrok-skip-browser-warning': 'true'},
        timeout=5
    )

    # 4. 检查远程硬件的执行回执
    if response.status_code == 200:
        print("指令成功，远程硬件已确认执行。")
    else:
        print(f"指令失败，硬件网关返回错误码: {response.status_code}")

except requests.exceptions.RequestException as e:
    print(f"网络请求失败: {e}")