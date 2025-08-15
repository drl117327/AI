运行步骤
1.启动机器后运行local_hardware_gateway.py（让这个终端窗口保持打开状态，不要关闭它）。
2.在本地主控PC上，打开一个新的终端窗口。
  cd 到包含 ngrok.exe 的项目文件夹路径下。
  运行 ngrok 命令，将公网流量转发 5000 端口（ngrok http 5000）
  让这个终端窗口也保持打开状态
3.Ngrok 会显示它的状态界面。找到 Forwarding 那一行，复制那段 https://... .ngrok-free.app 的公开网址。
4.打开run_agent_physical.py，把步骤三中的网址粘贴到指定位置
# 示例
CAMERA_NGROK_URL = "https://a3e5dce36f33.ngrok-free.app"
ROBOT_ARM_NGROK_URL = "https://a3e5dce36f33.ngrok-free.app"
5.保存文件后运行run_agent_physical.py，即可输入指令控制机械臂


注：hand_eye_calibrate.py用与手眼标定，即确定摄像头和点击器的位置关系，final_config.json记录下了改位置关系
      set_camera.py用于调节摄像头分辨率