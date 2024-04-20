from airsim import MultirotorClient
import os

SERVER_IP = os.getenv("AIRSIMHOST", "127.0.0.1")
PORT = 41451
client = MultirotorClient(SERVER_IP, PORT, timeout_value=10)
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
