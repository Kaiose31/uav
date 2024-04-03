import airsim
import os

SERVER_IP = os.getenv("AIRSIMHOST", "127.0.0.1")
PORT = 41451

client = airsim.MultirotorClient(SERVER_IP, PORT, timeout_value=10)
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
