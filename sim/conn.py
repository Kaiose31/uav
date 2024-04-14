from .rotor import Rotor
import os

SERVER_IP = os.getenv("AIRSIMHOST", "34.68.92.182")
PORT = 41451

client = Rotor(SERVER_IP, PORT, timeout_value=10)
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
