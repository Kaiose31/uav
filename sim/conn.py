from .rotor import Rotor
import os

SERVER_IP = os.getenv("AIRSIMHOST", "104.154.176.103")
PORT = 41451

client = Rotor(SERVER_IP, PORT, timeout_value=10)
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
