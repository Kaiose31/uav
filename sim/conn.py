from .rotor import Rotor
import os

SERVER_IP = os.getenv("AIRSIMHOST", "127.0.0.1")
SERVER_IP = os.getenv("AIRSIMHOST","34.42.3.158")
PORT = 41451

client = Rotor(SERVER_IP, PORT, timeout_value=10)
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
