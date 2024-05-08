import socket
import cv2
import numpy as np
from gym import Env
import struct
import matplotlib.pyplot as plt
import cv2
import time
from freewayEnv import FreewayEnv

class EnvironmentServer:
    def __init__(self, env: FreewayEnv, port=1025):
        self.env: FreewayEnv = env

        self.data_payload = 2048

        self.client_socket = None
        self.client_address = None
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('localhost', port))
        self.server_socket.listen()

    def resetAndSend(self):
        self.env.reset()

        self.sendImageToClient()

    def play(self):
        for i in range(2):
            self.resetAndSend()
        while True:
            self.resetAndSend()

            while not self.env.terminated:
                self.performNextAction()
                self.sendImageToClient()
            self.performNextAction(step=False)
            



    def performNextAction(self, step=True):
        self.client_socket, self.client_address = self.server_socket.accept()
        #print("Accepted action")
        action_data = int.from_bytes(self.client_socket.recv((self.data_payload)), "big")
        #print(f"Received action - {action_data}")
        self.client_socket.close()
        #print("Closed action")

        if step:
            self.env.step(action_data)

    def sendImageToClient(self):
        # checks if the observation is a valid image
        if len(self.env.observation) != 210:
            return

        obs_array = np.asarray(self.env.observation, dtype="uint8")

        #freeway
        obs_array = obs_array[14:196, 8:160]

        #print(obs_array.shape)

        rbg_observation = cv2.cvtColor(obs_array, cv2.COLOR_RGB2BGR)
        rbg_observation = cv2.resize(rbg_observation, dsize=(0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

        # cv2.imwrite("images/cropped_img.tiff", rbg_observation)
        img_encode = cv2.imencode('.tiff', rbg_observation)[1]
        data_encode = np.array(img_encode)
        image_byte_encode = data_encode.tobytes()

        self.client_socket, self.client_address = self.server_socket.accept()
        self.client_socket.send(image_byte_encode)
        self.client_socket.close()
        #print("Sent observation")
        self.client_socket, self.client_address = self.server_socket.accept()
        self.client_socket.send(str(self.env.reward).encode())
        self.client_socket.close()
        #print(f"Sent reward - {self.env.reward}")
        self.client_socket, self.client_address = self.server_socket.accept()
        self.client_socket.send(struct.pack("?", self.env.terminated))
        self.client_socket.close()
        #print(f"Sent terminated - {self.env.terminated}")

        if self.env.terminated:
            print("Episode ended")

        #print("Closed percept")

    def close_sockets(self):
        if not self.client_socket is None:
            self.client_socket.close()
        self.server_socket.close()
