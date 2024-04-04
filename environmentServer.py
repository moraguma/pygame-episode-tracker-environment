import socket
import cv2
import numpy as np
from gym import Env
import struct
import matplotlib.pyplot as plt

class EnvironmentServer:
    def __init__(self, env, port=1025):
        self.env: Env = env
        self.frames_per_action = 2

        self.data_payload = 2048

        self.client_socket = None
        self.client_address = None
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.observation = None
        self.reward = None
        self.terminated = False

        # https://stackoverflow.com/questions/4465959/python-errno-98-address-already-in-use
        # avoids socket time wait after closing the socket
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('localhost', port))
        self.server_socket.listen()
        self.crop = True

    def resetAndSend(self):
        self.observation, _ = self.env.reset()
        self.reward = 0.0
        self.terminated = False
        self.sendImageToClient()

    def play(self):
        for i in range(2):
            self.resetAndSend()
        while True:
            self.resetAndSend()

            while not self.terminated:
                self.performNextAction()
                self.sendImageToClient()


    def performNextAction(self):
        self.client_socket, self.client_address = self.server_socket.accept()
        print("Accepted action")
        action_data = int.from_bytes(self.client_socket.recv((self.data_payload)), "big")
        print(f"Received action - {action_data}")
        self.client_socket.close()
        print("Closed action")

        #self.observation, self.reward, self.terminated, _, _ = self.env.step(action_raw_data)
        for i in range(self.frames_per_action):
            self.observation, self.reward, self.terminated, _, _ = self.env.step(action_data)

        #plt.imshow(self.observation)
        #plt.show()

    def sendImageToClient(self):
        # checks if the observation is a valid image
        if len(self.observation) != 210:
            return

        obs_array = np.asarray(self.observation, dtype="uint8")
        #riverraid
        # obs_array = obs_array[2:163, 8:160]

        #freeway
        obs_array = obs_array[14:196, 8:160]

        print(obs_array.shape)

        rbg_observation = cv2.cvtColor(obs_array, cv2.COLOR_RGB2BGR)
        rbg_observation = cv2.resize(rbg_observation, dsize=(0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

        # cv2.imwrite("images/cropped_img.tiff", rbg_observation)
        img_encode = cv2.imencode('.tiff', rbg_observation)[1]
        data_encode = np.array(img_encode)
        image_byte_encode = data_encode.tobytes()

        self.client_socket, self.client_address = self.server_socket.accept()
        self.client_socket.send(image_byte_encode)
        self.client_socket.close()
        print("Sent observation")
        self.client_socket, self.client_address = self.server_socket.accept()
        self.client_socket.send(struct.pack("f", self.reward))
        self.client_socket.close()
        print(f"Sent reward - {self.reward}")
        self.client_socket, self.client_address = self.server_socket.accept()
        self.client_socket.send(struct.pack("?", self.terminated))
        self.client_socket.close()
        print(f"Sent terminated - {self.terminated}")

        #plt.imshow(self.observation)
        #plt.show()
        
        print("Closed percept")

    def close_sockets(self):
        if not self.client_socket is None:
            self.client_socket.close()
        self.server_socket.close()
