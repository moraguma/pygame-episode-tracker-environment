import socket
import cv2
import numpy as np
from gym import Env
import struct
import matplotlib.pyplot as plt
import cv2

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

        self.lose_reward = -1
        self.max_steps = 1000
        self.total_steps = 0

        self.min_initial_steps = 0
        self.max_initial_steps = 100

        self.chicken_pattern = cv2.cvtColor(cv2.imread("/home/moraguma/git/pygame-episode-tracker-environment/freeway_images/chicken.png"), cv2.COLOR_BGR2RGB)
        w, h = self.chicken_pattern.shape[1], self.chicken_pattern.shape[0]
        self.chicken_w, self.chicken_h = w, h
        self.REWARD_PER_PIXEL = -0.0005
        self.WIN_REWARD = 10

        # https://stackoverflow.com/questions/4465959/python-errno-98-address-already-in-use
        # avoids socket time wait after closing the socket
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('localhost', port))
        self.server_socket.listen()
        self.crop = True

    def resetAndSend(self):
        self.total_steps = 0

        self.observation, _ = self.env.reset()

        pre_steps = np.random.randint(self.min_initial_steps, self.max_initial_steps)
        for i in range(pre_steps):
            self.observation, self.reward, self.terminated, _, _ = self.env.step(0)
        self.reward = self.get_custom_reward(self.observation)

        #self.save_observation(self.observation)

        self.sendImageToClient()

    def play(self):
        for i in range(2):
            self.resetAndSend()
        while True:
            self.resetAndSend()

            while not self.terminated:
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
            i = 0
            self.reward = 0.0
            while i < self.frames_per_action and self.reward == 0.0:
                self.observation, self.reward, self.terminated, _, _ = self.env.step(action_data)
                i += 1
            
            self.total_steps += 1
            if self.reward != 0:
                self.reward = self.WIN_REWARD
                print("Won!")
            else:
                self.reward = self.get_custom_reward(self.observation)

            if self.total_steps >= self.max_steps:
                self.reward = self.lose_reward
                self.terminated = True
                print("Timeout!")

        #self.save_observation(self.observation)

    def sendImageToClient(self):
        # checks if the observation is a valid image
        if len(self.observation) != 210:
            return

        obs_array = np.asarray(self.observation, dtype="uint8")

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
        self.client_socket.send(str(self.reward).encode())
        self.client_socket.close()
        #print(f"Sent reward - {self.reward}")
        self.client_socket, self.client_address = self.server_socket.accept()
        self.client_socket.send(struct.pack("?", self.terminated))
        self.client_socket.close()
        #print(f"Sent terminated - {self.terminated}")

        if self.terminated:
            print("Episode ended")

        #print("Closed percept")

    def close_sockets(self):
        if not self.client_socket is None:
            self.client_socket.close()
        self.server_socket.close()


    def get_custom_reward(self, obs):
        img = obs[:, 43:51, :].copy()

        res = cv2.matchTemplate(img, self.chicken_pattern, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        return max_loc[1] * self.REWARD_PER_PIXEL


    def save_observation(self, obs):
        img = obs[:, 43:51, :].copy()

        res = cv2.matchTemplate(img, self.chicken_pattern, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + self.chicken_w, top_left[1] + self.chicken_h)
        print(top_left)

        cv2.putText(img, str(bottom_right), top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        cv2.imwrite(f"/home/moraguma/git/pygame-episode-tracker-environment/test/frame{self.total_steps}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR)) 
