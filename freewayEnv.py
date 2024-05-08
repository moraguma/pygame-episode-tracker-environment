import numpy as np
from gym import Env
import cv2
import time


class FreewayEnv:
    def __init__(self, env: Env):
        self.env = env
        self.frames_per_action = 2

        self.observation = None
        self.reward = None
        self.terminated = False

        self.lose_reward = -1
        self.step_report = 100
        self.max_steps = 500
        self.total_steps = 0

        self.min_initial_steps = 0
        self.max_initial_steps = 100

        self.current_time = 0

        self.chicken_pattern = cv2.cvtColor(cv2.imread("/home/moraguma/git/pygame-episode-tracker-environment/freeway_images/chicken.png"), cv2.COLOR_BGR2RGB)
        w, h = self.chicken_pattern.shape[1], self.chicken_pattern.shape[0]
        self.chicken_w, self.chicken_h = w, h
        self.REWARD_PER_PIXEL = -0.0005
        self.WIN_REWARD = 1
    

    def reset(self):
        self.total_steps = 0
        self.current_time = time.time()

        self.observation, _ = self.env.reset()

        pre_steps = np.random.randint(self.min_initial_steps, self.max_initial_steps)
        for i in range(pre_steps):
            self.observation, self.reward, self.terminated, _, _ = self.env.step(0)
        self.reward = self.get_custom_reward(self.observation)
    

    def step(self, action_data):
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
            print(f"Timeout in {time.time() - self.current_time}s")
        elif self.total_steps % self.step_report == 0:
            print(f"{self.total_steps} / {self.max_steps} steps")


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