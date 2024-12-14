from flask import Flask
from flask import request
from freewayEnv import FreewayEnv
import gym
import json
import numpy as np
import cv2
import time
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

global elapsed_time
global env
global just_terminated

just_terminated = False

app = Flask(__name__)


@app.route("/initialize", methods=["POST"])
def initialize():
    global elapsed_time
    global env

    time_i = time.time_ns()

    env = FreewayEnv(gym.make("ALE/Freeway-v5", render_mode="rgb_array"))
    env.reset()

    elapsed_time = time.time_ns() - time_i
    return {"info": "Agent initialized"}


@app.route("/step", methods=["POST"])
def step():
    if request.method == "POST":
        global elapsed_time
        global env
        global just_terminated

        time_i = time.time_ns()

        data_dict = json.loads(request.data)

        if not just_terminated:
            env.step(data_dict["action"])
        else:
            just_terminated = False
        
        elapsed_time += time.time_ns() - time_i
        return {}


@app.route("/get", methods=["POST"])
def get():
    if request.method == "POST":
        global elapsed_time
        global env
        global just_terminated

        time_i = time.time_ns()

        result = {
            "state": encode_state(),
            "reward": env.reward,
            "terminal": env.terminated
        }

        if env.terminated:
            elapsed_time += time.time_ns() - time_i
            time_i = time.time_ns()

            print("Finished episode - Time elapsed: {:.3f}".format(float(elapsed_time) / 10**9))

            env.reset()
            just_terminated = True
        
        elapsed_time += time.time_ns() - time_i

        return result


def encode_state():
    global env

    # checks if the observation is a valid image
    if len(env.observation) != 210:
        return

    obs_array = np.asarray(env.observation, dtype="uint8")

    #freeway
    obs_array = obs_array[14:196, 8:160]

    #print(obs_array.shape)

    rbg_observation = cv2.cvtColor(obs_array, cv2.COLOR_RGB2BGR)
    rbg_observation = cv2.resize(rbg_observation, dsize=(0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

    # cv2.imwrite("images/cropped_img.tiff", rbg_observation)
    img_encode = cv2.imencode('.tiff', rbg_observation)[1]
    data_encode = np.array(img_encode)

    return data_encode.tolist()