from freewayEnv import FreewayEnv
import gym
import requests
import json
import numpy as np
from graphCreator import Grapher


def step_request(observation, reward, terminated, eval=False):
    step_info = {
        "observation": observation.tolist(),
        "reward": reward,
        "terminal": terminated
    }

    if eval:
        return requests.post("http://localhost:5000/eval", json=step_info)
    return requests.post("http://localhost:5000/step", json=step_info)


def run_episode(env: FreewayEnv, eval=False):
    """
    Runs and episode from beginning to end and returns the total reward obatined
    """

    # Initialize episode
    env.reset()

    cummulative_reward = 0.0

    total_steps = 0
    while not env.terminated:
        result = step_request(env.observation, env.reward, env.terminated, eval)
        result_dict = json.loads(result.text)
        actions = int(result_dict["action"])
        env.step(actions)

        cummulative_reward += env.reward
        total_steps += 1
    
    if not eval:
        step_request(env.observation, env.reward, env.terminated)

    return cummulative_reward


TRAINING_PARAMETERS = {
    "observation": {
        "type": "float32",
        "preset": "freeway",    
    },
    "action": {
        "type": "int64",
        "shape": [],
        "mins": 0,
        "maxs": 2
    },
    "network": [
        {
            "type": "conv2d",
            "filter_count": 32,
            "kernel_size_x": 8,
            "kernel_size_y": 8,
            "padding": "same",
            "activation": "relu"
        },{
            "type": "maxpooling2d",
            "pool_size_x": 8,
            "pool_size_y": 8,
            "strides": 4
        },{
            "type": "conv2d",
            "filter_count": 64,
            "kernel_size_x": 4,
            "kernel_size_y": 4,
            "padding": "same",
            "activation": "relu"
        },{
            "type": "maxpooling2d",
            "pool_size_x": 4,
            "pool_size_y": 4,
            "strides": 2
        },{
            "type": "conv2d",
            "filter_count": 64,
            "kernel_size_x": 3,
            "kernel_size_y": 3,
            "padding": "same",
            "activation": "relu"
        },{
            "type": "maxpooling2d",
            "pool_size_x": 3,
            "pool_size_y": 3,
            "strides": 1
        },{
            "type": "flatten"
        },{
            "type": "dense",
            "units": 100,
            "activation": "relu"
        },{
            "type": "dense",
            "units": 50,
            "activation": "relu"
        }
    ],
    "optimizer": {
        "type": "adam",
        "learning_rate": 0.001
    },
    "replay_buffer": {
        "max_size": 100000
    },
    "discount": 0.9,
    "batch_size": 64,
    "initial_collect_steps": 100
}

EPISODES = 200
EVAL_INTERVAL = 3
EVAL_EPISODES = 1

if __name__ == '__main__':
    train_env = FreewayEnv(gym.make("ALE/Freeway-v5", render_mode="rgb_array"))
    train_env.reset()
    eval_env = FreewayEnv(gym.make("ALE/Freeway-v5", render_mode="rgb_array"))
    eval_env.reset()

    # Initialize DQNLearningServer
    info = requests.post("http://localhost:5000/initialize", json=TRAINING_PARAMETERS)

    data = np.zeros((int(EPISODES / EVAL_INTERVAL), 2))

    # Train for EPISODES
    for i in range(EPISODES):
        run_episode(train_env)

        if i % EVAL_INTERVAL == 0:
            avg_reward = 0.0
            for j in range(EVAL_EPISODES):
                avg_reward += run_episode(eval_env, True)
            avg_reward /= EVAL_EPISODES

            print(f"Finished episode {i} - Avg reward {avg_reward}")
            data[int(i / EVAL_INTERVAL), :] = [int(i / EVAL_INTERVAL), avg_reward]

    train_env.close()
    eval_env.close()

    Grapher.create("FreewayTF", 10, data)