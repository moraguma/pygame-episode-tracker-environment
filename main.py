from time import sleep
from gym.utils.play import play
import gym
from keyMappings import KEY_MAPPING
from environmentServer import EnvironmentServer
from freewayEnv import FreewayEnv


def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    sleep(0.1)
    envServer.sendImageToClient(obs_t)
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = FreewayEnv(gym.make("ALE/Freeway-v5", render_mode="rgb_array"))
    print(env.env.action_space)
    envServer = EnvironmentServer(env)
    try:
        envServer.play()
    except KeyboardInterrupt:
        envServer.close_sockets()
        print("KeyBoardInterrupt")