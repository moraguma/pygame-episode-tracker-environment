import socket
import cv2
import numpy as np


class EnvironmentServer:
    def __init__(self, port=1025):
        self.client_socket = None
        self.client_address = None
        port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # https://stackoverflow.com/questions/4465959/python-errno-98-address-already-in-use
        # avoids socket time wait after closing the socket
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('localhost', port))
        self.server_socket.listen()
        self.crop = True

    def sendImageToClient(self, observation):
        # checks if the observation is a valid image
        if len(observation) != 210:
            return

        obs_array = np.asarray(observation, dtype="uint8")
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
        print("accept")
        self.client_socket.send(image_byte_encode)
        print("file sent")
        self.client_socket.close()

    def close_sockets(self):
        self.client_socket.close()
        self.server_socket.close()
        return
