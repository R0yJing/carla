from ModifiedTensorBoard import ModifiedTensorBoard
import numpy as np
import cv2
from collections import deque
from keras import models
import os
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from behavioral_cloning.constants import MAX_REPLAY_BUFFER_SIZE
from behavioral_cloning.modules import main_module
from tensorflow.keras.optimizers import Adam
from keras.applications.xception import Xception
from expert import expert_policy


class Agent():
    def __init__(self):
        self.policy = main_module()
        self.replay_buffer = deque()
        self.reached_max = False

    def update_replay_memory(self, trajectory):
        if self.reached_max or len(self.replay_buffer) == MAX_REPLAY_BUFFER_SIZE:
            self.replay_buffer.popleft()
            self.reached_max = True
    
        self.replay_buffer.append(trajectory)
    #note observation is replaced by the location of the vehicle instead as the PID controller is
    #not based on pictorial data

    
    def get_action(self, ob, cmd):
        branch_index = 1
        if cmd == "left":
            branch_index = 0
        elif cmd == "right":
            branch_index = 2

        return self.policy.predict(ob, cmd)