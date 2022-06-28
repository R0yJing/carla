import math
from constants import IM_HEIGHT, IM_WIDTH
from environment import CarEnv
from neural_net import agent
import carla 
import random
import numpy as np
import time

agt = agent(simulating=True)
env = CarEnv(port=6000, training=True)
env.reset()
while True:
    s,t,b = agt.get_single_action(env.front_camera, env.get_speed(), "straight")
    control = carla.VehicleControl(steer=s, throttle=t, brake=b)
    env.run_step(control)
    time.sleep(0.5)
