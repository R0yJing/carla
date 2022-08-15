
from environment import CarEnv
import tensorflow as tf
import sys
sys.path.append(r"C:\Users\autpucv\Documents\coiltraine-master\coiltraine-master")
from coiltraine_agent import *
agent = Agent()
import time
env = CarEnv([], [], training=False)
env.reset()
while True:
    if env.current_direction == 5:
        env.current_direction = 2
    control = agent.get_action(env.front_camera, env.get_speed(), env.current_direction)
    obs, done = env.run_step(control)
   
    if done:
        env.reset()
    time.sleep(0.1)