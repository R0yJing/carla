import sys
sys.path.append('.')
import numpy as np
from .ddpg_tf2 import Agent
from environment import CarEnv
from carla import VehicleControl
#parent directory
import os
from .utils import *
from constants import * 
from time import time
def main(evaluate=False):
    
    
    env = CarEnv([0,0,0], [0], skip_turn_samples=True )
    agent = Agent(input_dims=((IM_HEIGHT, IM_WIDTH, 3), (1,), (3,)),
            n_actions=3, load_checkpoint=False)
    n_games = 249

    figure_file = 'DDPG\plots\ddpg.png'

    best_score = -201
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation, _ = env.reset1()
            action = env.sample()
            if action is not None:
                observation_, reward, done, info = env.step1(action)
                agent.remember(observation, action, reward, observation_, done)
                n_steps += 1
        agent.learn()
        agent.load_models()
        evaluate = True
    else:
        evaluate = False
    step_timer = time()
    #TODO balance the number of left, straight and right turns
    
    for i in range(n_games):
        observation, _ = env.reset1()
        #kickstart the vehicle
        env.step1(VehicleControl(steer=0, throttle=1, brake=0))
        done = False
        score = 0
        print(f"game {i}")
        max_ep_length = 6e4
        ep_length = 0

        while not done and ep_length < max_ep_length:
            if time() - step_timer < 0.2:
                continue
            else:
                step_timer = time()
            action = agent.choose_action(observation, evaluate)
        
            observation_, reward, done, info = env.step1(get_control(action))

            score += reward
            agent.remember(observation, action, reward, observation_, done)
            ep_length += 1
            observation = observation_
        print("start learning")
        if not load_checkpoint:
            agent.learn()
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
