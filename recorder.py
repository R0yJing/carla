SAMPLE_TIME = 0.5

import pickle
import time
from constants import AUGMENTATION_BATCH_SIZE, MAX_BRANCH_BUFFER_SIZE, MAX_REPLAY_BUFFER_SIZE, MAX_TEST_BRANCH_BUFFER_SIZE, MAX_TEST_DATA_SIZE, NOISE_DURATION
import random
import argparse
from expert import Expert
from environment import CarEnv
from collections import deque
import math
import atexit
from pympler import asizeof
from image_augmenter import image_augmenter

#subject to change
AVERAGE_ANGLE_FOR_SHARP_TURN = math.radians(5)

def calculate_time_before_noise(perc_noise, total_time, noise_duration):
    noise_amt = (total_time * perc_noise) / noise_duration
    #total_tim
    return total_time / noise_amt - noise_duration
class recorder:
    def __init__(self, collect_training=True):
        self.collect_training_data = collect_training
        self.noise_timer = None
        self.start_timer = time.time()
        self.noise_steer = None
        self.turn_timer = None
        self.recording = None
        self.left_turns = 0 #14303
        self.right_turns = 0 #16661
        self.straight = 0#99999999
        self.temp_img_buffer = []
        self.augmenter = image_augmenter()
        self.timer = time.time()
    def add_to_buffer(self,trajectory_point):
        
        if len(self.temp_img_buffer) == AUGMENTATION_BATCH_SIZE:
            imgs = self.augmenter.aug([path[0] for path in self.temp_img_buffer])
            imgs = [img/255 for img in imgs]
            
            for i in range(AUGMENTATION_BATCH_SIZE):
                self.temp_img_buffer[i][0] = imgs[i]
                self.expert_trajectory.append(self.temp_img_buffer[i])
            self.temp_img_buffer = []
        else:
            self.temp_img_buffer.append(trajectory_point)
    def record(self, env : CarEnv):
        if not self.collect_training_data:
            global MAX_BRANCH_BUFFER_SIZE
            MAX_BRANCH_BUFFER_SIZE = MAX_TEST_BRANCH_BUFFER_SIZE
        self.expert_trajectory = []
        #env.reset()
        #observation_records = deque(maxlen=MAX_REPLAY_BUFFER_SIZE)
        record_start = time.time()
        last_sample_time = time.time()
        expert = Expert(env)
    
        print("recording started...")
        self.timer = time.time()
        TOTAL_TIMESPAN = 36000
        time_before_noise = calculate_time_before_noise(0.1, TOTAL_TIMESPAN, NOISE_DURATION)
            
        while True:
            
            control = expert.get_action_pid()
            #frequency of noise = 10% 
            if env.training:
                if time.time() - self.start_timer > time_before_noise:
                    
                    self.start_timer = time.time() + NOISE_DURATION
                    self.noise_timer = time.time()

                if self.noise_timer is not None:
                    time_elapsed = time.time() - self.noise_timer
                    if time_elapsed < 1.5:
                        
                        if self.noise_steer is None:
                           
                            env.sensor_active = False
                            env.reset_source_and_target()
                            print("generating noise")
                            
                        
                        self.noise_steer = time_elapsed/0.5*0.1


                    else:
                        self.noise_steer = 0.3 -0.1*(time_elapsed - 1.5)/0.5
                        env.sensor_active = True
                        #self.noise_timer = None
                        if time_elapsed > 2.5:
                            print("recovered from noise")
                            self.noise_timer = None
                    control.steer = min(1, control.steer+self.noise_steer)


            
            obs, done = env.run_step(control)
            if done:
                print("resetting...")
                env.reset()
                expert = Expert(env)
                continue
            
            if time.time() - last_sample_time >= SAMPLE_TIME:
                if len(self.expert_trajectory) == MAX_REPLAY_BUFFER_SIZE:
                    print("max replay buf size reached")

                    break
                if not env.sensor_active:
                #     print("sensor not currently active")
                #     print(f"timestamp = {time.time() -record_start}")
                    continue
               
                
                if env.current_direction == "left":
                    if self.left_turns < MAX_BRANCH_BUFFER_SIZE:
                        self.left_turns += 1 
                    else:
                        continue
                if env.current_direction == "straight":
                    if self.straight < MAX_BRANCH_BUFFER_SIZE:
                        self.straight += 1
                    else:
                        continue
                if env.current_direction == "right":
                    if self.right_turns < MAX_BRANCH_BUFFER_SIZE:
                        self.right_turns += 1
                    else:
                        continue
                if self.left_turns < MAX_BRANCH_BUFFER_SIZE:
                    print("bias set to left")
                    env.set_turn_bias("left") 
                    pass
                elif self.straight< MAX_BRANCH_BUFFER_SIZE:
                    env.set_turn_bias("straight")
                    pass
                elif self.right_turns < MAX_BRANCH_BUFFER_SIZE:
                    env.set_turn_bias("right")
                    pass
                else:
                    random.shuffle(self.expert_trajectory)
                    break
                self.add_to_buffer(env.get_path())
                print(f"total time = {time.time() - self.timer}")
                print("size of point")                
                last_sample_time = time.time()
       
    def save_recording(self):
        images = [trajectory[0] for trajectory in self.expert_trajectory]
        speeds = [trajectory[1] for trajectory in self.expert_trajectory]
        cmds = [trajectory[2] for trajectory in self.expert_trajectory]
        actions = [trajectory[3] for trajectory in self.expert_trajectory]
        import os
        files = None
        
        if self.collect_training_data:
            print("saved to training folder")
            files = os.listdir(r'.\recordings\training')

            savefile = f'.\\recordings\\training\\recording-{len(files)}.pkl'
            
            with open(savefile, 'wb') as f:
                
                pickle.dump([images, speeds, cmds, actions], f, protocol=4)
        else:
            print("saved to testing folder")
        
            files = os.listdir(r'.\recordings\testing')
            savefile = f'.\\recordings\\testing\\recording-{len(files)}.pkl'

            with open(savefile, 'wb') as f:
        
                pickle.dump([images, speeds, cmds, actions], f, protocol=4)

        
        print("size of array")
        print(asizeof.asizeof([images, speeds, cmds, actions]))
        print(f"total time = {time.time() - self.timer}")
        print()

parser = argparse.ArgumentParser()
parser.add_argument("--port", help="choose port", default=2000)
parser.add_argument("--train", default=True)

args = parser.parse_args()

recorder = recorder(collect_training=args.train)
env = CarEnv(port=args.port, training=args.train)
env.reset()
def exit_handler():
    print("exiting. saving...")
    recorder.save_recording()
    
atexit.register(exit_handler)
recorder.record(env)

recorder.save_recording()