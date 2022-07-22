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
import carla
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
        scale = 2
        while True:
            control = expert.get_action()
            #frequency of noise = 10% 
            if env.training:
                if time.time() - self.start_timer > time_before_noise / scale:
                    self.noise_sign = random.choice([-1,1])
                    self.start_timer = time.time() + NOISE_DURATION / scale
                    self.noise_timer = time.time()
                if env.get_wp_from_loc(env.get_current_location()).is_junction:
                    print("reached junction")
                if env.autocar.is_at_traffic_light():

                    traffic_light = env.autocar.get_traffic_light()
                    if traffic_light.get_state() == carla.TrafficLightState.Red:
                        
                        env.w.debug.draw_string(env.get_current_location(), "is at red traffic light ", life_time=3)
                        traffic_light.set_state(carla.TrafficLightState.Green)

                if self.noise_timer is not None:
                    time_elapsed = (time.time() - self.noise_timer)
                    if time_elapsed < 1.5 / scale:
                        
                        if self.noise_steer is None:
                           
                            env.sensor_active = False
                            env.reset_source_and_target()
                            print("generating noise")
                            
                        
                        self.noise_steer = self.noise_sign * time_elapsed * scale /0.5*0.1


                    else:
                        self.noise_steer =self.noise_sign*(0.3 -0.1*(time_elapsed * scale - 1.5)/0.5)
                        env.sensor_active = True
                        #self.noise_timer = None
                        if time_elapsed > 2.5 / scale:
                            print("recovered from noise")
                            self.noise_timer = None
                    control.steer = min(1, max(control.steer+self.noise_steer, -1))


            
            obs, done = env.run_step(control)
            if env.target_updated:
                env.w.debug.draw_string(env.get_current_location(), f"updated target" , life_time=3)
                expert.set_dest(env.target_loc)
                env.target_updated = False
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
               
                
                if env.current_direction == 3:
                    if self.left_turns < MAX_BRANCH_BUFFER_SIZE:
                        self.left_turns += 1 
                    else:
                        continue
                if env.current_direction == 2:
                    if self.straight < MAX_BRANCH_BUFFER_SIZE:
                        self.straight += 1
                    else:
                        continue
                if env.current_direction == 4:
                    if self.right_turns < MAX_BRANCH_BUFFER_SIZE:
                        self.right_turns += 1
                    else:
                        continue
                if self.left_turns < MAX_BRANCH_BUFFER_SIZE:
                    env.set_turn_bias(3) 
                    pass
                elif self.straight< MAX_BRANCH_BUFFER_SIZE:
                    env.set_turn_bias(2)
                    pass
                elif self.right_turns < MAX_BRANCH_BUFFER_SIZE:
                    env.set_turn_bias(4)
                    pass
                else:
                    random.shuffle(self.expert_trajectory)
                    break
                self.expert_trajectory.append(env.get_path())
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
        random.shuffle(self.expert_trajectory)
        batch_size = 1200
        num_files = len(self.expert_trajectory) // batch_size
        for i in range(num_files):
            batch = self.expert_trajectory[i*batch_size:(i+1)*batch_size]
            with open(f"recordings2\\training\\{i}.pkl", 'wb') as f:
                
                pickle.dump(batch, f)
        try:
            with open(f"recordings2\\training\\{num_files}.pkl", 'wb') as f:
                    
                    pickle.dump(self.expert_trajectory[-(len(self.expert_trajectory)%1200):], f)
        except:
            pass

        return
        # if self.collect_training_data:
        #     print("saved to training folder")
        #     files = os.listdir(r'.\recordings\training')

        #     savefile = f'.\\recordings\\training\\recording-{len(files)}.pkl'
            
        #     with open(savefile, 'wb') as f:
                
        #         pickle.dump([images, speeds, cmds, actions], f, protocol=4)
        # else:
        #     print("saved to testing folder")
        
        #     files = os.listdir(r'.\recordings\testing')
        #     savefile = f'.\\recordings\\testing\\recording-{len(files)}.pkl'

        #     with open(savefile, 'wb') as f:
        
        #         pickle.dump([images, speeds, cmds, actions], f, protocol=4)

        
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