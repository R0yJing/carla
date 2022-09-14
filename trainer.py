from itertools import count
from environment import *
from expert import Expert 
from constants import *
DATA_DIR = "some/dir"
#from neural_net_v2 import agent
from neural_net_v4 import agent
import time
from collections import deque
import carla
from lateral_augmentations import augment_steering
class imitation_learning_trainer:
    def __init__(self, debug=False):
        #disable the timeout when trainingself.left_turns_counter = 0
        self.right_turns_counter = 0
        self.follow_lane_counter = 0
        
        self.num_obs_at_traffic_light_counter = [0]
        self.status_timer = time.time()
        self.expert_steers = []
        self.agent_steers = [] 
        self.counters = [0,0,0]
        self.debug = debug
        self.env = CarEnv(self.counters, self.num_obs_at_traffic_light_counter, training=True, port=2000, debugg=debug, skip_turn_samples=False)
        #self.expert = Expert(self.env) 
        self.agent = agent(None, max_val_lim=0)
        self.collected_enough_samples()
    def DR(self, observation, iter, beta):
        ob, spd, cmd = observation 
        
        if (beta * (LAMBDA**iter) >=  random.random()):
            #print("switching to expert control")
            return self.expert.get_action()
        else:
            #print("switching to agent control")
            s,t,b=self.agent.get_action(ob[0], spd, cmd)
            
            return carla.VehicleControl(steer=s, throttle=t, brake=b)
      #  time.sleep(999)
    # def has_collected_enough_samples_per_episode(self):
    #     if self. follow_lane_counter == EPISODIC_BUFFER_LEN / 3 and \
    #     self.right_turns_counter == EPISODIC_BUFFER_LEN / 3 and \
    #     self.left_turns_counter == EPISODIC_BUFFER_LEN / 3:
    #         self.left_turns_counter = 0
    #         self.right_turns_counter = 0
    #         self. follow_lane_counter = 0
    #         return True
    #     return False
    
    @property
    def NUM_SAMPLES_PER_COMMAND_PER_ITER(self):
        return NUM_SAMPLES_PER_COMMAND_PER_ITER if not self.debug else DEBUG_NUM_SAMPLES_PER_COMMAND_PER_ITER
    def try_add_sample(self, obs, action, n_samples_per_cmd_type=None):
        if n_samples_per_cmd_type is None:
            n_samples_per_cmd_type = self.NUM_SAMPLES_PER_COMMAND_PER_ITER
        else:
            n_samples_per_cmd_type = 22
        if not self.env.sensor_active:
            return 
        imgs, speed, cmd = obs
        if self.counters[cmd - 2] >= self.NUM_SAMPLES_PER_COMMAND_PER_ITER:
            return 
      
        left_bias_steer = augment_steering(-45, action[0], speed)
        right_bias_steer = augment_steering(45, action[0], speed)
        left_bias_action = [left_bias_steer] + action[1:]
        right_bias_action = [right_bias_steer] + action[1:]
        self.agent.insert_input(imgs[0], speed, cmd, action)
        if not (self.env.waiting_for_light()): 
        
            self.counters[cmd - 2] += 3
            #might not see the traffic light so theres no reason to stop
            self.agent.insert_input(imgs[1], speed, cmd, left_bias_action)
            self.agent.insert_input(imgs[2], speed, cmd, right_bias_action)
            try:
                if self.env.autocar.get_traffic_light() != None:
                    self.env.traffic_light_counter[0] += 3
            except:
                pass
            #green or amber light
        
        else:

            print("adding red light samples")
            self.num_obs_at_traffic_light_counter[0] += 1
            self.counters[cmd - 2] += 1
    
    def translate_action_to_control(self, action):
        steer, throttle, brake = action
        #avoid fake braking
        # if brake < 0.1:
        #     brake = 0
        
        # if self.env.current_direction == 5:
        #     steer = 0
        v = self.env.autocar.get_velocity()

        return carla.VehicleControl(throttle=throttle, steer = steer, brake=brake)

   
    
    def collected_enough_samples(self):
        
        return all([samples >= self.NUM_SAMPLES_PER_COMMAND_PER_ITER for samples in self.counters])
    @property

    def SAMPLE_TIME(self):
        return 0 if self.debug else SAMPLE_TIME

    def reset_counters(self):
        for i in range(len(self.counters)):
            self.counters[i] = 0
    def sample_and_relabel_trajectory(self, i_iter):
        
        if i_iter > 0:
            self.env.set_autopilot(False)
        ob, done = self.env.reset()
        self.reset_counters()
        #next target already set
        self.expert = Expert(self.env)
        
        #done in this case means we are still collecting turn samples
        if not done:
            self.expert.update_target()
            self.env.target_updated = False
        self.status_timer = time.time()
        #try to start the car
        beta = 0.75
        t = time.time()

        while (not self.collected_enough_samples()):
            # print(f"orientation {self.env.orientation_wrt_road}")
            # print(f"dist {self.env.abs_distance_from_lane_edge}")
            
            preferred_turn = self.counters.index(min(self.counters)) + 2

            drive_action = self.DR(ob, i_iter, beta)
            expert_action = self.expert.get_action()
          
            reached_dest = self.env.reached_dest()
            self.env.w.debug.draw_string(self.expert.target_loc, "expert destination")

            
            ob, done = self.env.run_step(drive_action)
            #if reached_dest or force_update_targ
                
            if self.expert.target_loc != self.env.target_loc:
                self.expert.update_target()
                self.env.force_update_targ = False
                self.env.set_turn_bias(preferred_turn)
                
                self.env.target_updated = False 
            #in case a collision occurred or something reset
            if done:
                self.env.reset()
                self.expert = Expert(self.env)
                continue
            expert_action = [expert_action.steer, expert_action.throttle, expert_action.brake]
            
        
            if time.time() - self.start_time > self.SAMPLE_TIME: 
                dirs = ["follow lane", "left", "right"]
                #self.env.w.debug.draw_string(self.env.get_current_location(), f"{dirs[self.env.current_direction - 2]}" if self.env.current_direction is not None else "undefined", life_time=SAMPLE_TIME)
                self.try_add_sample(ob, expert_action)

                self.start_time = time.time()
        self.env.teleport()

        self.env.set_autopilot(True)
    #path is the rollout of the current policy
    def main_loop(self):
        
        self.start_time = time.time()
        i = 0
        while i < TOTAL_NUM_ITER:
            print(f"iteration {i}")
            self.sample_and_relabel_trajectory(i)
            if not self.agent.train():
                break 
            i += 1
            
        self.agent.show_plots()

        #self.show_statistics()
        

trainer = imitation_learning_trainer(debug=False)
trainer.main_loop()
