from itertools import count
from environment import *
from expert import Expert 
from constants import *
DATA_DIR = "some/dir"
from neural_net_v2 import agent
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
        self.env = CarEnv(self.counters, self.num_obs_at_traffic_light_counter, training=True, port=2000, debugg=debug)
        #self.expert = Expert(self.env)
        self.agent = agent(debug)
    
    def DR(self, observation, iter, beta):
        ob, spd, cmd = observation 

        s,t,_=self.agent.get_action(ob[0], spd, cmd)
        

        if (beta * LAMBDA**iter >=  random.random() or self.env.traffic_light_violated()):
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
    def try_add_sample(self, obs, action):
        if not self.env.sensor_active:
            return 
        imgs, speed, cmd = obs

        # if cmd == 2:
        #     #getting too many non traffic light samples
        #     if self.counters[0] - self.num_obs_at_traffic_light_counter[0] > math.floor(NUM_SAMPLES_PER_COMMAND_PER_ITER * 0.5):
        #         return
        if self.counters[cmd - 2] >= self.NUM_SAMPLES_PER_COMMAND_PER_ITER:
            return 
        #in the worst case scenario, nums at traffic light = 0 therefore this is an upper bound
        # elif sum(self.counters) - self.num_obs_at_traffic_light_counter[0] > math.ceil(NUM_SAMPLES_PER_COMMAND_PER_ITER * 2.5):
        #     return 
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
            if self.env.autocar.get_traffic_light() != None:
                self.env.traffic_light_counter[0] += 3
            #green or amber light
        
        else:

            print("adding red light samples")
            self.num_obs_at_traffic_light_counter[0] += 1
            self.counters[cmd - 2] += 1
    
    def translate_action_to_control(self, action):
        steer, throttle, brake = action
        #avoid fake braking
        if brake < 0.1:
            brake = 0
        
        # if self.env.current_direction == 5:
        #     steer = 0
        v = self.env.autocar.get_velocity()

        return carla.VehicleControl(throttle=throttle, steer = steer, brake=brake)

   
    def print_steers(self):
        from matplotlib import pyplot as plt
        plt.plot(self.expert_steers)
        plt.plot(self.agent_steers)

        plt.title('steer graph')
        plt.ylabel('steer')
        plt.xlabel('time')
        plt.legend(['expert', 'agent'], loc='upper left')
        plt.show()
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
            self.env.autocar.set_autopilot(False, 8000)
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
            
            print(self.counters, self.env.sensor_active)
            preferred_turn = self.counters.index(min(self.counters)) + 2
        
            # if len(self.expert_steers) == 50:
            #     self.print_steers()
            #     return 
        
            drive_action = self.DR(ob, i_iter, beta)
            
            
            # if self.env.current_direction == 3 or self.env.current_direction == 4:
            #     print("turning")
            #     pass
            #agent_control = self.translate_action_to_control(action)
            expert_action = self.expert.get_action()
            #print(f"orientation {self.env.orientation_wrt_road}")
            # print(self.counters)
            # print(self.num_obs_at_traffic_light_counter)
            #self.env.set_guideline_control(expert_action)
            ob, done = self.env.run_step(drive_action)
            
            #in case a collision occurred or something reset
            if done:
                self.env.reset()
                self.expert = Expert(self.env)
                continue
            expert_action = [expert_action.steer, expert_action.throttle, expert_action.brake]
            if self.env.target_updated:
                self.env.set_turn_bias(preferred_turn)

                self.expert.update_target()
                self.env.target_updated = False 
        
            if time.time() - self.start_time > self.SAMPLE_TIME: 
                dirs = ["follow lane", "left", "right"]
                #self.env.w.debug.draw_string(self.env.get_current_location(), f"{dirs[self.env.current_direction - 2]}" if self.env.current_direction is not None else "undefined", life_time=SAMPLE_TIME)
                self.try_add_sample(ob, expert_action)

                self.start_time = time.time()
        self.env.teleport()

        self.env.autocar.set_autopilot(True,8000)
    #path is the rollout of the current policy
    def main_loop(self):
        
        self.start_time = time.time()
        i = 0
        stopped = False
        while i < N_ITER:
            print(f"iteration {i}")
            self.sample_and_relabel_trajectory(i)
            self.agent.train()
            i += 1
        self.agent.show_graph()
        #self.show_statistics()
        

trainer = imitation_learning_trainer(debug=False)
trainer.main_loop()
