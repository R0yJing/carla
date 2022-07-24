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
        #disable the timeout when training
        self.env = CarEnv(training=True, port=2000)
        #self.expert = Expert(self.env)
        self.agent = agent(debug)
        self.left_turns_counter = 0
        self.right_turns_counter = 0
        self.follow_lane_counter = 0
        self.status_timer = time.time()
        self.expert_steers = []
        self.agent_steers = [] 
        self.counters = [0,0,0]
        self.debug = debug
    def DR(self, observation, iter, beta):
        ob, spd, cmd = observation 
        
        if beta * LAMBDA**iter >=  random.random() or self.env.traffic_light_violated():
            print("switching to expert control")
            return self.env.expert.get_action()
        else:
            print("switching to agent control")
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
    def try_add_sample(self, obs, action):
        imgs, speed, cmd = obs
        left_bias_steer = augment_steering(-45, action[0], speed)
        right_bias_steer = augment_steering(45, action[0], speed)
        left_bias_action = [left_bias_steer] + action[1:]
        right_bias_action = [right_bias_steer] + action[1:]

        self.counters[cmd - 2] += 3

        self.agent.insert_input(imgs[0], speed, cmd, action)
        self.agent.insert_input(imgs[1], speed, cmd, left_bias_action)
        self.agent.insert_input(imgs[2], speed, cmd, right_bias_action)

    def sample_and_relabel_trajectories(self):
        timesteps = 0
        #this is not mini batch size
        while timesteps < MIN_TIMESTEPS_PER_BATCH:
            self.sample_and_relabel_trajectory()
            timesteps += EPISODIC_BUFFER_LEN
    def translate_action_to_control(self, action):
        steer, throttle, brake = action
        #avoid fake braking
        if brake < 0.1:
            brake = 0
        
        # if self.env.current_direction == 5:
        #     steer = 0
        v = self.env.autocar.get_velocity()

        return carla.VehicleControl(throttle=throttle, steer = steer, brake=brake)

    def print_status(self, agent_action, expert_action):
        if len(self.expert_steers) < 100:
            self.expert_steers.append(expert_action.steer)
            self.agent_steers.append(agent_action.steer)
    
        expert_action = [expert_action.steer, expert_action.throttle, expert_action.brake]
    
        # print("command = " + self.env.current_direction + "\n" + \
        # "speed = " + str(self.env.get_speed()) + "\n\n" + \
        # "agent action" + "\n" + \
        # str([agent_action.steer, agent_action.throttle, agent_action.brake]) + "\n" +
        # "expert action = \n" + \
        # str(expert_action) + "\n")

        
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
        return min(self.counters) >= 3
    def sample_and_relabel_trajectory(self, i_iter):
        
        # if self.debug:
        #     rand_img = np.random.uniform(0, 255, (88, 200, 3)).astype('uint8')
        #     for i in range(3):
                
        #         self.try_add_sample(((rand_img, rand_img, rand_img), 10, i + 2), [1,2,3])
        #     return
        ob, done = self.env.reset()
        self.status_timer = time.time()
        #try to start the car
        #each sample should be 1km long?
        beta = 0.75
        t = time.time()
        while (time.time() - t < TOTAL_SAMPLE_TIME):
            
            preferred_turn = self.counters.index(min(self.counters)) + 2

            # if len(self.expert_steers) == 50:
            #     self.print_steers()
            #     return 
        
            drive_action = self.DR(ob, i_iter, beta)
            
            
            # if self.env.current_direction == 3 or self.env.current_direction == 4:
            #     print("turning")
            #     pass
            #agent_control = self.translate_action_to_control(action)
            expert_action = self.env.expert.get_action()
        
            ob, done = self.env.run_step(drive_action)
            #in case a collision occurred or something reset
            if done:
                self.env.reset()
                self.expert = Expert(self.env)
                continue
            expert_action = [expert_action.steer, expert_action.throttle, expert_action.brake]
            if self.env.target_updated:
                self.env.set_turn_bias(preferred_turn)
                self.env.target_updated = False 
        
            if time.time() - self.start_time > SAMPLE_TIME: 
                dirs = ["follow lane", "left", "right"]
                self.env.w.debug.draw_string(self.env.get_current_location(), f"{dirs[self.env.current_direction - 2]}" if self.env.current_direction is not None else "undefined", life_time=SAMPLE_TIME)
                if self.env.sensor_active:
                    print("adding sample")
                    self.try_add_sample(ob, expert_action)

                self.start_time = time.time()
    
    #path is the rollout of the current policy
    def main_loop(self):
        
        self.start_time = time.time()
        i = 0
        stopped = False
        while not stopped and i < N_ITER:
            print(f"iteration {i}")
            self.sample_and_relabel_trajectory(i)
            stopped = self.agent.train()
            i += 1
        self.agent.show_graph()
        #self.show_statistics()
        

trainer = imitation_learning_trainer(True)
trainer.main_loop()
