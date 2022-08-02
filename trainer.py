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
        self.status_timer = time.time()
        self.expert_steers = []
        self.agent_steers = [] 
        self.counters = [0,0,0]
        self.debug = debug
        self.env = CarEnv(self.counters, training=True, port=2008)
        #self.expert = Expert(self.env)
        self.agent = agent(debug)
    
    def DR(self, observation, iter, beta):
        ob, spd, cmd = observation 

        s,t,_=self.agent.get_action(ob[0], spd, cmd)
        self.env.w.debug.draw_string(self.env.get_current_location(), f"{round(s, 2)}", life_time=0.1)

   
        if False and (beta * LAMBDA**iter >=  random.random() or self.env.traffic_light_violated()):
            #print("switching to expert control")
            return self.expert.get_action()
        else:
            #print("switching to agent control")
            s,t,b=self.agent.get_action(ob[0], spd, cmd)
            # if s > 0 and (cmd == 3 or cmd == 4):
        
            #     s += 0.1
            # #s += 0.1 * (s / abs(s))
            # elif s < 0 and (cmd == 3 or cmd == 4):
            #     s -= 0.1
            # #s += 0.1 * (s / abs(s))
            # if cmd == 3 and s > 0 or cmd == 4 and s < 0:
            #     s *= -1
            if b <= 0.1:
                b = 0
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

        if self.counters[cmd - 2] >= NUM_SAMPLES_PER_COMMAND_PER_ITER:
            return 
        left_bias_steer = augment_steering(-45, action[0], speed)
        right_bias_steer = augment_steering(45, action[0], speed)
        left_bias_action = [left_bias_steer] + action[1:]
        right_bias_action = [right_bias_steer] + action[1:]
        self.agent.insert_input(imgs[0], speed, cmd, action)
        if not (self.env.autocar.is_at_traffic_light() and self.env.autocar.get_traffic_light().state == carla.TrafficLightState.Red): 
            self.counters[cmd - 2] += 3
            #might not see the traffic light so theres no reason to stop
            self.agent.insert_input(imgs[1], speed, cmd, left_bias_action)
            self.agent.insert_input(imgs[2], speed, cmd, right_bias_action)
        else:
            
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
        return sum(self.counters) >= NUM_SAMPLES_PER_ITER

    def reset_counters(self):
        for i in range(len(self.counters)):
            self.counters[i] = 0
    def sample_and_relabel_trajectory(self, i_iter):
        
        # if self.debug:
        #     rand_img = np.random.uniform(0, 255, (88, 200, 3)).astype('uint8')
        #     for i in range(3):
                
        #         self.try_add_sample(((rand_img, rand_img, rand_img), 10, i + 2), [1,2,3])
        #     return
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
        
            # if len(self.expert_steers) == 50:
            #     self.print_steers()
            #     return 
        
            drive_action = self.DR(ob, i_iter, beta)
            
            
            # if self.env.current_direction == 3 or self.env.current_direction == 4:
            #     print("turning")
            #     pass
            #agent_control = self.translate_action_to_control(action)
            expert_action = self.expert.get_action()
            print(f"orientation {self.env.orientation_wrt_road}")
            
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
        
            if time.time() - self.start_time > SAMPLE_TIME: 
                dirs = ["follow lane", "left", "right"]
                #self.env.w.debug.draw_string(self.env.get_current_location(), f"{dirs[self.env.current_direction - 2]}" if self.env.current_direction is not None else "undefined", life_time=SAMPLE_TIME)
                if self.env.sensor_active:
                    print(f"adding {self.env.current_direction} sample")
                    
                    self.try_add_sample(ob, expert_action)

                self.start_time = time.time()
    
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
        

trainer = imitation_learning_trainer(True)
trainer.main_loop()
