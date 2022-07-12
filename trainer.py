from environment import *
from expert import Expert 
from constants import *
DATA_DIR = "some/dir"
from neural_net_v2 import agent
import time
from collections import deque
import carla

class imitation_learning_trainer:
    def __init__(self):
        #disable the timeout when training
        self.env = CarEnv(training=True, port=2000)
        self.expert = Expert(self.env)
        self.agent = agent(fake_training=True)
        self.episodic_left_cmd_counter = 0
        self.episodic_right_cmd_counter = 0
        self.episodic_straight_cmd_counter = 0
        self.status_timer = time.time()
        self.expert_steers = []
        self.agent_steers = []
      #  time.sleep(999)
    # def has_collected_enough_samples_per_episode(self):
    #     if self.episodic_straight_cmd_counter == EPISODIC_BUFFER_LEN / 3 and \
    #     self.episodic_right_cmd_counter == EPISODIC_BUFFER_LEN / 3 and \
    #     self.episodic_left_cmd_counter == EPISODIC_BUFFER_LEN / 3:
    #         self.episodic_left_cmd_counter = 0
    #         self.episodic_right_cmd_counter = 0
    #         self.episodic_straight_cmd_counter = 0
    #         return True
    #     return False
    def try_add_sample(self, img, speed, cmd,action):
        if cmd == "left" and self.episodic_left_cmd_counter < EPISODIC_BUFFER_LEN / 3:
            self.episodic_left_cmd_counter += 1
        elif cmd == "straight" and self.episodic_straight_cmd_counter < EPISODIC_BUFFER_LEN / 3:
            self.episodic_straight_cmd_counter += 1
        elif cmd == "right" and self.episodic_right_cmd_counter < EPISODIC_BUFFER_LEN / 3:
            self.episodic_right_cmd_counter += 1
        else: return 

        self.agent.insert_input(img, speed, cmd, action)
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
        v = self.env.autocar.get_velocity()
        spd = math.sqrt(v.x**2 + v.y**2)

        print("acceleration = " + str(self.env.throttle))
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
    def sample_and_relabel_trajectory(self):
        ob, done = self.env.reset()
        self.status_timer = time.time()
        #try to start the car
        #each sample should be 1km long?
        while True:
            # if len(self.expert_steers) == 50:
            #     self.print_steers()
            #     return 
            spd = self.env.get_speed()
            command = self.env.current_direction
            ob0 = ob
            action = self.agent.get_action(ob, spd , 4) #self.env.current_direction)
            
            agent_control = self.translate_action_to_control(action)
            expert_action = self.expert.get_action_pid()

            if time.time() - self.status_timer >= 0.5:
                self.status_timer = time.time()
                self.print_status(agent_control, expert_action)

            ob, done = self.env.run_step(agent_control)
            expert_action = [expert_action.steer, expert_action.throttle, expert_action.brake]
          

           
            if done:
                self.env.reset()
               # self.env.reset()
            elif time.time() - self.start_time > SAMPLE_TIME: 
                dirs = ["follow lane", "left", "right"]
                self.env.w.debug.draw_string(self.env.get_current_location(), f"{dirs[self.env.current_direction - 2]}" if self.env.current_direction is not None else "undefined", life_time=SAMPLE_TIME)
                self.env.w.debug.draw_string(self.env.get_current_location() + carla.Location(0, y=1), str(int(100*self.env.autocar.get_control().steer)/100.0), life_time=SAMPLE_TIME)
                if self.env.sensor_active:
                    print("adding sample")
                    self.try_add_sample(ob0, spd, command, expert_action)
                if self.episodic_left_cmd_counter < EPISODIC_BUFFER_LEN / 3:
                    
                    self.env.set_turn_bias("left") 
                elif self.episodic_straight_cmd_counter< EPISODIC_BUFFER_LEN / 3:
                    self.env.set_turn_bias("straight")
                elif self.episodic_right_cmd_counter < EPISODIC_BUFFER_LEN / 3:
                    self.env.set_turn_bias("right")
                else: 
                    self.episodic_left_cmd_counter = self.episodic_straight_cmd_counter = self.episodic_right_cmd_counter = 0
                    return
                self.start_time = time.time()
        
    #path is the rollout of the current policy
    def main_loop(self):
        #self.agent.train()
        self.measure_disk_timer = time.time()
        self.start_time = time.time()
        for i in range(3):
            print(f"iteration {i}")
            self.sample_and_relabel_trajectories()
            self.agent.train()
        self.agent.show_accuracy_graph()


trainer = imitation_learning_trainer()
trainer.main_loop()
