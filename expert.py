import carla
from sys import path
import random
import time
from constants import TARGET_SPEED, TARGET_TOLERANCE
path.insert(0, r"C:\Users\autpucv\WindowsNoEditor\PythonAPI\carla")
from agents.navigation.controller import VehiclePIDController
from agents.navigation.basic_agent import BasicAgent
from environment import CarEnv
class Expert:
    def __init__(self, env : CarEnv, ignore_t_lights=True):
        self.basic_agent = BasicAgent(env.autocar, TARGET_SPEED, {'ignore_traffic_lights' : ignore_t_lights})#
        self.basic_agent.set_destination(env.target_loc)
        self.env = env
        self.autocar = env.autocar
        assert env.target_loc is not None
        self.set_dest(env.target_loc)

        ##############set dest and src
    def car_respawned(self):
        
        return self.autocar != self.env.autocar
    def set_dest(self, destination):
        self.basic_agent.set_destination(destination)
        self.target_loc = destination

        #self.agent.set_destination(destination)
    #get recommended action based on observation and cmd
        
    
    def update_target(self):
        print("target updated")
    
        self.set_dest(self.env.target_loc)
        self.env.w.debug.draw_string(self.env.target_loc, "expert target")
    def get_action_pid(self) -> carla.VehicleControl:
        wp = self.env.current_target_wp
        return self.agent.run_step(TARGET_SPEED, wp)
    
    def get_action(self):
        '''technically need observation, but BasicAgent does not'''
        control = None
        # if self.env.collided and self.reverse_timer is None:

        #     print("expert: collision occured, reversing")
        #     self.reverse_timer = time.time()
            
        # if self.reverse_timer:
        #     current_time = time.time()
            
        #     if current_time - self.reverse_timer >= 2:
        #         self.last_reverse = current_time
        #         self.reverse_timer = None
        #         self.env.collided = False
        #         return carla.VehicleControl(brake=1, throttle=0)
                
        #     elif self.last_reverse is not None or self.first_collision:
        #         if self.first_collision:
        #             self.first_collision = False
        #         print("reversing")
        #         if self.last_reverse is not None and current_time - self.last_reverse <= 5:
                
        #             self.busy_reverse_timer += current_time - self.last_reverse
        #             self.last_reverse = None

        #         self.reverse_bkwds = not self.reverse_bkwds
        #         return carla.VehicleControl(reverse=self.reverse_bkwds, throttle=0.5)

        # elif self.busy_reverse_timer > 0 \
        #  and self.last_reverse is not None \
        #  and time.time() - self.last_reverse >= 20:
        #     print("resetting busy reverse timer...")
        #     self.busy_reverse_timer = 0
        control = self.basic_agent.run_step()
        
        return control
        #return [control.steer, control.throttle, control.brake]
    
