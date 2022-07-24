import carla
from sys import path
import random
import time
from constants import TARGET_SPEED, TARGET_TOLERANCE
path.insert(0, r"C:\Users\autpucv\WindowsNoEditor\PythonAPI\carla")
from agents.navigation.controller import VehiclePIDController
from agents.navigation.basic_agent import BasicAgent

class Expert:
    def __init__(self, autocar : carla.Vehicle):
        self.agent = VehiclePIDController(autocar, args_lateral = {'K_P': 1, 'K_D': 0.0, 'K_I': 0}, args_longitudinal = {'K_P': 1, 'K_D': 0.0, 'K_I': 0.0})
        self.basic_agent = BasicAgent(autocar, TARGET_SPEED, {'ignore_vehicles' : True})#
        #self.basic_agent.set_destination(env.target_loc)
        self.reverse_timer = None
        self.last_reverse = None
        self.reverse_bkwds = False
        self.busy_reverse_timer = 0
        
        self.first_collision = True
        ##############set dest and src
    def ignore_vehicles(self, ignore):
        self.basic_agent.ignore_vehicles(ignore)  
    def set_dest(self, destination):
        self.basic_agent.set_destination(destination)
        #self.agent.set_destination(destination)
    #get recommended action based on observation and cmd
        

    def update_target(self, target_loc):
        print("target updated")
        self.set_dest( target_loc)
        self.basic_agent.set_destination( target_loc)
    def get_action_pid(self, current_target_wp) -> carla.VehicleControl:
        wp =  current_target_wp
        return self.agent.run_step(TARGET_SPEED, wp)
    
    def get_action(self):
        '''technically need observation, but BasicAgent does not'''
        control = None
        # if   collided and self.reverse_timer is None:

        #     print("expert: collision occured, reversing")
        #     self.reverse_timer = time.time()
            
        # if self.reverse_timer:
        #     current_time = time.time()
            
        #     if current_time - self.reverse_timer >= 2:
        #         self.last_reverse = current_time
        #         self.reverse_timer = None
        #           collided = False
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
    
