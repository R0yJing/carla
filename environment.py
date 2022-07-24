
import random
import sys

import cv2
from expert import Expert

from lateral_augmentations import augment_steering
sys.path.insert(0, r"C:\Users\autpucv\WindowsNoEditor\PythonAPI\carla")
from pympler import asizeof
import carla
import time
from constants import COLLISION_TIMEOUT, MIN_SPEED, TARGET_SPEED, TARGET_TOLERANCE, WAYPOINT_TIMEOUT
from constants import IM_HEIGHT, IM_WIDTH
import numpy as np
import math
sys.path.insert(0, r"C:\Users\autpucv\WindowsNoEditor\PythonAPI\carla")

class CarEnv:
    def __init__(self, training=True, port=2000, debug=False):
        from sys import path 
       
        self.cl = carla.Client('localhost',port)
        self.training = training
        self.w = self.cl.get_world()
        self.throttle = 0
        self.target_updated = False
        #self.tm = self.cl.get_trafficmanager(8000)
        self.preferred_direction = 3
        self.sps = self.w.get_map().get_spawn_points()
        self.cleanup()
        self.autocar = None
        self.spectator = self.w.get_spectator()
        self.front_camera = None
        self.debug = debug
        self.route = []
        self.collided = False
        self.source_loc = None
        self.final_dest = None
        self.sensor_active= False
        self.auto_control = False
        self.sensors = []
        self.stop = False
        self.spawn_vehicle()
        self.expert = Expert(self.autocar)
        
        print("environment initialised!")
        #obstacle_sensor = self.w.spawn_actor(obstacle_bp, carla.Transform(carla.Location()), attach_to=autocar)
        #obstacle_sensor.listen(lambda event  : self.process_obstacle(event))
    def traffic_light_violated(self):
        return self.autocar.is_at_traffic_light() and self.autocar.get_traffic_light().state == carla.TrafficLightState.Red and self.get_speed() > 0
    def over_turned(self):
        #inverse of a math coordinate system
        #cannot teach the vehicle to turn around the road (this means the agent
        # will sometimes get confused so starts to make U turns!)
        
        car_loc = self.get_current_location()
        car_dir = self.target_loc - car_loc

        target_dir = self.target_loc - self.source_loc
        angle = math.degrees(self.calculate_angle_between(car_dir, target_dir))
        
        if angle >= 80 or angle <= -80:
        
            return True
        return False

    def calculate_angle_between(self, vec2, vec1):
        x2 = vec2.x
        x1 = vec1.x

        y2 = vec2.y
        y1 = vec1.y
  
        dot = x1 * x2 + y1 * y2
        det = x1 * y2 - y1 * x2

        return math.atan2(det, dot)


    def calculate_turn_direction(self, current_dir_vector, previous_dir_vector):
        
        #+ve is left
        #-ve is right
        angle = self.calculate_angle_between(current_dir_vector, previous_dir_vector)

        # if 2 * math.pi >angle > math.pi:
        #     angle = -(2 * math.pi - angle)
        self.sensor_active = True

        print(f"angle = {angle / math.pi * 180}")
        if 35 / 180 * math.pi > angle > -35/180*math.pi:
            print("following")
            return 2
        
        #turn right
        elif -35/180*math.pi >= angle:
            print("turning left")
            return 3

        elif 35/180*math.pi <= angle:
            print("turning right")
            return 4
        else:
            print("turn undefined")
            self.sensor_active = False
            return 2
            #print("undefined")

    def calc_turn_dir(self, current_dir_vector, previous_dir_vector):
        
        #+ve is left
        #-ve is right
        angle = self.calculate_angle_between(current_dir_vector, previous_dir_vector)

        # if 2 * math.pi >angle > math.pi:
        #     angle = -(2 * math.pi - angle)
        self.sensor_active = True
        
        print(f"angle = {angle / math.pi * 180}")
        if 5 / 180 * math.pi > angle > -5/180*math.pi:
            print("following")
            return "straight"
        
        #turn right
        elif -5/180*math.pi >= angle:
            print("turning left")
            return "left"

        elif 5/180*math.pi <= angle:
            print("turning right")
            return "right"
        else:
            print("turn undefined")
            self.sensor_active = False
            return "straight"
            #print("undefined")
    def set_dest(self): pass

    #generate a loop route, currently not working
    def generate_loop(self):
        #these two lines of code is necessary before calling form_loop,
        #which returns a list of waypoints to follow
        wp = self.get_wp_from_loc(self.source_loc)
        #set_target(wp)
        #set_world(self.w)

        self.route, dist = form_loop(wp, [], 0)
        print(f"loop is {dist} m long")
        self.target_loc = self.route[0].transform.location
        count = 2
        for wp in self.route[:-2]:
            self.w.debug.draw_string(wp.transform.location, f"{count}", life_time=60)
            count += 1
        self.w.debug.draw_string(wp.transform.location, f"{1}", life_time=60)
    
    #generate a non-circular route, currently working
    def set_initial_target(self):
        '''assuming car is generated already'''
        wp = self.get_wp_from_loc(self.source_loc)
        self.target_loc = self.get_next_wp(wp, 15).transform.location
        
        self.expert = Expert(self.autocar)
        self.expert.ignore_vehicles(False)
        self.expert.set_dest(self.target_loc)
        self.w.debug.draw_string(self.target_loc, "Initial target", life_time=60)

        # self.route.append(self.target_loc)
        # wp = self.get_wp_from_loc(self.target_loc)

        # dist = self.target_loc.distance(self.source_loc)
        # start_generating = time.time()
        
        # while dist < 1000:
        #     print("generating route...")
        #     if time.time() - start_generating > 10:
        #         break
        #     wp_old = wp
        #     wp = random.choice(wp.next(13))
        #     d = wp.transform.location.distance(wp_old.transform.location) 
        #     if dist >= 8:
        #         self.route.append(wp.transform.location)
        #         dist += d
        # self.route.append(None)
        # self.final_dest = self.route[-2]

    def spawn_vehicle(self, ignore_lights = True):
        if self.autocar is not None:
            actors = self.w.get_actors()
            for sensor in self.sensors: 
                if sensor.destroy():
                    print("dsetroyed ")
                    print(sensor)
                else:
                    print("cannot destroy sensor")
                    print(sensor)
            # self.cl.apply_batch([
            #     carla.command.DestroyActor(x) for x in self.sensors])
                
            self.autocar.destroy()
        model_3 = self.w.get_blueprint_library().filter("mercedes")[0]
        while True:
            i = 0
            try:
                trans = random.choice(self.sps)
                #wp = self.get_wp_from_loc(trans.location)
                self.source_loc = trans.location
                #self.source_loc = wp.transform.location
                if self.autocar is None:

                    self.autocar = self.w.spawn_actor(model_3, trans) 
                else:
                    self.autocar.set_transform(trans)

                break
            except Exception as e: 
             
                print(f"attempting to spawn {i}")
                i+=1


        #sp.location += (carla.Location(x=0, y=-5))
        
        #self.tm.ignore_lights_percentage(self.autocar, 100 if ignore_lights else 0)
        #self.tm.auto_lane_change(self.autocar,True)
        l, wid, h = self.car_dim_info(self.autocar)
        bplib = self.w.get_blueprint_library()
        collision_sensor_bp = bplib.find("sensor.other.collision")
        rgb_cam_bp = bplib.find("sensor.camera.rgb")
        rgb_cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        rgb_cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        rgb_cam_bp.set_attribute("fov", "110")
        
        l, w, h = self.car_dim_info(self.autocar)
        loc = carla.Location(x=l*3/4,z=1.25)#z=2.1*h)
        self.camera = self.w.spawn_actor(rgb_cam_bp, carla.Transform(loc), attach_to=self.autocar)
        self.left_camera = self.w.spawn_actor(rgb_cam_bp, carla.Transform(location=loc, rotation=carla.Rotation(yaw=-45)), attach_to=self.autocar)
        self.right_camera = self.w.spawn_actor(rgb_cam_bp, carla.Transform(location=loc, rotation=carla.Rotation(yaw=45)), attach_to=self.autocar)
        print(self.autocar.get_transform().location)
        print(self.autocar.get_location())
        #self.w.debug.draw_string(loc + self.autocar.get_location(), "x", life_time=60)
        self.collision_sensor = self.w.spawn_actor(collision_sensor_bp, carla.Transform(carla.Location(x=2.5, z=0.7)), attach_to=self.autocar)
        self.sensors = [self.camera, self.left_camera, self.right_camera, self.collision_sensor]
        self.collision_sensor.listen(lambda event : self.process_collision(event))

        self.camera.listen(lambda event : self.process_img(event, 0))
        self.left_camera.listen(lambda event :  self.process_img(event, 1))
        self.right_camera.listen(lambda event :  self.process_img(event, 2))

        #self.left_camera.listen(self.process_img)
    


        # while num_actors < 20:
            
        #     actor = self.w.try_spawn_actor(random.choice(v_bps), random.choice(self.sps))
        #     if actor is not None:
        #         actor.set_autopilot(True, self.tm.get_port())
        #         num_actors += 1
            
    def teleport(self):
        self.source_loc = self.get_wp_from_loc(random.choice(self.sps).location).transform.location
        trans = carla.Transform(self.source_loc)
        self.autocar.set_transform(trans)
    
    def _reset(self):
        '''initialise variables'''
        assert self.autocar is not None

        self.sensor_active = False
        self.collision_timer = None
        self.distance_travelled = 0
        self.route_wp_counter = 0
        self.teleport()
        #self.generate_loop()
        
        while self.front_camera is None or self.left_camera is None or self.right_camera is None:
            print("front camera is none")
            time.sleep(0.01)
        
        
        time.sleep(4)
        self.stop_periodically = False
        self.sensor_active = True
        ##################reverse timer logic####################
        self.current_direction = 2
        self.total_dist_travelled = 0
        self.expert_mode = False 
        #####################################
        #need to wait before camera can receive sensor (otherwise throttle is 0 and 
        # agent will get confused)
        #start counting
        self.waypoint_timer = time.time()
    def reset(self):
        self._reset()
    
        # initial_transform = self.autocar.get_transform()

        # init_dist = self.autocar.get_location().distance(self.target_loc)
        #guide vehicle to drive in the right direction
        self.set_initial_target()
        while True:
            initial_timer = time.time()
            reached_dest = self.autocar.get_location().distance(self.current_target_wp.transform.location) < TARGET_TOLERANCE
            while not reached_dest:
                reached_dest = self.autocar.get_location().distance(self.current_target_wp.transform.location) < TARGET_TOLERANCE
                self.view_spectator()

                if self.collision_timer is not None:
                    self.collision_timer = None

                    break
                print(self.autocar.get_location().distance(self.target_loc))
                self.autocar.apply_control(self.expert.get_action())    
            if reached_dest:
                break
            else:
                self.teleport()
                self.set_initial_target()
        self.update_target()
        self.target_updated = False
        print("finished guiding process!")
        self.waypoint_timer = time.time()
        return ((self.front_camera, self.left_camera, self.right_camera), self.get_speed(), self.current_direction), False

    def cleanup(self):

        try:
            self.cl.apply_batch([carla.command.DestroyActor(x) for x in [actor for actor in list(filter(
                lambda x: isinstance(x, carla.Sensor) and (isinstance(x.parent, carla.Vehicle) or x.parent is None), self.w.get_actors()))]])
            
            self.cl.apply_batch([carla.command.DestroyActor(x) for x in [actor for actor in list(filter(
                lambda x: isinstance(x, carla.Vehicle), self.w.get_actors()))]])
             
            print("clean up successful")
        except Exception as e:
            print("cannot destroy all actors")
            print(e)
    
    def car_dim_info(self, car):
        bbox = car.bounding_box
        length, width, height = bbox.extent.x, bbox.extent.y, bbox.extent.z
        return length, width, height
    

    def get_speed(self):
        v = self.autocar.get_velocity()
        return math.sqrt(v.x**2 + v.y**2)
    def get_path(self):
        
        if self.current_direction == None:
            raise Exception("env.getpath: direction not defined")
        control = self.autocar.get_control()
        control = [control.steer, control.throttle, control.brake]
        spd  = self.get_speed()
        left_steer = augment_steering(-45, control.steer, spd)
        right_steer =  augment_steering(45, control.steer, spd)

        left_control = carla.VehicleControl(control)
        right_control = carla.VehicleControl(control)
        left_control.steer = left_steer
        right_control.steer = right_steer
        print(asizeof.asizeof([self.front_camera, spd, self.current_direction, control]))
        return [self.front_camera, self.left_camera, self.right_camera, self.get_speed(), self.current_direction, control, left_control, right_control]


    def get_biased_target_if_any(self, prev_loc, current_loc, dist=10):

        wp = self.get_wp_from_loc(current_loc)
        prev_dir_vec = current_loc - prev_loc 
        wps = wp.next(dist)
        use_preferred = random.random() > 0.2
        preferred_wps = [] 
        if not use_preferred:
            return random.choice(wps).transform.location
        for wp in wps:
            current_dir_vec = wp.transform.location - current_loc
            direction = self.calculate_turn_direction(current_dir_vec, prev_dir_vec)
            if self.preferred_direction == direction and self.preferred_direction is not None:
                preferred_wps.append(wp)

        if len(preferred_wps) == 0:
        
            return random.choice(wps).transform.location
        else:
            return random.choice(preferred_wps).transform.location
    def update_target(self):
        print("updating target")
        prev_dir = self.target_loc - self.source_loc
        if self.preferred_direction != None:
            next_target = self.get_biased_target_if_any(self.source_loc, self.target_loc)
            self.source_loc = self.target_loc
            self.target_loc = next_target
        else:
            self.total_dist_travelled += self.target_loc.distance(self.source_loc)
            self.route_wp_counter = (self.route_wp_counter + 1) % len(self.route)
            self.source_loc = self.target_loc

            self.target_loc = self.route[self.route_wp_counter].transform.location
        self.target_updated = True
        if self.target_loc is None:
            return
        else:
            self.expert.set_dest(self.target_loc)
        self.w.debug.draw_string(self.target_loc, "next target", life_time=10)

        current_dir = self.target_loc - self.source_loc
        self.current_direction = self.calculate_turn_direction(current_dir, prev_dir)
        #print("current direction" + ["follow", "left", "right","straight"][self.current_direction - 2])
        self.waypoint_timer = time.time()
    def set_turn_bias(self, direction):
        self.preferred_direction = direction

    def get_next_wp(self, wp, dist=10):
        return random.choice(wp.next(dist))
    @property
    def current_target_wp(self):
        return self.get_wp_from_loc(self.target_loc)

    def get_wp_from_loc(self, loc):
        '''only return a wp that is suitable to drive to'''
        return self.w.get_map().get_waypoint(loc, project_to_road=True, lane_type=(carla.LaneType.Driving))

    def get_current_location(self):
        
        return self.autocar.get_location()
    

    def timedout(self):
        
        if time.time() - self.waypoint_timer > WAYPOINT_TIMEOUT and not (self.autocar.is_at_traffic_light() and self.autocar.get_traffic_light().state == carla.TrafficLightState.Red): #and\
            #  (self.collision_timer is None or \
            # self.collision_timer >= COLLISION_TIMEOUT and self.autocar.get_speed() >= MIN_SPEED)):
        
            return True
        return False
    def reset_source_and_target(self):
        '''if timed out should be called before resetting the waypoint timer'''
        print("resetting src and target")
        wp = self.get_wp_from_loc(self.get_current_location())
        #if stopping at a traffic light, will still not time out
        if self.timedout():

            new_target_loc = random.choice(wp.next(10)).transform.location
            self.current_direction = self.calculate_turn_direction(new_target_loc - wp.transform.location, self.target_loc - self.source_loc)
            self.target_loc = new_target_loc
            
        else: #only the noise period
            self.target_loc = self.get_biased_target_if_any(self.source_loc, wp.transform.location, 20)
            #self.w.debug.draw_string(self.target_loc, "noise reset", life_time=10)
            self.source_loc = wp.transform.location
        self.expert.set_dest(self.target_loc)
        action = self.expert.get_action()
        if action.throttle == 0.0 or action.brake >= 0.5:
            return False
        
        self.waypoint_timer = time.time()
        return True
    def reset_to_last_checkpoint(self):
        self.autocar.set_transform(self.position_history[-1])
        control = self.control_history[-1]
        
    def run_step(self, control):
        done = False
        if self.collision_timer is not None and time.time() - self.collision_timer >= COLLISION_TIMEOUT:
            self.collision_timer = None
            print("collided")
            if self.get_speed() < MIN_SPEED:
                done = True
        
        elif self.collision_timer is None and self.timedout():
            print("timed out, replanning route")
            
            if not self.reset_source_and_target():
                done = True
            else:
                self.target_updated = True
                self.collision_timer = None
                self.sensor_active = True
                
            #self.w.debug.draw_string(self.target_loc, "replanned target", life_time=10)
        
        elif self.get_current_location().distance(self.target_loc) < TARGET_TOLERANCE:
            self.total_dist_travelled += self.target_loc.distance(self.source_loc)
            self.target_updated = True
            self.update_target()
            
            self.waypoint_timer = time.time()
            if self.target_loc is None:
                done = True
                print("reached fin dest, resetting...")
        if self.over_turned():
            self.target_updated = True
            self.update_target()
            print("over turned")
           
        
        trans_car = self.autocar.get_transform()
        trans_car.rotation.pitch -= 90
        trans_car.location.z += 20
    
        self.spectator.set_transform(trans_car)
        s = round(control.steer, 2)
        t = round(control.throttle, 2)
        b = round(control.brake, 2)
        self.w.debug.draw_string(trans_car.location, f"steer: {s}\n throttle: {t}\n brake: {b}", life_time=1.5)

        self.autocar.apply_control(control)
        return ((self.front_camera, self.left_camera, self.right_camera), self.get_speed(), self.current_direction), done
    
    def view_spectator(self):
        trans_car = self.autocar.get_transform()
        trans_car.rotation.pitch -= 90
        trans_car.location.z += 20
    
        #self.w.debug.draw_string(trans_car.location, "vehicle", life_time=0.1)
        self.spectator.set_transform(trans_car)
        
#carla.Transform(carla.Location(x=random.randint(0, 100), y=random.randint(0,100),z=5)))
#sp.location += (carla.Location(x=0, y=-5))

    def process_img(self, event, sensor_id):
        
        i = np.array(event.raw_data)
        t = i.dtype
        i.resize((IM_HEIGHT, IM_WIDTH, 4))
        #i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i[:, :, :3]
        
        #cv2.imshow("", i3)
        #cv2.waitKey(1)
        if sensor_id == 0:
            self.front_camera = i3
           
        elif sensor_id == 1:
            self.left_camera = i3
        else: 
            self.right_camera = i3
        #wait for the least amount of time possible
        if self.debug:
            cv2.imshow("front cam", self.front_camera)
            cv2.imshow("left cam", self.left_camera)
            cv2.imshow("right cam", self.right_camera)
            cv2.waitKey(1)

    def process_collision(self, event):
        if self.collision_timer is not None:
            return
        self.collision_timer = time.time()
        
        imp = event.normal_impulse

        print("collision occured")
        self.sensor_active = False
        impulse = math.sqrt(imp.x**2 + imp.y**2 + imp.z**2)
        print(impulse)
# env =CarEnv(training=True)
# trans_car = env.autocar.get_transform()
# trans_car.rotation.pitch -= 90
# trans_car.location.z += 20

# # env.w.debug.draw_string(trans_car.location, "vehicle", life_time=0.1)
# env.spectator.set_transform(trans_car)
# time.sleep(9999)
# while True:
    
#     trans = autocar.get_transform()
#     trans.location.z = 20
#     trans.rotation.pitch -= 90
#     spectator.set_transform(trans)
#     if busy_reverse_timer > 20:
#         busy_reverse_timer = 0
#         print("too many reverses")
#         autocar.set_simulate_physics(False)
#         autocar.set_transform(random.choice(sps))
#         autocar.set_simulate_physics(True)
       
#     elif reverse_timer :
#         if time.time() - reverse_timer >= 2:
#             reverse_timer = None
#             last_reverse = time.time()  
#             autocar.apply_control(carla.VehicleControl(brake=1, throttle=0))
#             reversing = False
#         #time between end of last reverse and the start of current reverse
#         elif last_reverse is not None and time.time() - last_reverse <= 3 and not reversing:
#             #last reverse must not be none
#             reversing = True
#             busy_reverse_timer += time.time() - last_reverse
            
#         # if last reverse
        
#     else:
#         if busy_reverse_timer > 0 and last_reverse is not None and time.time() - last_reverse > 10:
#             busy_reverse_timer = 0
#             print("busy rev timer timedout!")
#         if time.time() - start < 2:
#             print("random action")
#             autocar.apply_control(carla.VehicleControl(steer=random.random(), throttle=0.5))
#         elif time.time() - start < 5:
#             print("agent action")
#             control = agent.run_step()
#             autocar.apply_control(control)
#             trans= autocar.get_transform()
#         else:
#             start = time.time()
#     time.sleep(0.1)
