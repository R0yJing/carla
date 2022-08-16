
import random
import sys

import cv2
from pyrsistent import m

from lateral_augmentations import augment_steering
sys.path.insert(0, r"C:\Users\autpucv\WindowsNoEditor\PythonAPI\carla")
from pympler import asizeof
import carla
import time
from constants import *
import numpy as np
import math
# from form_loop import form_loop, set_target, set_world
from agents.navigation.controller import VehiclePIDController
class StartEndPair:

    def __init__(self, start, end):
        self.start = start
        self.end = end
    
class CarEnv:
    def __init__(self, counter, traffic_light_counter, training=True, port=2000, debug=False, enable_fast_simulation=False):
        from sys import path 
        self.counters = counter
        self.traffic_light = None
        self.cl = carla.Client('localhost',port)
        self.training = training
        self.w = self.cl.get_world()
        self.traffic_light_counter = traffic_light_counter
        if enable_fast_simulation:
            s = self.w.get_settings()
            s.fixed_delta_seconds = 0.05
            self.w.apply_settings(s)

        self.throttle = 0
        self.target_updated = False
        #self.tm = self.cl.get_trafficmanager(8000)
        self.preferred_direction = 3
        self.sps = self.w.get_map().get_spawn_points()
        self.cleanup()
        self.autocar = None
        self.spectator = self.w.get_spectator()
        self.front_camera = None
        self.left_turns, self.right_turns = self.get_all_turns()
        self.route = []
        self.collided = False
        self.source_loc = None
        self.final_dest = None
        self.sensor_active= False
        self.auto_control = False
        self.sensors = []
        self.stop = False
        self.spawn_vehicle()
        self.wps_close_to_traffic_lights = self.get_waypoints_close_to_traffic_lights()

        self.debug = debug
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
    def get_prev_loc(self, dest_loc, source_loc, dist=5):
        new_loc = carla.Location(source_loc)
        sign = (source_loc - dest_loc).x / abs((source_loc - dest_loc).x)
        new_loc.x += sign * dist
        return new_loc
    
    def _cmp_wp(self, wp0, wp1): return wp0.transform.location == wp1.transform.location
    def get_angle(self, orientation): return orientation - 360 if orientation > 180 else orientation + 360 if orientation < -180 else orientation 
    
    def get_angle_normalised(self, angle):
        return self.get_angle(self.get_angle(angle))

    def get_angle_between(self, obj0, obj1):
        angle0 = -1
        angle1 = -1
        if type(obj0) == carla.Waypoint:
            angle0 = self.get_angle(obj0.transform.rotation.yaw)
            angle1 = self.get_angle(obj1.transform.rotation.yaw)
        else:
            angle0 = self.get_angle(obj0.rotation.yaw)
            angle1 = self.get_angle(obj1.rotation.yaw)

        return self.get_angle_normalised(angle0 - angle1)
    def get_waypoints_close_to_traffic_lights(self):
        wps = self.w.get_map().generate_waypoints(2)
        vehicle_bp = self.w.get_blueprint_library().filter("*vehicle*")[0]

        dummy_car = self.autocar
        traffic_light_wps = []
        traffic_lights = []
        i = 0
        for wp in wps:
            dummy_car.set_transform(wp.transform)
            t_light  =dummy_car.get_traffic_light()
            if t_light is None:
                continue

            relative_angle = self.get_angle_between(dummy_car.get_transform(), t_light.get_transform())
            
            i += 1
            
            if  dummy_car.is_at_traffic_light():

                if relative_angle < -85 and relative_angle > -95:
                    pass 
                
                else:
                    continue
                #self.w.debug.draw_string(t_light.get_transform().location, f"light {i}", life_time = 300)
                #self.w.debug.draw_string(wp.transform.location, f"car {i}\n angle = {self.get_angle_between(wp.transform, t_light.get_transform())}", life_time = 300)
            
            
                    
                traffic_light_wps.append(wp)
            
          
                #traffic_lights.append(t_light)
            
        

        return traffic_light_wps
    def get_all_turns(self):
        
        wps =  self.w.get_map().get_topology()
        locs = []
        left_locs = []
        right_locs = []

        for p0, p1 in wps:
            print(p0.transform.rotation.yaw - p1.transform.rotation.yaw )
            orientation= self.get_angle(self.get_angle(p0.transform.rotation.yaw) - self.get_angle(p1.transform.rotation.yaw ))   
            print(self.get_angle(p0.transform.rotation.yaw) - self.get_angle(p1.transform.rotation.yaw ))
            print(orientation)
           
            print()
            tolerance = 5
            if abs(orientation) >= 90 + tolerance or abs(orientation) < 30:
                print("error")
                continue
            if orientation > 0:
                
                dup = any([self._cmp_wp(pair.start, p0) for pair in left_locs])
                if not dup:
                    left_locs.append(StartEndPair(p0, p1))
            elif orientation < 0:

                dup = any([self._cmp_wp(pair.start, p0) for pair in right_locs])
                if not dup:    
                    right_locs.append(StartEndPair(p0, p1))
                
    
        return left_locs, right_locs
                
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
    def set_turn_start_end(self, turn):
        while True:
            try:
                startend = random.choice(self.left_turns if turn == 3 else self.right_turns)
                self.autocar.set_transform(startend.start.transform)
                self.source_loc = startend.start.transform.location
                self.target_loc = startend.end.transform.location
                #might collide with ground
                time.sleep(0.1)
                self.collision_timer = None
                self.sensor_active = True
                #self.w.debug.draw_string(self.target_loc, "turn end!!!!!!!!!!!!!!!!!", life_time=10)
                
                print(f"{'left' if turn == 3 else 'right'} turn initialised successfully")
                self.current_direction = turn

                break
            except:
                pass
    #generate a non-circular route, currently working
    def get_all_vehicles(self):
        vehicles = [actor for actor in list(filter(
                lambda x: (isinstance(x, carla.Vehicle) and x.id != self.autocar.id),self.w.get_actors()))]
        return vehicles
    @property
    def SAMPLE_LIM(self):
        return NUM_SAMPLES_PER_COMMAND_PER_ITER if self.training else BENCHMARK_LIMIT
    def set_initial_target(self):
        if self.has_collected_enough_turn_samples():
            wp = self.get_wp_from_loc(self.source_loc)
            self.target_loc = self.get_next_wp(wp, 15).transform.location
        # else:
            
        #     start = random.choice(self.wps_close_to_traffic_lights)
        #     end = random.choice(start.next(15))
        #     self.target_loc = end.transform.location
        #     self.source_loc = start.transform.location
        #     self.autocar.set_transform(start.transform)
        #     time.sleep(0.1)
        #     self.collision_timer = None
        #     self.sensor_active = True
            
        elif self.counters[1] < self.SAMPLE_LIM:
            self.set_turn_start_end(3)
        elif self.counters[2] < self.SAMPLE_LIM:
            self.set_turn_start_end(4)
       
           
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

        self.controller = VehiclePIDController(self.autocar, args_lateral = {'K_P': 1, 'K_D': 0.0, 'K_I': 0}, args_longitudinal = {'K_P': 1, 'K_D': 0.0, 'K_I': 0.0})

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
    def get_random_start_point_for_turns(self):
        pair = None
        if self.counters[1] < self.SAMPLE_LIM:
            pair = random.choice(self.left_turns)
        elif self.counters[2] < self.SAMPLE_LIM:
            pair = random.choice(self.right_turns)
        else:
            raise Exception()
        return pair
    
    def has_collected_enough_traffic_light_samples(self):
        #half divided between follow lane w/ traffic light and w/o
        return True 
        return self.traffic_light_counter[0] >= NUM_SAMPLES_PER_COMMAND_PER_ITER * 0.5


    def teleport(self):

        #10% should be at a traffic light
        start = None
        
        if self.has_collected_enough_turn_samples():
            
            self.current_direction = 2
            if not self.has_collected_enough_traffic_light_samples():
                start = random.choice(self.wps_close_to_traffic_lights)

            else:
                start_transform = random.choice(self.sps)
                start = self.get_wp_from_loc(start_transform.location)
                

            end = random.choice(start.next(15))

            angle = self.get_angle_between(start, end)
            assert abs(angle) < 95
            
            self.current_direction = 2 

            self.source_loc = start.transform.location
            self.target_loc = end.transform.location
            self.autocar.set_transform(start.transform)

        elif self.counters[1] < self.SAMPLE_LIM:
            self.set_turn_start_end(3)
        elif self.counters[2] < self.SAMPLE_LIM:
            self.set_turn_start_end(4)
        time.sleep(0.1)
        self.sensor_active = True
        self.collision_timer = None
        # loc = random.choice(self.sps).location
        # self.source_loc = self.get_wp_from_loc(loc).transform.location
        # trans = carla.Transform(self.source_loc)
        # self.autocar.set_transform(trans)
    
    def _reset(self):
        '''initialise variables'''
        self.sensor_active = False
        self.traffic_light = None
        self.reached_traffic_light = False
        self.collision_timer = None
        self.distance_travelled = 0
        self.route_wp_counter = 0
        self.teleport()
        #self.generate_loop()
        
        while self.front_camera is None or self.left_camera is None or self.right_camera is None:
            
            time.sleep(0.01)
        print("cameras active!")        
        #self.sensor_active = True
        ##################reverse timer logic####################
        self.total_dist_travelled = 0
    
        #####################################
        #need to wait before camera can receive sensor (otherwise throttle is 0 and 
        # agent will get confused)
        #start counting
    def has_collected_enough_turn_samples(self):
        
        return not any([num < 0 for num in self.counters[1:]])
    
    def spawn_vehicles(self):
        bplib = self.w.get_blueprint_library()
        vehicles_bps = bplib.filter("*vehicle*")
        walkers_bps = bplib.filter("*walker*")
        num_walkers = 0
        num_cars = 0
        while num_cars < 20:
            try:
                wp = self.get_wp_from_loc(random.choice(self.sps).location)
                wp = random.choice(wp.next(random.randint(0, 10)))
                
                vehicle = self.w.spawn_actor(random.choice(vehicles_bps), wp.transform)
                
                vehicle.set_autopilot(True)
                num_cars += 1
            except:
                continue

        while num_walkers < 10:
            try:
                walker_controller_bp = self.w.get_blueprint_library().find('controller.ai.walker')
                wp = self.get_wp_from_loc(random.choice(self.sps).location)
                wp = random.choice(wp.next(random.randint(0, 10)))
                self.w.SpawnActor(walker_controller_bp, wp.transform, random.choice(walkers_bps))
                num_walkers += 1
            except:
                continue
            
    @property 
    def TARGET_TOLERANCE(self): return 3 if self.has_collected_enough_turn_samples() else 1
    def reset(self):
        self._reset()
    
        # initial_transform = self.autocar.get_transform()

        # init_dist = self.autocar.get_location().distance(self.target_loc)
        #guide vehicle to drive in the right direction
        done = False

        #self.set_initial_target()

        if not self.has_collected_enough_turn_samples():

            done = True
        
            #next wp is guaranteed to be oriented in the right directon
            # while True:
            #     reached_dest = self.autocar.get_location().distance(self.current_target_wp.transform.location) < TARGET_TOLERANCE
            #     while not reached_dest:
            #         reached_dest = self.autocar.get_location().distance(self.current_target_wp.transform.location) < TARGET_TOLERANCE
            #         self.view_spectator_fps()

            #         if self.collision_timer is not None:
            #             self.collision_timer = None

            #             break
            #         print(self.autocar.get_location().distance(self.target_loc))
            #         self.autocar.apply_control(self.controller.run_step(TARGET_SPEED, self.current_target_wp))    
            #     if reached_dest:
            #         break 
            #     self.teleport()
                #self.set_initial_target()
            #self.update_target()

        print("finished guiding process!")
        self.collision_timer = None
        self.sensor_active = True
        self.waypoint_timer = time.time() 
        return ((self.front_camera, self.left_camera, self.right_camera), self.get_speed(), self.current_direction), done

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

    @property
    def current_wp(self): return self.get_wp_from_loc(self.get_current_location())

    @property
    def orientation_wrt_road(self):
        
        return self.get_angle(self.current_wp.transform.rotation.yaw - self.autocar.get_transform().rotation.yaw)

    def reached_dest(self):
        return self.get_current_location().distance(self.target_loc) < self.TARGET_TOLERANCE
    @property
    def distance_from_lane_edge(self):
        
        lw = self.current_wp.lane_width
        dist = abs((self.current_wp.transform.location - self.get_current_location()).y)
        return lw / 2 - dist
        
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
            dir = self.calculate_turn_direction(self.target_loc - self.source_loc, prev_dir)
            
        else:
            self.total_dist_travelled += self.target_loc.distance(self.source_loc)
            self.route_wp_counter = (self.route_wp_counter + 1) % len(self.route)
            self.source_loc = self.target_loc

            self.target_loc = self.route[self.route_wp_counter].transform.location
        self.target_updated = True
        if self.target_loc is None:
            return
            
    #    self.w.debug.draw_string(self.target_loc, "next target", life_time=10)

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
        
        if time.time() - self.waypoint_timer > WAYPOINT_TIMEOUT:
            if self.traffic_light is not None:
                if self.light_is_red() and time.time() - self.waypoint_timer > self.traffic_light.get_red_time():
                    return True
                else:
                    return False
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
        self.target_updated = True
        

    def vehicle_deadlock(self):
        if self.get_speed() > 0:
            return False
        for vehicle in self.get_all_vehicles():

            if vehicle.get_location().distance(self.autocar.get_location()) <= 5:
                v = vehicle.get_velocity()
                if v.x == 0 and v.y == 0:
                    if abs(self.get_angle(self.get_angle(vehicle.get_transform().yaw) - self.get_angle(self.autocar.get_transform().yaw))) >= 90:
                        return True
        return False
    def light_is_red(self):
        if self.traffic_light != None:
            return self.traffic_light.state == carla.TrafficLightState.Red
        return None
    
    def turn_made(self):
        angle = self.current_target_wp.transform.rotation.yaw - self.autocar.get_transform().rotation.yaw
            
        relative_angle = abs(self.get_angle_normalised(angle))
        print(relative_angle)
        return relative_angle <= 5 and not self.has_collected_enough_turn_samples()
    def run_step(self, control):
        
        self.view_spectator_birds_eye()
        
        done = False
        if self.collision_timer is not None and (time.time() - self.collision_timer >= COLLISION_TIMEOUT):
            self.collision_timer = None

            print("collided")
            if self.get_speed() < MIN_SPEED:
                done = True
        
        elif self.collision_timer is None and self.timedout() :
            print("timed out, replanning route")
            if not self.has_collected_enough_turn_samples():
                done = True
            else:
                self.target_updated = True
                self.reset_source_and_target()
                self.collision_timer = None
                #self.sensor_active = True
                if self.get_speed() == 0 and time.time() - self.waypoint_timer > WAYPOINT_TIMEOUT *2:
                    done = True
            #self.w.debug.draw_string(self.target_loc, "replanned target", life_time=10)
        
        elif self.get_current_location().distance(self.target_loc) < self.TARGET_TOLERANCE or not self.training and self.turn_made():
            self.total_dist_travelled += self.target_loc.distance(self.source_loc)
            
            
            self.waypoint_timer = time.time()
            if not self.training:
                self.counters[self.current_direction - 2] += 1
                
            if self.target_loc is None or not self.has_collected_enough_turn_samples():
                done = True
                print("reached fin dest, resetting...")
            else:
                
                self.target_updated = True
                
                if not self.has_collected_enough_traffic_light_samples():
                    print(self.autocar.get_traffic_light())
                    at_t_light = self.autocar.get_traffic_light() != None
                    if at_t_light and not self.reached_traffic_light:
                        self.reached_traffic_light = True
                        self.traffic_light = self.autocar.get_traffic_light()
                        
                        self.w.debug.draw_string(self.autocar.get_location(), "\n\reached light", life_time=3)
                        print("reached light")
                    elif self.reached_traffic_light and self.autocar.get_traffic_light() != self.traffic_light:
                        self.reached_traffic_light = False
                        self.traffic_light = None
                        self.w.debug.draw_string(self.autocar.get_location(), "\n\npassed light", life_time=3)
                        print("passed light")
                        done = True
                
                
                if self.has_collected_enough_turn_samples() and len(self.get_wp_from_loc(self.target_loc).next(10)) > 1:
                    
                    done = True
                else:
                    self.update_target()
        if self.over_turned():
            if self.has_collected_enough_turn_samples():
                
                self.target_updated = True
                self.update_target()
                print("over turned")
        if not self.has_collected_enough_turn_samples():
            angle = self.current_target_wp.transform.rotation.yaw - self.autocar.get_transform().rotation.yaw
            
            relative_angle = abs(self.get_angle_normalised(angle))
            if relative_angle >135:
                done = True
            
        s = round(control.steer, 2)
        t = round(control.throttle, 2)
        b = round(control.brake, 2)
#        self.w.debug.draw_string(trans_car.location, f"steer: {s}\n throttle: {t}\n brake: {b}", life_time=1.5)

        self.autocar.apply_control(control)
        return ((self.front_camera, self.left_camera, self.right_camera), self.get_speed(), self.current_direction), done
    @property
    def current_dir_str(self):
        return "follow" if self.current_direction == 2 else "left" if self.current_direction == 3 else "right"
    
    def view_spectator_fps(self):
        trans_car = self.autocar.get_transform()
        
        trans_car.location.z += 2
    
        #self.w.debug.draw_string(trans_car.location, "vehicle", life_time=0.1)
        self.spectator.set_transform(trans_car)
        control = self.autocar.get_control()
        s = round(control.steer, 2)
        t = round(control.throttle, 2)
        b = round(control.brake, 2)
        test_wp=None
        relative_angle = 999
        current_wp = self.current_wp
        
        min_angle = 999
        for wp in current_wp.next(0.5):
            if abs(self.get_angle_between(current_wp, wp)) < min_angle:
                min_angle = abs(self.get_angle_between(current_wp, wp))
                test_wp = wp
    
        
        message_loc = self.autocar.get_location()
        loc_diff = test_wp.transform.location - self.autocar.get_location()
        loc_diff.x *= 20
        loc_diff.y *= 20
        relative_angle = self.get_angle_between(self.autocar.get_transform(), test_wp.transform)
        loc_diff.x = loc_diff.x * math.cos(relative_angle)
        loc_diff.y = loc_diff.y * math.sin(relative_angle)
        message_loc += loc_diff
        
        
        self.w.debug.draw_string(message_loc, f"s={s}, t={t}, b={b}\n {self.current_dir_str}", life_time=0.05)
      
    def view_spectator_birds_eye(self):
        trans_car = self.autocar.get_transform()
        trans_car.rotation.pitch -= 90
        trans_car.location.z += 20
    
        #self.w.debug.draw_string(trans_car.location, "vehicle", life_time=0.1)
        self.spectator.set_transform(trans_car)
        control = self.autocar.get_control()
        s = round(control.steer, 2)
        t = round(control.throttle, 2)
        b = round(control.brake, 2)
       
        self.w.debug.draw_string(self.target_loc, f"target", life_time=3)

        self.w.debug.draw_string(self.get_current_location(), f"s={s}, t={t}, b={b}\n {self.current_dir_str}")
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
        if sensor_id == 0:
           
            cv2.imshow("front cam", self.front_camera)
            #cv2.imshow("left cam", self.left_camera)
            #cv2.imshow("right cam", self.right_camera)
            cv2.waitKey(1)

    def process_collision(self, event):
        if self.collision_timer is not None:
            return
        self.collision_timer = time.time()
        
        imp = event.normal_impulse

        print("collision occured")
        self.sensor_active = False
    
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