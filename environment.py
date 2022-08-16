
import random
import sys

import cv2
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
    def __init__(self, counter, traffic_light_counter, training=True, port=2000, debugg=False, enable_fast_simulation=False, use_baseline_agent=False, skip_turn_samples=False):
        from sys import path 
        
        self.debug = debugg
        self.missed_turns_force_respawn_counter = 0
        self.use_baseline = use_baseline_agent
        self.counters = counter
        self.skip_turn_samples = skip_turn_samples
        self.guideline_control = None
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
        self.source_loc = None
        self.final_dest = None
        self.sensor_active= False
        self.auto_control = False
        self.sensors = []
        self.stop = False
        self.spawn_vehicle()
    
        self.wps_close_to_traffic_lights = None #self.get_waypoints_close_to_traffic_lights()

        print("environment initialised!")
    
        #obstacle_sensor = self.w.spawn_actor(obstacle_bp, carla.Transform(carla.Location()), attach_to=autocar)
        #obstacle_sensor.listen(lambda event  : self.process_obstacle(event))
    def traffic_light_violated(self):
        return self.autocar.is_at_traffic_light() and self.autocar.get_traffic_light().state == carla.TrafficLightState.Red and self.get_speed() > 0
    def get_angle_fwp_wrt_twp(self):
        if len(self.current_wp.next(DIST_BETWEEN_WPS)) == 1:

            f_wp = self.current_wp.next(DIST_BETWEEN_WPS)[0]
            future_loc = f_wp.transform.location

            future_car_vec = future_loc - self.autocar.get_location()
            target_loc = self.target_wp.transform.location
            target_car_vec = target_loc - self.autocar.get_location()
            angle = abs(self.calculate_angle_between_vec(future_car_vec, target_car_vec))
            
            return round(angle, 2)
        return "undefined angle" 
    def missed_turn(self):
        
        target_loc = self.target_wp.transform.location
        target_car_vec = target_loc - self.autocar.get_location()
        angle = abs(self.calculate_angle_between_vec(self.autocar.get_transform().rotation.get_forward_vector(), target_car_vec))
        if angle >= 110:
            
            self.angle = angle
            #vehicle missed a turn 
            return True
        return False
    def over_turned(self):
        #inverse of a math coordinate system
        #cannot teach the vehicle to turn around the road (this means the agent
        # will sometimes get confused so starts to make U turns!)
        
        car_loc = self.get_current_location()
        car_dir = self.target_loc - car_loc

        target_dir = self.target_loc - self.source_loc
        angle = self.calculate_angle_between_vec(car_dir, target_dir)
        if abs(angle) > 80:
            
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
        #if angle1 is -ve (meaning left)
        #then angle0 - angle1 will be positive which is the condition for left
        return self.get_angle_normalised(angle0 - angle1)
    def dist_between_transform(self, t0, t1):
        
        return t0.location.distance(t1.location)
    def get_waypoints_close_to_traffic_lights(self):
        
        vehicle_bp = self.w.get_blueprint_library().filter("*vehicle*")[0]

        dummy_car = self.autocar
        traffic_light_wps = []
        traffic_lights = []
        i = 0
    
        wps = self.w.get_map().generate_waypoints(1)
        
        for wp in wps:
            dummy_car.set_transform(wp.transform)
            t_light  =dummy_car.get_traffic_light()
            if t_light is None:
                continue

            relative_angle = self.get_angle_between(dummy_car.get_transform(), t_light.get_transform())
            if relative_angle > -88 :
                continue
            
                
            traffic_light_wps.append(wp)

                    # if relative_angle < -85 and relative_angle > -95:
                    #     pass 
                    
                    # else:
                    #     continue
                    # #self.w.debug.draw_string(t_light.get_transform().location, f"light {i}", life_time = 300)
                    # #self.w.debug.draw_string(wp.transform.location, f"car {i}\n angle = {self.get_angle_between(wp.transform, t_light.get_transform())}", life_time = 300)
                
                
                        
                    # traffic_light_wps.append(wp)
                
            
                    #traffic_lights.append(t_light)
                
            

        return traffic_light_wps
    def get_all_turns(self):
        
        wps =  self.w.get_map().get_topology()
        locs = []
        left_locs = []
        right_locs = []

        for p0, p1 in wps:
            (p0.transform.rotation.yaw - p1.transform.rotation.yaw )
            orientation= self.get_angle(self.get_angle(p0.transform.rotation.yaw) - self.get_angle(p1.transform.rotation.yaw ))   
            (self.get_angle(p0.transform.rotation.yaw) - self.get_angle(p1.transform.rotation.yaw ))
            (orientation)
           
            ()
            tolerance = 5
            if abs(orientation) >= 90 + tolerance or abs(orientation) < 30:
                ("error")
                continue
            if orientation > 0:
                
                dup = any([self._cmp_wp(pair.start, p0) for pair in left_locs])
                p1 = random.choice(p1.next(5))
                if not dup:
                    left_locs.append(StartEndPair(p0, p1))
            elif orientation < 0:
                
                dup = any([self._cmp_wp(pair.start, p0) for pair in right_locs])
                p1 = random.choice(p1.next(5))

                if not dup:    
                    right_locs.append(StartEndPair(p0, p1))
                
    
        return left_locs, right_locs
                
    def calculate_angle_between_vec(self, vec2, vec1):
        x2 = vec2.x
        x1 = vec1.x

        y2 = vec2.y
        y1 = vec1.y
  
        dot = x1 * x2 + y1 * y2
        det = x1 * y2 - y1 * x2

        return math.degrees(math.atan2(det, dot))

    
    def calculate_turn_direction(self, current_wp, next_wp):
        
        #+ve is left
        #-ve is right
        angle = self.get_angle_between(current_wp, next_wp)

        # if 2 * math.pi >angle > math.pi:
        #     angle = -(2 * math.pi - angle)
        self.sensor_active = True

        (f"angle = {angle / math.pi * 180}")
        if 35 / 180 * math.pi > angle > -35/180*math.pi:
            ("following")
            return 5 if self.use_baseline else 2
    
        elif 35/180*math.pi <= angle:
        
            return 3
        elif -35/180*math.pi >= angle:
        
            return 4
        else:
            ("turn undefined")
            self.sensor_active = False
            return 2
            #("undefined")

    def calc_turn_dir(self, current_dir_vector, previous_dir_vector):
        
        #+ve is left
        #-ve is right
        angle = self.calculate_angle_between_vec(current_dir_vector, previous_dir_vector)

        # if 2 * math.pi >angle > math.pi:
        #     angle = -(2 * math.pi - angle)
        self.sensor_active = True

        (f"angle = {angle / math.pi * 180}")
        if 5 / 180 * math.pi > angle > -5/180*math.pi:
            ("following")
            return "straight"
        
        #turn right
        elif -5/180*math.pi >= angle:
            ("turning left")
            return "left"

        elif 5/180*math.pi <= angle:
            ("turning right")
            return "right"
        else:
            ("turn undefined")
            self.sensor_active = False
            return "straight"
            #("undefined")
    def set_dest(self): pass

    #generate a loop route, currently not working
    def generate_loop(self):
        #these two lines of code is necessary before calling form_loop,
        #which returns a list of waypoints to follow
        wp = self.get_wp_from_loc(self.source_loc)
        #set_target(wp)
        #set_world(self.w)

        self.route, dist = form_loop(wp, [], 0)
        (f"loop is {dist} m long")
        self.target_loc = self.route[0].transform.location
        count = 2
        for wp in self.route[:-2]:
            self.w.debug.draw_string(wp.transform.location, f"{count}", life_time=60)
            count += 1
        self.w.debug.draw_string(wp.transform.location, f"{1}", life_time=60)
    def in_proximity(self, loc):
        vehicles = self.get_all_vehicles()
        for vehi in vehicles:
            if vehi.get_location().distance(loc) <= 8:
                return True 
        return False
    def set_turn_start_end(self, turn):
        while True:
            try:
                startend = random.choice(self.left_turns if turn == 3 else self.right_turns)
                if self.in_proximity(startend.start.transform.location):
                    continue
                self.autocar.set_transform(startend.start.transform)
                self.source_loc = startend.start.transform.location
                self.target_loc = startend.end.transform.location
                #might collide with ground
                time.sleep(0.1)
                self.collision_timer = None
                self.sensor_active = True
                #self.w.debug.draw_string(self.target_loc, "turn end!!!!!!!!!!!!!!!!!", life_time=10)
                
                (f"{'left' if turn == 3 else 'right'} turn initialised successfully")
                self.current_direction = turn

                break
            except Exception as e:
                pass
    #generate a non-circular route, currently working
    def get_all_vehicles(self):
        vehicles = [actor for actor in list(filter(
                lambda x: (isinstance(x, carla.Vehicle) and x.id != self.autocar.id),self.w.get_actors()))]
        return vehicles
    @property
    def SAMPLE_LIM(self):
        return DEBUG_NUM_SAMPLES_PER_COMMAND_PER_ITER \
            if self.debug and self.training \
                else NUM_SAMPLES_PER_COMMAND_PER_ITER \
                    if self.training and not self.debug \
                        else DEBUG_BENCHMARK_LIMIT \
                            if not self.training and self.debug \
                                else BENCHMARK_LIMIT
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
        #     ("generating route...")
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
    
        
    def spawn_vehicle(self, ignore_lights = True, spawn_trans=None):
        if self.autocar is not None and self.autocar.is_alive:
            
            self.destroy_car()
        
        model_3 = self.w.get_blueprint_library().filter("vehicle.ford.mustang")[0]
        if spawn_trans is None:
            while True:
                i = 0
                try:
                    
                    trans = random.choice(self.sps)
                    #trans = self.get_wp_from_loc(trans.location).transform
                    #wp = self.get_wp_from_loc(trans.location)
                    self.source_loc = trans.location
                    #self.source_loc = wp.transform.location
                    if self.autocar is None:
                        if self.in_proximity(self.source_loc):
                            continue
                        self.autocar = self.w.spawn_actor(model_3, trans) 
                    else:
                        self.autocar.set_transform(trans)

                    break
                except Exception as e: 
                
                    (f"attempting to spawn {i}")
                    i+=1
        else:
            start_timer = time.time()
            
            assert not self.autocar.is_alive
            self.autocar = self.w.spawn_actor(model_3, spawn_trans)
            if not self.autocar.is_alive:
                raise Exception("cannot respawn vehicle!")
        self.controller = VehiclePIDController(self.autocar, args_lateral = {'K_P': 1, 'K_D': 0.0, 'K_I': 0}, args_longitudinal = {'K_P': 1, 'K_D': 0.0, 'K_I': 0.0})

        #sp.location += (carla.Location(x=0, y=-5))
        
        #self.tm.ignore_lights_percentage(self.autocar, 100 if ignore_lights else 0)
        #self.tm.auto_lane_change(self.autocar,True)
        l, wid, h = self.car_dim_info(self.autocar)
        bplib = self.w.get_blueprint_library()
        collision_sensor_bp = bplib.find("sensor.other.collision")
        rgb_cam_bp = bplib.find("sensor.camera.rgb")
        rgb_cam_bp.set_attribute("image_size_x", f"{(IM_WIDTH if not self.use_baseline else COIL_IM_WIDTH)}")
        rgb_cam_bp.set_attribute("image_size_y", f"{(IM_HEIGHT if not self.use_baseline else COIL_IM_HEIGHT)}")
        rgb_cam_bp.set_attribute("fov", "110")
        

        l, w, h = self.car_dim_info(self.autocar)
        loc = carla.Location(x=l*3/4,z=1.25)#z=2.1*h)
        self.camera = self.w.spawn_actor(rgb_cam_bp, carla.Transform(loc), attach_to=self.autocar)
        self.left_camera = self.w.spawn_actor(rgb_cam_bp, carla.Transform(location=loc, rotation=carla.Rotation(yaw=-45)), attach_to=self.autocar)
        self.right_camera = self.w.spawn_actor(rgb_cam_bp, carla.Transform(location=loc, rotation=carla.Rotation(yaw=45)), attach_to=self.autocar)
        obs_bp = self.w.get_blueprint_library().find('sensor.other.obstacle')
        obs_bp.set_attribute("distance", str(8))
        obs_location = carla.Location(0,0,0)
        obs_rotation = carla.Rotation(0,0,0)
        obs_transform = carla.Transform(obs_location,obs_rotation)
        
        lane_inv_bp = bplib.find("sensor.other.lane_invasion")
        self.lane_inv_sensor = self.w.spawn_actor(lane_inv_bp, obs_transform, attach_to=self.autocar)

        self.obs_sensor = self.w.spawn_actor(obs_bp, obs_transform, attach_to=self.autocar)
        
        #self.w.debug.draw_string(loc + self.autocar.get_location(), "x", life_time=60)
        self.collision_sensor = self.w.spawn_actor(collision_sensor_bp, carla.Transform(carla.Location(x=2.5, z=0.7)), attach_to=self.autocar)
        self.sensors = [self.camera, self.left_camera, self.right_camera, self.collision_sensor, self.obs_sensor, self.lane_inv_sensor]
        self.obs_sensor.listen(self.process_obs)
        self.collision_sensor.listen(lambda event : self.process_collision(event))
        self.traffic_jam = False
        self.camera.listen(lambda event : self.process_img(event, 0))
        self.left_camera.listen(lambda event :  self.process_img(event, 1))
        self.right_camera.listen(lambda event :  self.process_img(event, 2))
        self.lane_inv_sensor.listen(self.process_lane_inv)
        self.lane_invaded = False 
        self.on_lane = True
        self.lane_id = None 
        #self.left_camera.listen(self.process_img)
    


        # while num_actors < 20:
            
        #     actor = self.w.try_spawn_actor(random.choice(v_bps), random.choice(self.sps))
        #     if actor is not None:
        #         actor.set_autopilot(True, self.tm.get_port())
        #         num_actors += 1
    def process_lane_inv(self, event): 
        # print( [marking.lane_change for marking in event.crossed_lane_markings])
        #can only be on lane after the timer has been exceeded
        if self.on_lane:
            self.on_lane = False
            self.last_lane_invasion_timer = time.time()
            print("lane invaded!")
    def process_obs(self, event):
        if isinstance(event.other_actor, carla.Vehicle):
            v = event.other_actor.get_velocity()
            if v.x == 0 and v.y == 0 and self.get_wp_from_loc(event.other_actor.get_location()).lane_id == self.target_wp.lane_id and\
                abs(self.get_angle_between(self.autocar.get_transform(), event.other_actor.get_transform())) < 20:
                print("obstacle detected!")

                self.traffic_jam = True
        self.traffic_jam = False
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

    def guide(self):
        #to be used only for going straight
        assert self.current_direction == 2
        assert self.target_loc != None
        assert self.autocar is not None
        if abs(self.get_angle_between(self.autocar.get_transform(), self.target_wp.transform)) <= 95:
            ("no need to guide")
            return
        while not self.reached_dest():
            self.controller.run_step(TARGET_SPEED, self.target_wp)
            if self.collision_timer is not None:
                self._reset()
        self.update_target()
    
    def round_angle(self, angle, tolerance=5):
        
        lower_bound = angle // 5 * 5
        upper_bound = angle + 5 if angle > 0 else angle - 5
        return lower_bound if abs(angle) - abs(lower_bound) < abs(upper_bound) - abs(angle) else upper_bound
    def teleport(self):

        #10% should be at a traffic light
        start = None
        end = None
        if self.has_collected_enough_turn_samples():
            
            
            if not self.has_collected_enough_traffic_light_samples():
                start = random.choice(self.wps_close_to_traffic_lights)
                end = random.choice(start.next(DIST_BETWEEN_WPS))
                angle = self.get_angle_between(start, end)
                if angle >= 35 and angle <= 100:
                    self.current_direction = 3
                elif angle <= -35 and angle >= -100:
                    self.current_direction = 4 
                elif abs(angle) < 35:
                    if self.use_baseline:
                        self.current_direction = 5
                    else:
                        self.current_direction = 2
                else:
                    ("error determining direction")
                    self.current_direction = 2

            else:
                found = False
                while not found:
                    
                    random.shuffle(self.sps)
                    for trans in self.sps:
                        start = self.get_wp_from_loc(trans.location)
                        for _end in start.next(DIST_BETWEEN_WPS):
                            if abs(self.get_angle_between(start, _end)) < 1 and\
                                not self.in_proximity(start.transform.location):
                                found = True
                                self.current_direction = 2
                                end = _end
                                break
                    if not found:
                        raise Exception("cannot find straight spawn pts")
                

            #angle = self.get_angle_between(start, end)
            #assert abs(angle) < 95
            
            self.current_direction = 2 
            
            self.source_loc = start.transform.location
            self.target_loc = end.transform.location
            self.autocar.set_transform(start.transform)
            ("vehicle teleported")


        
        elif self.counters[2] < self.SAMPLE_LIM:
            self.set_turn_start_end(4)
        elif self.counters[1] < self.SAMPLE_LIM:
            self.set_turn_start_end(3)
        time.sleep(0.1)

        self.lane_id = self.current_wp.lane_id

        self.sensor_active = True
        self.collision_timer = None
        # loc = random.choice(self.sps).location
        # self.source_loc = self.get_wp_from_loc(loc).transform.location
        # trans = carla.Transform(self.source_loc)
        # self.autocar.set_transform(trans)
    
    def _reset(self):
        '''initialise variables'''
        self.sensor_active = False
        self.one_lane = True
        self.last_lane_invasion_timer = -1
        self.npc_car = None
        self.traffic_light = None
        self.reached_traffic_light = False
        self.collision_timer = None
        self.distance_travelled = 0
        self.route_wp_counter = 0
        self.was_at_traffic_light = False
        self.missed_turns_force_respawn_counter = 0
        self.tlight = None
        self.dist_to_tlight = 9e9
        #self.generate_loop()
        
        
        ("cameras active!")        
        #self.sensor_active = True
        ##################reverse timer logic####################
    
    
        #####################################
        #need to wait before camera can receive sensor (otherwise throttle is 0 and 
        # agent will get confused)
        #start counting
    def reached_light(self):
        return self.autocar.is_at_traffic_light()
    def has_collected_enough_turn_samples(self):
        if not self.skip_turn_samples:
            return not any([num < self.SAMPLE_LIM for num in self.counters[1:]])
        return True

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
    '''cannot have a large tolerance for turns (can crash into side of road!)'''
    @property 
    def TARGET_TOLERANCE(self):
        return 2 #return 3 if self.has_collected_enough_turn_samples() else 1
    def reset(self):
        self._reset()
        self.teleport()
        while self.front_camera is None or self.left_camera is None or self.right_camera is None:
            
            time.sleep(0.01)
        
        # if self.has_collected_enough_turn_samples():
        #     self.guide()
        # initial_transform = self.autocar.get_transform()

        # init_dist = self.autocar.get_location().distance(self.target_loc)
        #guide vehicle to drive in the right direction
        done = False

        #self.set_initial_target()

        if not self.has_collected_enough_turn_samples():
            done = True

        self.collision_timer = None
        self.sensor_active = True
        self.waypoint_timer = time.time() 
        if self.current_direction == 3 or self.current_direction == 4:
            self.waypoint_timer += 2

        return ((self.front_camera, self.left_camera, self.right_camera), self.get_speed(), self.current_direction), done

    def cleanup(self):

        try:
            self.cl.apply_batch([carla.command.DestroyActor(x) for x in [actor for actor in list(filter(
                lambda x: isinstance(x, carla.Sensor) and (isinstance(x.parent, carla.Vehicle) or x.parent is None), self.w.get_actors()))]])
            
            self.cl.apply_batch([carla.command.DestroyActor(x) for x in [actor for actor in list(filter(
                lambda x: isinstance(x, carla.Vehicle), self.w.get_actors()))]])
             
            ("clean up successful")
        except Exception as e:
            ("cannot destroy all actors")
            (e)
    
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
        (asizeof.asizeof([self.front_camera, spd, self.current_direction, control]))
        return [self.front_camera, self.left_camera, self.right_camera, self.get_speed(), self.current_direction, control, left_control, right_control]

    @property
    def current_wp(self): return self.get_wp_from_loc(self.get_current_location())

    @property
    def orientation_wrt_road(self):
        
        return self.get_angle_normalised(self.current_wp.transform.rotation.yaw, self.autocar.get_transform().rotation.yaw)

    def reached_dest(self):
        return self.get_current_location().distance(self.target_loc) < self.TARGET_TOLERANCE
    
    

    @property
    def distance_from_lane_edge(self):
        
        
        # nearest_quadrant = round(angle / 90, 0) * 90
        current_wp = self.current_wp
        if current_wp.lane_id != self.source_wp.lane_id:
            current_wp = current_wp.get_right_lane()

            if current_wp is None or current_wp.lane_id != self.source_wp.lane_id:
                
                current_wp = self.current_wp
            
                print("error")
        
        lw = current_wp.lane_width
    
        dist = self.dist_between_transform(current_wp.transform, self.autocar.get_transform())        
        return lw / 2 - dist
    def get_dist_between_src_dest(self):
        return self.dist_between_transform(self.source_wp.transform, self.target_wp.transform)

    def get_dist_from_source_wp(self):
        return self.dist_between_transform(self.source_wp.transform, self.autocar.get_transform())

    def get_biased_target_if_any(self, current_wp, dist=10):
        
    
        #prev_dir_vec = current_loc - prev_loc 
        wps = current_wp.next(dist)
        use_preferred = random.random() > 0.2
        preferred_wps = [] 
        if not use_preferred:
            return random.choice(wps).transform.location
        for wp in wps:
            if wp.lane_id != current_wp.lane_id:
                print("next waypoint lane id diff")
            #current_dir_vec = wp.transform.location - current_loc
            #direction = self.calculate_turn_direction(current_dir_vec, prev_dir_vec)
            direction = self.calculate_turn_direction(current_wp, wp)
            if self.preferred_direction == direction and self.preferred_direction is not None:
                if direction == 3 or direction == 4:
                    
                    #extend the turn wp
                    wp = wp.next(4)[0]
                preferred_wps.append(wp)
                
        if len(preferred_wps) == 0:
            selection = None
            random.shuffle(wps)
            for wp in wps:
                direction = self.calculate_turn_direction(current_wp, wp)
                if direction == 3 or direction == 4:
                    selection = wp.next(4)[0]
                    break
                else:
                    selection = wp
            return selection.transform.location
        else:
            return random.choice(preferred_wps).transform.location
    def update_target(self):
       
        prev_dir = self.target_loc - self.source_loc
        if self.preferred_direction != None:
            
            next_target = self.get_biased_target_if_any(self.target_wp)
            
            self.source_loc = self.target_loc
            self.target_loc = next_target
            #dir = self.calculate_turn_direction(self.target_loc - self.source_loc, prev_dir)
            
        else:
            self.route_wp_counter = (self.route_wp_counter + 1) % len(self.route)
            self.source_loc = self.target_loc

            self.target_loc = self.route[self.route_wp_counter].transform.location
        self.target_updated = True
        if self.target_loc is None:
            return
            
    #    self.w.debug.draw_string(self.target_loc, "next target", life_time=10)

        current_dir = self.target_loc - self.source_loc
        
        self.current_direction = self.calculate_turn_direction(self.source_wp, self.target_wp)
        

        #("current direction" + ["follow", "left", "right","straight"][self.current_direction - 2])
        self.waypoint_timer = time.time()
        if self.current_direction == 3 or self.current_direction == 4:
            self.waypoint_timer += 2
    def set_turn_bias(self, direction):
        self.preferred_direction = direction

    def get_next_wp(self, wp, dist=10):
        return random.choice(wp.next(dist))
    @property
    def target_wp(self):
        return self.get_wp_from_loc(self.target_loc)
    @property
    def source_wp(self):
        return self.get_wp_from_loc(self.source_loc)
        
    def get_wp_from_loc(self, loc):
        '''only return a wp that is suitable to drive to'''
        return self.w.get_map().get_waypoint(loc, project_to_road=True, lane_type=(carla.LaneType.Driving))

    def get_current_location(self):
        
        return self.autocar.get_location()
    
    def set_guideline_control(self, control):
        self.guideline_control = control
    def timedout(self):
        
        if time.time() - self.waypoint_timer > WAYPOINT_TIMEOUT * ((30 / 12) if self.use_baseline else 1):
            #self.tlight is not None or
            
            if not ( self.traffic_jam):
                
                return True 
            else:
                self.waypoint_timer = time.time()
            
        return False
    def reset_source_and_target(self):
        '''if timed out should be called before resetting the waypoint timer'''
        ("resetting src and target")
        

        current_wp = self.current_wp
        if not self.on_lane:
            if self.current_wp.lane_id != self.target_wp.lane_id and self.current_wp.lane_id != self.source_wp.lane_id:
                current_wp = self.current_wp.get_right_lane()
                
                if current_wp is None:
                    return False
                     
    
        try:
            next_wp = random.choice(current_wp.next(DIST_BETWEEN_WPS))
            self.current_direction = self.calculate_turn_direction(self.current_wp, next_wp)

            self.target_loc = next_wp.transform.location
            # else: #only the noise period
            #     self.w.debug.draw_string(self.autocar.get_location(), "noise reset", life_time=5)
            #     self.target_loc = self.get_biased_target_if_any(self.current_wp, 20)
            #     #self.w.debug.draw_string(self.target_loc, "noise reset", life_time=10)
            #     self.source_loc = self.current_wp.transform.location
            self.waypoint_timer = time.time()
            self.collision_timer = None
            self.target_updated = True
            self.w.debug.draw_string(self.target_loc, "target reset", life_time=5)
        
            return True
        except IndexError as e:
            print(e)
            return False

        
        # if self.get_speed() > 0:
        #     return False
    
        # for vehicle in self.get_all_vehicles():
        #     v = vehicle.get_velocity()
        #     if v.x == 0 and v.y == 0:
    
        #         if abs(self.get_angle_between(self.current_wp.transform, vehicle.get_transform())) <= 5 and\
        #             self.vehicle_in_front(vehicle):
        #             self.npc_car = vehicle
        #             return True
    
        # return False
    def light_is_red(self):
        return self.autocar.get_traffic_light().state == carla.TrafficLightState.Red
        # if self.traffic_light != None:
        #     return self.traffic_light.state == carla.TrafficLightState.Red
        # return None
    
    def turn_made(self):
        angle = self.target_wp.transform.rotation.yaw - self.autocar.get_transform().rotation.yaw
            
        relative_angle = abs(self.get_angle_normalised(angle))
        (relative_angle)
        return relative_angle <= 5
    
    
    
    def run_step(self, control):
        
        self.view_spectator_birds_eye()
        if not self.on_lane and time.time() - self.last_lane_invasion_timer > LANE_INV_TIMEOUT:
            print("on lane again")
            self.on_lane = True
        
        done = False
        if self.collision_timer is not None and (time.time() - self.collision_timer >= COLLISION_TIMEOUT):
            
        
            ("collided")
            
            if self.get_speed() < MIN_SPEED:
                done = True
            else:
                self.collision_timer = None

        #should test for timeout if the collision timeout is exceeded
        elif self.collision_timer is None and self.timedout() :
            if not self.has_collected_enough_turn_samples() or self.get_speed() < MIN_SPEED:
                print("timed out, replanning route")
            
                done = True
            else:
                self.target_updated = True
                self.reset_source_and_target()
                self.collision_timer = None
                #self.sensor_active = True
                if self.get_speed() == 0 and time.time() - self.waypoint_timer > WAYPOINT_TIMEOUT *2:
                    done = True
            #self.w.debug.draw_string(self.target_loc, "replanned target", life_time=10)
        
        elif self.get_current_location().distance(self.target_loc) < self.TARGET_TOLERANCE or self.turn_made():
            
            self.waypoint_timer = time.time()
            if not self.training:
                direction = 2 if self.current_direction == 5 else self.current_direction
                self.counters[direction - 2] += 1
                
            if self.target_loc is None or not self.has_collected_enough_turn_samples():
                done = True
                print("reached fin dest, resetting...")
            else:
                #keep going as we are not training for making turns
                self.target_updated = True
                self.update_target()

        
        if self.autocar.get_traffic_light() is not None and self.tlight is None:
            angle = self.get_angle_between(self.current_wp.transform, self.autocar.get_traffic_light().get_transform()) 
            if not (angle > -88 and angle <= -90) and self.current_wp.lane_id == self.get_wp_from_loc(self.autocar.get_traffic_light().get_location()).lane_id:
                
                if self.dist_between_transform(self.autocar.get_transform(), self.autocar.get_traffic_light().get_transform()) < 15:
                    self.tlight = self.autocar.get_traffic_light()
                    self.dist_to_tlight = self.dist_between_transform(self.autocar.get_transform(), self.tlight.get_transform()) 
        elif self.tlight is not None and self.dist_to_tlight + 1 <= self.dist_between_transform(self.autocar.get_transform(), self.tlight.get_transform()):
            self.tlight = None
            self.dist_to_tlight = 9e9
            if not self.training:
                self.traffic_light_counter[0] += 1
            if not self.has_collected_enough_traffic_light_samples(): 
                print("passed traffic light")
                done = True 
# if self.over_turned():
        #     if self.has_collected_enough_turn_samples():
        #         self.autocar.apply_control(carla.VehicleControl(brake=1))
        #         self.stop = True
        #         self.target_updated = True
        #         self.update_target()
        #         print("over turned")
        if self.missed_turn():
            
            if self.training:
                if self.has_collected_enough_turn_samples():
                
         
                    if not self.reset_source_and_target():
                        print("cannot reset source targ")
                        done = True
                else:
                    done = True
            else:
                if not self.force_respawn_at_last_chkpt():
                    done = True

        angle = self.target_wp.transform.rotation.yaw - self.autocar.get_transform().rotation.yaw
            
        relative_angle = abs(self.get_angle_normalised(angle))
        if self.training and relative_angle >=111:
            if not self.has_collected_enough_turn_samples():
                print("relatie angle exceeded")
                done = True
        
        if not done:
            self.autocar.apply_control(control)
        return ((self.front_camera, self.left_camera, self.right_camera), self.get_speed(), self.current_direction), done
    def destroy_car(self):
        for sensor in self.sensors:
            sensor.destroy()
        self.autocar.destroy()
        # self.cl.apply_batch([
        #         carla.command.DestroyActor(x) for x in self.sensors])
        
    

        
    def force_respawn_at_last_chkpt(self, max_timeout = 30):
        if self.missed_turns_force_respawn_counter == 2:
            return False
        
        if self.in_proximity(self.source_loc):
            destroyed = True
            self.destroy_car()
        
        
            
            
        temp = self.missed_turns_force_respawn_counter + 1
        #sourceloc still exists after reset
        self._reset()
        self.missed_turns_force_respawn_counter = temp
        ##################
        success = True
        
        if self.autocar.is_alive:
        
            self.autocar.set_transform(self.source_wp.transform)
        else:
            
            s = time.time()
            timeout = 20
            while not self.autocar.is_alive and time.time() - s < timeout:
                try:
                    if self.in_proximity(self.source_loc):
                        continue
                    #TODO cannot directly spawn
                    self.autocar = None
                    self.spawn_vehicle()
                    self.autocar.set_transform(self.source_wp.transform)
                except Exception as e: 
                    print(e)
                    success  =False
                    #reset
                    self.destroy_car()
                    self.spawn_vehicle()
                    break
            if not self.autocar.is_alive:
                #timedout
                self.autocar = None
                self.spawn_vehicle()
                self.autocar.set_transform(self.source_wp.transform)
                success = False
        time.sleep(0.5)
        self.collision_timer = None
        self.sensor_active = True
        self.waypoint_timer = time.time() 
        if self.current_direction == 3 or self.current_direction == 4:
            self.waypoint_timer += 1

        assert self.source_wp != None and self.target_wp != None
        
        return success

    def waiting_for_light(self):
        return self.autocar.is_at_traffic_light()

    @property
    def current_dir_str(self):
        return "follow" if self.current_direction == 2 else "straight" if self.current_direction == 5 else "left" if self.current_direction == 3 else "right" if self.current_direction == 4 else 'undefined direction'
    
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
        
        
        self.w.debug.draw_string(message_loc, f"s={s}, t={t}, b={b}\n {self.current_dir_str}\n", life_time=0.1)
        if self.reached_light():
            self.w.debug.draw_string(message_loc, f"is at traffic light\n", life_time=0.1)

    def view_spectator_birds_eye(self):
        trans_car = self.autocar.get_transform()
        trans_car.rotation.pitch -= 90
        trans_car.location.z += 30
    
        #self.w.debug.draw_string(trans_car.location, "vehicle", life_time=0.1)
        self.spectator.set_transform(trans_car)
        control = self.autocar.get_control()
        s = round(control.steer, 2)
        t = round(control.throttle, 2)
        b = round(control.brake, 2)
        car_loc = self.get_current_location()
        car_dir = self.target_loc - car_loc
        # x = self.current_wp.lane_width * math.cos((self.current_wp.transform.rotation.yaw + 90)/180 * 3.1415) 
        # y = self.current_wp.lane_width * math.sin((self.current_wp.transform.rotation.yaw + 90)/180 * 3.1415) 
        # vec = carla.Location(x, y, 0)

        #self.w.debug.draw_line(self.current_wp.transform.location,  self.current_wp.transform.location + vec)
        target_dir = self.target_loc - self.source_loc
        angle = (self.calculate_angle_between_vec(car_dir, target_dir))
        #self.w.debug.draw_string(self.autocar.get_location(), f"{angle}")
        self.w.debug.draw_string(self.target_loc, f"target")
        dist_npc_car = self.npc_car.get_location().distance(self.autocar.get_location()) if self.npc_car is not None else -1
        self.w.debug.draw_string(self.get_current_location(), f"{round(s, 2), round(t, 2), round(b, 2)}\n{self.current_dir_str}\ncar in front: {self.traffic_jam}\ncollided: {self.collision_timer is not None}\ninfracted: {not self.on_lane}\ndist from edge: {round(self.distance_from_lane_edge, 2)}")
        self.w.debug.draw_string(self.get_current_location(), "v")

        #dist to npc car: {dist_npc_car}\ns={s}, t={t}, b={b}
        #self.w.debug.draw_string(self.get_current_location(),
        #    f"tjam: {self.traffic_jam}\n{self.current_dir_str}\nmissed target: {self.over_turned()}\nmissed turn: {self.missed_turn()}\n{self.get_angle_fwp_wrt_twp()}\n ")

#carla.Transform(carla.Location(x=random.randint(0, 100), y=random.randint(0,100),z=5)))
#sp.location += (carla.Location(x=0, y=-5))
    @property
    def im_dim(self):
        if self.use_baseline:
            return COIL_IM_HEIGHT, COIL_IM_WIDTH
        return IM_HEIGHT, IM_WIDTH
    def process_img(self, event, sensor_id):
        
        i = np.array(event.raw_data)
        t = i.dtype
        i.resize((*self.im_dim, 4))
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
        # if sensor_id == 0:
           
        #     cv2.imshow("front cam", self.front_camera)
        #     #cv2.imshow("left cam", self.left_camera)
        #     #cv2.imshow("right cam", self.right_camera)
        #     cv2.waitKey(1)

    def process_collision(self, event):
        if self.collision_timer is not None:
            return
        self.collision_timer = time.time()
        
        imp = event.normal_impulse

        ("collision occured")
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
#         ("too many reverses")
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
#             ("busy rev timer timedout!")
#         if time.time() - start < 2:
#             ("random action")
#             autocar.apply_control(carla.VehicleControl(steer=random.random(), throttle=0.5))
#         elif time.time() - start < 5:
#             ("agent action")
#             control = agent.run_step()
#             autocar.apply_control(control)
#             trans= autocar.get_transform()
#         else:
#             start = time.time()
#     time.sleep(0.1)
