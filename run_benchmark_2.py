from expert import Expert
from neural_net_v2 import *
from environment import CarEnv
from constants import BENCHMARK_LIMIT, COLLISION_TIMEOUT, DEBUG_BENCHMARK_LIMIT, MIN_SPEED
#from coil_agent.imitation_learning import ImitationLearning
import carla
import sys

sys.path.append(r"C:\Users\autpucv\Documents\coiltraine-master\coiltraine-master")
#from coiltraine_agent import *


from neural_net_v2 import agent
def infraction_occured(env : CarEnv):
    
    return not env.on_lane
    
debug = False

def b_lim():
    return BENCHMARK_LIMIT if not debug else DEBUG_BENCHMARK_LIMIT
def get_current_command_num(env : CarEnv):
    if not env.has_collected_enough_turn_samples():
        return 1 if env.counters[1] < b_lim() else 2
  
    return 0

def main():
    
    debug=False
    #baseline_agt =ImitationLearning('Town01', False)
    baseline_agt = agent(debug=True)
    #baseline_agt = Agent()
    total_dist_travelled_per_ep = 0
    dist_per_episode = 1000
    counters = [0,0,0]
    traffic_light_counter = [0]
    debug = False
    env = CarEnv(counters, traffic_light_counter, training=False, debugg=debug, use_baseline_agent=False, skip_turn_samples= True, port=2000)
    total_dist_travelled = 0
    missed_turns = 0

    num_collisions = 0
    collided = False
    turn_infractions = 0
    follow_infractions = 0
    total_num_episodes = 10
    dist_between_infractions = []
    acc_dist_from_last_infraction = 0
#b_lim() * 3

    import time
    start = time.time()
    total_turn_timer = -1
    
    reached_dest = False
    added_to_acc_dist = False
    num_failures = 0
    total_dist_travelled_per_ep = 0
    total_time_elapsed = 0
    for i in range(total_num_episodes):
        
        ((ob_front, _, _), _, _), done = env.reset()
        expert = Expert(env)
        env.preferred_direction = random.choice([3, 4])#2 if not env.use_baseline else 5
        total_turn_timer = time.time() - total_turn_timer
    
        fake_speed= 6
        s,t,b = baseline_agt.get_action(ob_front, fake_speed, env.current_direction)
        infract_registered = False
        
        env.run_step(carla.VehicleControl(steer=s,throttle=t, brake=b))
        missed_turns_counter_per_ep = 0
        infract_registered = False
        collided = False
        acc_dist_from_last_infraction = 0
        no_infrac_wp = env.source_wp
        infract_timer = -1
        override= False
        total_dist_travelled_per_ep = 0
        override = False
        override_timer = None 
        total_override_time = 0
        
        print("#######################")
        print(f"episode {i}")
        
        while True:
            control = None

            if override:
                control = expert.get_action()
                print("overriding")
            else:
                s,t,b = baseline_agt.get_action(ob_front, env.get_speed(), env.current_direction)
                if t > b:
                    b = 0
                control = carla.VehicleControl(steer=s,throttle=t, brake=b)
            
            reached_dest = env.reached_dest()
            #if collision timer is none
            if (infraction_occured(env) and not infract_registered) or (env.collision_timer is not None and not collided):
                
                fv = False
                if env.collision_timer is not None and not collided:
                    print("collision start")
                    collided = True
                    num_collisions += 1
                if infraction_occured(env) and not infract_registered:
                    fv = True

                    print("infraction start")
                    infract_timer = time.time()
                    infract_registered = True
                    if env.current_direction == 3 or env.current_direction == 4:
                        turn_infractions += 1
                    else:
                        follow_infractions += 1

                dist_between_infractions.append(acc_dist_from_last_infraction + env.get_dist_from_source_wp() )
                acc_dist_from_last_infraction = 0
                
                

            elif infract_registered or collided:
              
                f = False 
                if collided:
                    if env.collision_timer is None:
                        print("collision end")
                        fv = True
                        collided = False
                    elif time.time() - env.collision_timer >= COLLISION_TIMEOUT - 0.1:
                        #intervene and manually set autocar's location to target
                        if not env.force_update_location():
                            
                            #should not have happed normally
                            done = True

                        collided = False
                    elif time.time() - env.collision_timer < COLLISION_TIMEOUT - 0.1:
                        if not override:
                            override = True
                            override_timer = time.time()

                    
                if infract_registered:
                    if not infraction_occured(env):
                        fv = True
                        
                        infract_registered = False
                        no_infrac_wp = env.current_wp
                        print("infraction end")
                    elif time.time() - infract_timer > 5 and not override:
                        assert infract_timer != -1
                        print("starting override")
                        override = True
                        override_timer = time.time()


            if not infract_registered and not collided:
                if override:
                    assert override_timer is not None
                if override and override_timer is not None:
                        if env.get_speed() > 1:
                            print("ending override")
                           
                            override = False
                            override_timer += time.time() - override_timer
                            override_timer = None
                            
                        elif time.time() - override_timer > 3:
                            
                            override = False
                            total_override_time += time.time() - override_timer
                            override_timer = None
                            
                    # else:
                    #     print("ending override")
                    #     override = False

            reached_dest = env.reached_dest()
            #a force_update_location will make sure collided = False and infract = False
            if reached_dest or env.force_update_targ:
                
                    #prefer turns instead of straight
                env.preferred_direction = random.choice([3, 4])
                total_dist_travelled_per_ep += env.get_dist_between_src_dest()
                
                #only count if the vehicle is currently on lane and that it is not in a collision
                if not infract_registered and not collided:
                    
                    infract_timer = -1
                    if no_infrac_wp is not None:
                        d = env.dist_between_transform(no_infrac_wp.transform, env.target_wp.transform)
                        acc_dist_from_last_infraction += d
                        no_infrac_wp = None
                    else:
                        acc_dist_from_last_infraction += env.get_dist_between_src_dest()
                print(f"total dist so far {total_dist_travelled_per_ep} m")
                print(f"time taken so far {env.time_counter / 60} mins")
            #reached intermediate waypoint
            
            if env.missed_turn() and env.missed_turns_force_respawn_counter == 3:
                
                print("missed a turn")
                missed_turns_counter_per_ep += 1
                infract_registered = False
                collided = False
                missed_turns += 1
                

                #recovered form collision
            #timer will be reset to none only when collision timeout is exceeded
        
            if not done:
                forced_update_targ = env.force_update_targ
                ((ob_front, _, _), _, _), done = env.run_step(control)
                if expert.car_respawned():
                    expert = Expert(env)
                    print("car respawned")
                if expert.target_loc != env.target_loc:
                    
                    expert.update_target()
                    env.w.debug.draw_string(expert.target_loc, "expert location", life_time=5)
            if not done and env.get_speed() == 0 and not override:
                print("start override")
                override = True
                override_timer = time.time()
            if done or env.time_counter >= 6.5 * 60 or total_dist_travelled_per_ep >= dist_per_episode:
                if env.time_counter < 6.5 * 60 and total_dist_travelled_per_ep < dist_per_episode:
                    if env.get_speed() > 1:
                        continue

                if total_dist_travelled_per_ep < dist_per_episode:
                    
                    num_failures += 1
                elif env.time_counter >= 6.5 * 60:
                    #total time elapsed should not be greater than 6.5 min
                    num_failures += 1
                if not collided and not infract_registered:
        
                        #there will be no more infractions so add the last
                        if acc_dist_from_last_infraction > 0:
                            dist_between_infractions.append(acc_dist_from_last_infraction)
                total_time_elapsed += env.time_counter
                total_dist_travelled += total_dist_travelled_per_ep
                break
            
             

    end = time.time() - start

    print("############################")
    print(f"Number of collisions {num_collisions}")
    print(f"Number of turn infractions {(turn_infractions)}")
    print(f"Number of straight infractions {(follow_infractions)}")
    print(f"Success rate: {round((total_num_episodes - num_failures) / total_num_episodes * 100, 2)}%")
    print(f"Number of missed turns {missed_turns}")
    print(f"Average distance between infractions: {round(sum(dist_between_infractions) / len(dist_between_infractions))}")
    print(f"Total distance travelled: {total_dist_travelled}")
    print(f"Total time elapsed: {round(total_time_elapsed / 60, 2)} mins")

main()