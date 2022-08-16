from neural_net_v2 import *
from environment import CarEnv
from constants import BENCHMARK_LIMIT, DEBUG_BENCHMARK_LIMIT, MIN_SPEED
#from coil_agent.imitation_learning import ImitationLearning
import carla
import sys

sys.path.append(r"C:\Users\autpucv\Documents\coiltraine-master\coiltraine-master")
from coiltraine_agent import *


from neural_net_v2 import agent
def infraction_occured(env : CarEnv):
    
    return not env.on_lane
    
debug = True

def b_lim():
    return BENCHMARK_LIMIT if not debug else DEBUG_BENCHMARK_LIMIT
def get_current_command_num(env : CarEnv):
    if not env.has_collected_enough_turn_samples():
        return 1 if env.counters[1] < b_lim() else 2
  
    return 0

def main():
    #baseline_agt =ImitationLearning('Town01', False)
    #baseline_agt = agent(True)
    baseline_agt = Agent()
    total_dist_travelled = 0
    dist_per_episode = 1000
    counters = [0,0,0]
    traffic_light_counter = [0]
    debug = True
    env = CarEnv(counters, traffic_light_counter, training=False, debugg=debug, use_baseline_agent=True, skip_turn_samples= True)
    missed_turns = 0
    num_infractions = 0
    infracted =False 
    num_collisions = 0
    collided = False
    turn_infractions = 0
    follow_infractions = 0
    total_num_episodes = 10
    dist_between_infractions = []
    acc_dist_from_last_infraction = 0
#b_lim() * 3

    import time
    total_turn_timer = -1
    start = time.time()
    reached_dest = False
    added_to_acc_dist = False
    num_failures = 0
    total_dist_travelled = 0
    
    for i in range(total_num_episodes):
        
        ((ob_front, _, _), _, _), done = env.reset()
        
        env.preferred_direction = random.choice([3, 4])#2 if not env.use_baseline else 5
        total_turn_timer = time.time() - total_turn_timer
        s,t,b = baseline_agt.get_action(ob_front, env.get_speed(), env.current_direction)
        infract_registered = False
        env.run_step(carla.VehicleControl(steer=s,throttle=t, brake=b))
        infract_end_wp = None
        missed_turns_counter_per_ep = 0
        infract_registered = False
        collided = False
        added_to_acc_dist = False
        acc_dist_from_last_infraction = 0
        no_infrac_wp = env.source_wp

        
        print("#######################")
        print(f"episode {i}")
        while True:
        
            s,t,b = baseline_agt.get_action(ob_front, env.get_speed(), env.current_direction)
            b =0
            control = carla.VehicleControl(steer=s,throttle=t, brake=b)
            
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
                if not infraction_occured(env):
                    fv = True
                    infract_end_wp = env.current_wp
                    infract_registered = False
                    print("infraction end")
                no_infrac_wp = env.current_wp
                
            if env.reached_dest():
                
                    #prefer turns instead of straight
                env.preferred_direction = random.choice([3, 4])
                total_dist_travelled += env.get_dist_between_src_dest()
               
                #only count if the vehicle is currently on lane and that it is not in a collision
                if not infract_registered and not collided:
                    # if infract_end_wp is not None:
                    #     acc_dist_from_last_infraction += env.dist_between_transform(infract_end_wp.transform, env.target_wp.transform)
                    # else:
                
                    print(env.get_dist_between_src_dest())
                    print()
                    if no_infrac_wp is not None:
                        d = env.dist_between_transform(no_infrac_wp.transform, env.target_wp.transform)
                        acc_dist_from_last_infraction += d
                        no_infrac_wp = None
                    else:
                        acc_dist_from_last_infraction += env.get_dist_between_src_dest()

            #reached intermediate waypoint
            
            if env.missed_turn():
                
            
                missed_turns_counter_per_ep += 1
                infract_registered = False
                collided = False
                missed_turns += 1
             

                #recovered form collision
            #timer will be reset to none only when collision timeout is exceeded
          
            ((ob_front, _, _), _, _), done = env.run_step(control)
        
            if done:
                if total_dist_travelled < dist_per_episode:
                    
                    num_failures += 1

                if not collided and not infract_registered:
                      
                        #there will be no more infractions so add the last
                        if acc_dist_from_last_infraction > 0:
                            dist_between_infractions.append(acc_dist_from_last_infraction)

                break
            
            

    end = time.time() - start

    print("############################")
    print(f"collisions {num_collisions}")
    print(f"turn infractions {(turn_infractions)}")
    print(f"straight infractions {(follow_infractions)}")
    print(f"success rate: {round((total_num_episodes - num_failures) / total_num_episodes * 100, 2)}%")
    print(f"missed turns {missed_turns}")
    print(f"average distance between infractions: {round(sum(dist_between_infractions) / len(dist_between_infractions))}")
    print(f"total distance travelled: {total_dist_travelled}")
    print(f"total time elapsed: {round(end / 60, 2)} mins")

main()