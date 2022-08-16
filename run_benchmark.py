from neural_net_v2 import *
from environment import CarEnv
from constants import BENCHMARK_LIMIT, DEBUG_BENCHMARK_LIMIT, MIN_SPEED
#from coil_agent.imitation_learning import ImitationLearning
import carla
import sys

sys.path.append(r"C:\Users\autpucv\Documents\coiltraine-master\coiltraine-master")
from coiltraine_agent import *

#from neural_net_v2 import agent
def infraction_occured(env : CarEnv):
    print(env.distance_from_lane_edge)
    
    return not env.on_lane
    
debug = True

def b_lim():
    return BENCHMARK_LIMIT if not debug else DEBUG_BENCHMARK_LIMIT
def get_current_command_num(env : CarEnv):
    if not env.has_collected_enough_turn_samples():
        return 1 if env.counters[1] < b_lim() else 2
    print("going straight")
    return 0
def main():
    #baseline_agt =ImitationLearning('Town01', False)
    baseline_agt = agent(True)
    #baseline_agt = Agent()

    counters = [0,0,0]
    traffic_light_counter = [0]
    debug = False
    env = CarEnv(counters, traffic_light_counter, training=False, debugg=debug, use_baseline_agent=False)
    missed_turns = 0
    num_infractions = 0
    infracted =False 
    num_collisions = 0
    collided = False
    turn_infractions = 0
    follow_infractions = 0
    turn_infract_time = 0
    follow_infract_time = 0
    turn_infract_timer = -1
    follow_infract_timer = -1
    dist_between_infractions = []
    acc_dist_from_last_infraction = 0
    total_num_episodes = b_lim() * 3

    import time
    total_turn_timer = -1
    start = 0
    reached_dest = False
    added_to_acc_dist = False
    for i in range(3):
        
        ((ob_front, _, _), _, _), done = env.reset()

        
        
        current_idx = get_current_command_num(env)
        if i == 2:
            env.preferred_direction = 2 if not env.use_baseline else 5
            total_turn_timer = time.time() - total_turn_timer
        s,t,b = baseline_agt.get_action(ob_front, env.get_speed(), env.current_direction)
        infract_registered = False
        env.run_step(carla.VehicleControl(steer=s,throttle=t, brake=b))
        infract_end_wp = None
        added_missed_turn = False
        while counters[current_idx] < b_lim():

           
            s,t,b = baseline_agt.get_action(ob_front, env.get_speed(), env.current_direction)
            # if t > b:
            #     b = 0
            # if b < 0.1:
            #     b = 0
            control = carla.VehicleControl(steer=s,throttle=t, brake=b)
            
            if infraction_occured(env) and not infract_registered:
            
                infract_registered = True
                acc_dist_from_last_infraction += env.get_dist_from_source_wp()
                dist_between_infractions.append(acc_dist_from_last_infraction)
                acc_dist_from_last_infraction = 0
                if env.current_direction == 3 or env.current_direction == 4:
                    turn_infractions += 1
                else:
                    follow_infractions += 1
            elif infract_registered and not infraction_occured(env):
                infract_end_wp = env.current_wp
                infract_registered = False
        
            if env.reached_dest() and not done and not added_to_acc_dist:
                added_to_acc_dist = True
                print("reached destination")
                if env.on_lane:
                    if infract_end_wp is not None:
                        acc_dist_from_last_infraction += env.dist_between_transform(infract_end_wp.transform, env.target_wp.transform)
                    else:
                        acc_dist_from_last_infraction += env.get_dist_between_src_dest()
                    
        
            if env.missed_turn() and not added_missed_turn:
                
                print("missed a turn")
                print(env.angle)
                missed_turns += 1
                added_missed_turn = True
            elif not env.missed_turn() and added_missed_turn:
                added_missed_turn = False        
            ((ob_front, _, _), _, _), done = env.run_step(control)
            
            if not done and not collided and env.collision_timer is not None:
                    collided = True
                    num_collisions += 1
                    print("collision detected")

                #recovered form collision
            #timer will be reset to none only when collision timeout is exceeded
            elif collided and done:
                collided = False
            if done:
                # if not env.has_collected_enough_turn_samples():
                #     if not env.turn_made():
                #         missed_turns += 1
                #if missed turn, then timedout

                
                
                env.reset()
                #######################
                infract_registered = False
                added_to_acc_dist = False
                acc_dist_from_last_infraction = 0
                added_missed_turn = False
            

                

    end = time.time() - start
    total_num_episodes = sum(env.counters)
    print(f"collisions {num_collisions / total_num_episodes * 100}% {num_collisions} out of {total_num_episodes}")
    print(f"turn infractions {(turn_infractions)}")
    print(f"straight infractions {(follow_infractions)}")

    print(f"missed turns {missed_turns}")
    print(f"avergae distance between infractions: {sum(dist_between_infractions) / len(dist_between_infractions) * 100}%")
    print(f"total time elapsed: {round(end / 60, 2)} mins")
    #TODO success rate
    #
main()