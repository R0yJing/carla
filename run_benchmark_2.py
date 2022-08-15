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
    #baseline_agt = agent(True)
    baseline_agt = Agent()
    total_dist_travelled = 0
    dist_per_episode = 50
    counters = [0,0,0]
    traffic_light_counter = [0]
    debug = True
    env = CarEnv(counters, traffic_light_counter, training=False, debugg=debug, use_baseline_agent=True)
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
    total_num_episodes = b_lim() * 3

    import time
    total_turn_timer = -1
    start = 0
    reached_dest = False
    added_to_acc_dist = False
    num_failures = 0
    for i in range(total_num_episodes):
        
        ((ob_front, _, _), _, _), done = env.reset()

        
        
        current_idx = get_current_command_num(env)
        
        env.preferred_direction = random.choice([3, 4])#2 if not env.use_baseline else 5
        total_turn_timer = time.time() - total_turn_timer
        s,t,b = baseline_agt.get_action(ob_front, env.get_speed(), env.current_direction)
        infract_registered = False
        env.run_step(carla.VehicleControl(steer=s,throttle=t, brake=b))
        infract_end_wp = None
        added_missed_turn = False
        infract_registered = False
        collided = False
        added_to_acc_dist = False
        acc_dist_from_last_infraction = 0
        total_dist_travelled = 0
        print("#######################")
        print(f"episode {i}")
        while True:
            
            print 
            s,t,b = baseline_agt.get_action(ob_front, env.get_speed(), env.current_direction)
            # if t > b:
            #     b = 0
            # if b < 0.1:
            #     b = 0
            control = carla.VehicleControl(steer=s,throttle=t, brake=b)
            
            if (infraction_occured(env) and not infract_registered) or env.collision_timer is not None and not collided:
                
               
                if env.collision_timer is None and not collided:
                    collided = True
                    num_collisions += 1
                if infraction_occured(env) and not infract_registered:
                    infract_registered = True
                    if env.current_direction == 3 or env.current_direction == 4:
                        turn_infractions += 1
                    else:
                        follow_infractions += 1
                acc_dist_from_last_infraction += env.get_dist_from_source_wp()
                dist_between_infractions.append(acc_dist_from_last_infraction)
                acc_dist_from_last_infraction = 0
                
            elif infract_registered or collided:
                
                if collided:
                    if not env.collision_timer is None:
                        collided = False
                if not infraction_occured(env):
                        
                    infract_end_wp = env.current_wp
                    infract_registered = False
            #reached intermediate waypoint
            if env.reached_dest() and not done and not added_to_acc_dist:
               
                    #prefer turns instead of straight
                env.preferred_direction = random.choice([3, 4])
                total_dist_travelled += env.get_dist_between_src_dest()
               
                added_to_acc_dist = True
                print("reached destination")
                if not infraction_occured(env) and not collided:
                    if infract_end_wp is not None:
                        acc_dist_from_last_infraction += env.dist_between_transform(infract_end_wp.transform, env.target_wp.transform)
                    else:
                        acc_dist_from_last_infraction += env.get_dist_between_src_dest()
                    
        
            if env.missed_turn() and not added_missed_turn:
                
                print("missed a turn")

                missed_turns += 1
                added_missed_turn = True
        
            elif not env.missed_turn() and added_missed_turn:
                added_missed_turn = False     



                #recovered form collision
            #timer will be reset to none only when collision timeout is exceeded
          
        
            ((ob_front, _, _), _, _), done = env.run_step(control)
            
            
            if done:
                # if not env.has_collected_enough_turn_samples():
                #     if not env.turn_made():
                #         missed_turns += 1
                #if missed turn, then timedout

                
                if total_dist_travelled < dist_per_episode:
                    num_failures += 1
                
                #######################
                
                break

                

    end = time.time() - start
    total_num_episodes = sum(env.counters)
    print(f"collisions {num_collisions / total_num_episodes * 100}% {num_collisions} out of {total_num_episodes}")
    print(f"turn infractions {(turn_infractions)}")
    print(f"straight infractions {(follow_infractions)}")
    print(f"success rates: {(total_num_episodes - num_failures) / total_num_episodes * 100}%")
    print(f"missed turns {missed_turns}")
    print(f"avergae distance between infractions: {sum(dist_between_infractions) / len(dist_between_infractions) * 100}%")
    print(f"total time elapsed: {round(end / 60, 2)} mins")
    #TODO success rate
    #
main()