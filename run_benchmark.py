from neural_net_v2 import *
from environment import CarEnv
from constants import BENCHMARK_LIMIT
def infraction_occured(env : CarEnv):
    return env.distance_from_lane_edge < 0
def main():
    trained_agt = agent(True)

    counters = [0,0,0]
    env = CarEnv(counters, False, debug=True)
    missed_turns = 0
    num_infractions = 0
    infracted =False 
    num_collisions = 0
    collided = False

    total_num_episodes = BENCHMARK_LIMIT * 3
    
    for i in range( total_num_episodes):

        ((ob_front, _, _), _, _), done = env.reset()

        if env.has_collected_enough_turn_samples():
            env.preferred_direction = 2
        while True:
            print(counters)
            s,t,b = trained_agt.get_action(ob_front, env.get_speed(), env.current_direction)
            if b > 0.1:
                b = 0

            ((ob_front, _, _), _, _), done = env.run_step(carla.VehicleControl(steer=s,throttle=t, brake=b))
            if infraction_occured(env):
                if not infracted:
                    num_infractions += 1
                    print("infraction detected!")
                    infracted = True
            else:
                infracted = False

            if done:

                if not collided and env.collision_timer is not None:
                    collided = True
                    num_collisions += 1
                    print("collision detected")
                if env.collision_timer is None:
                    collided = False
                elif not env.reached_dest():
                    #either timed out or turned in the opposite direction (most likely the latter)
                    if not env.has_collected_enough_turn_samples():
                        missed_turns += 1
                
                break

    print(f"% collisions {num_collisions / total_num_episodes}")
    print(f"% infractions {num_infractions / total_num_episodes}")
    print(f"% missed turns {missed_turns / total_num_episodes}")
    

main()