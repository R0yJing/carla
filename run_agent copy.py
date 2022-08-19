
from environment import CarEnv
from expert import Expert

import time
env = CarEnv([0, 0, 0], [], training=False, debugg=True, skip_turn_samples=True)

env.reset()
expert =Expert(env)

total_dist_travelled = 0
start = time.time()
import random
while total_dist_travelled < 1000:

    control = expert.get_action()
    reached_dest = env.reached_dest()
    ob, done  = env.run_step(control)
    
    
    if reached_dest or env.target_reset:
        env.preferred_direction = random.choice([2,3,4])
        total_dist_travelled += env.get_dist_between_src_dest()
        print("updating target")
        expert.update_target()
        if env.target_reset:
            env.target_reset = False
    elif done:
        env.reset()
        expert = Expert(env)
    #control = agent.get_action(env.front_camera, env.get_speed(), env.current_direction)
    #obs, done = env.run_step(control)

    time.sleep(0.1)

print(f"1000m took {(time.time() - start)/60} mins" )