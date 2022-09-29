import carla
from environment import CarEnv
from expert import Expert
from threading import Thread
def navigate(env):
    expert = Expert(env, False)
    while True:
        env.run_step(expert.get_action())
_ = CarEnv([0, 0, 0], [0], False)
_.cleanup()

envs = [CarEnv([0, 0, 0], [0], False), CarEnv([0, 0, 0], [0], False), CarEnv([0, 0, 0], [0], False)]
for env in envs:
    env.reset()
t0 = Thread(target=navigate, args=(envs[0], ), daemon=True)
t1 = Thread(target=navigate, args=(envs[1], ), daemon=True)
t2 = Thread(target=navigate, args=(envs[2], ), daemon=True)
threads = [t0, t1, t2]
for t in threads:
    t.start()

import time
timer = time.time()
ptr = 0
while True:
    if time.time() - timer > 3:
        ptr = (ptr + 1) % len(envs)
        timer = time.time()
        
    envs[ptr].view_spectator_birds_eye()
    time.sleep(0.01)
