import carla
import argparse
#from subprocess import Popen
import subprocess
parser = argparse.ArgumentParser()
parser.add_argument("-port", help="choose port", default=2000)
parser.add_argument("-t", help="town number", default="01")
args = parser.parse_args()
town = "Town" + args.t
#stdout, stderr = p.communicate()
print(args.port)
cl = carla.Client('localhost', int(args.port))
cl.set_timeout(5000)
cl.load_world(town)

# topology = cl.get_world().get_map().get_topology()
# for w0, w1 in topology:
#     loc0 = w0.transform.location
#     loc1 = w1.transform.location
#     cl.get_world().debug.draw_string(loc0, "t", life_time=600)
#     cl.get_world().debug.draw_string(loc1, "t", life_time=600)

# sps = cl.get_world().get_map().get_spawn_points()
# for t in sps:
#     cl.get_world().debug.draw_string(t.location, "SPAWN",life_time=1200)

