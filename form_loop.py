visited = []
target = None
DESIRED_TOTAL_DIST = 100
world = None
import carla

def set_world(w):
    global world, wp_tuples
    world = w
    world.get_spectator().set_transform(target.transform)
    wp_tuples = world.get_map().get_topology()
    w.debug.draw_string(wp_tuples[0][0].transform.location, "1")
    count = 2
    for src, target in wp_tuples:
        w.debug.draw_string(target.transform.location, f"{count}" )
        count += 1

def set_target(wp):
    global target

    target = wp

def _has_visited(wp):
    for n in visited:
        if n.transform.location == wp.transform.location:
            return True
    return False
def _get_dist(wp0, wp1):
    return wp0.transform.location.distance(wp1.transform.location) 

def _visit(wp):

    visited.append(wp)
    global world
    world.debug.draw_string(wp.transform.location, "visited", life_time=1)
def get_all_adjacent_wps(wp):
    adjs = []
    global wp_tuples
    for src, targ in wp_tuples:
        if src.transform.location == wp.transform.location:
            adjs += targ

    return adjs

'''generate a loop of length at least 1000 metres '''
def dfs(source_wp, target_wp, route, total_dist, next_d=10):
    global target, world
    if target_wp.transform.location == target.transform.location and total_dist > 0:
        if total_dist < DESIRED_TOTAL_DIST:
            return None
        print("found a loop!")
        return route
    elif total_dist > DESIRED_TOTAL_DIST:
        
        return None
    for neighbour in get_all_adjacent_wps(source_wp):
        if not _has_visited(neighbour):
            _visit(neighbour)
            rt = form_loop(neighbour, route + [neighbour], total_dist + _get_dist(neighbour, source_wp))
            if rt is not None:
                return rt
            else:
                visited.remove(neighbour)
    return None

def form_loop(source_wp, route, total_dist, next_d = 10) -> carla.Waypoint:
    return dfs(source_wp, source_wp, total_dist, next_d)
