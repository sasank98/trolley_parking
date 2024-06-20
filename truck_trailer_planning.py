#!/usr/bin/env python3


import numpy as np
import math
import heapq

import scipy.spatial.kdtree as kd
import matplotlib.pyplot as plt
from heapdict import heapdict

import CurvesGenerator.reeds_shepp as rs
from vehicles import TruckTrailer

from hybrid_astar import*

# class State:
#     def __init__(self, state_id, trajectory_rollout, steering_angle, traversal_dir, cost, parent_index):
#         self.state_id = state_id
#         self.trajectory_rollout = trajectory_rollout
#         self.steering_angle = steering_angle
#         self.traversal_dir = traversal_dir
#         self.cost = cost
#         self.parent_index = parent_index


class Cost:
    reverse = 10
    direction_change = 150
    steer_angle = 1
    steer_angle_change = 5
    hybrid_cost = 50
    jack_knife_cost = 200.0
    path_resolution = 0.2 #move step
  

def get_trailer_yaw(path_yaw, yaw_t_0, steps):

    yawt = [0.0 for _ in range(len(path_yaw))]
    yawt[0] = yaw_t_0

    for i in range(1, len(path_yaw)):
        yawt[i] += yawt[i - 1] + steps[i - 1] / TruckTrailer.length_t * math.sin(path_yaw[i - 1] - yawt[i - 1])

    return yawt


def reeds_shepp_cost_t(current_state, path, yaw_t):

    # Previos Node Cost
    cost = current_state.cost

    # Distance cost
    for i in path.lengths:
        if i >= 0:
            cost += 1
        else:
            cost += abs(i) * Cost.reverse

    # Direction change cost
    for i in range(len(path.lengths)-1):
        if path.lengths[i] * path.lengths[i+1] < 0:
            cost += Cost.direction_change

    # Steering Angle Cost
    for i in path.ctypes:
        # Check types which are not straight line
        if i!="S":
            cost += AckermannCar.maxSteerAngle * Cost.steer_angle

    # Steering Angle change cost
    turn_angle=[0.0 for _ in range(len(path.ctypes))]
    for i in range(len(path.ctypes)):
        if path.ctypes[i] == "R":
            turn_angle[i] = - AckermannCar.maxSteerAngle
        if path.ctypes[i] == "WB":
            turn_angle[i] = AckermannCar.maxSteerAngle

    for i in range(len(path.lengths)-1):
        cost += abs(turn_angle[i+1] - turn_angle[i]) * Cost.steer_angle_change

        #Added for truck trailer
        cost += Cost.jack_knife_cost * sum([abs(rs.pi_2_pi(x - y))
                                   for x, y in zip(path.yaw, yaw_t)])

    return cost

def collision_t(trajectory_rollout, map_env):

    car_radius = (TruckTrailer.axleToFront + TruckTrailer.axleToBack)/2 + 1
    dl = (TruckTrailer.axleToFront - TruckTrailer.axleToBack)/2

    trailer_radius = (TruckTrailer.axleTrailerToFront + TruckTrailer.axleTrailerToBack)/2 + 1
    dl_t  = (TruckTrailer.axleTrailerToFront - TruckTrailer.axleTrailerToBack)/2

    for i in trajectory_rollout:

        cx_t = i[0] + dl_t * math.cos(i[3])
        cy_t = i[1] + dl_t * math.sin(i[3])
        obstacle_points_t = map_env.obstacle_kdtree.query_ball_point([cx_t, cy_t], trailer_radius)

        if not obstacle_points_t:
            continue

        for p_t in obstacle_points_t:
            xo_t = map_env.obs_x[p_t] - cx_t
            yo_t = map_env.obs_y[p_t] - cy_t
            dx_t = xo_t * math.cos(i[2]) + yo_t * math.sin(i[3])
            dy_t = -xo_t * math.sin(i[2]) + yo_t * math.cos(i[3])

            if abs(dx_t) < trailer_radius and abs(dy_t) < TruckTrailer.width / 2 + 1:
                return True


        cx = i[0] + dl * math.cos(i[2])
        cy = i[1] + dl * math.sin(i[2])
        obstacle_points = map_env.obstacle_kdtree.query_ball_point([cx, cy], car_radius)

        if not obstacle_points:
            continue

        for p in obstacle_points:
            xo = map_env.obs_x[p] - cx
            yo = map_env.obs_y[p] - cy
            dx = xo * math.cos(i[2]) + yo * math.sin(i[2])
            dy = -xo * math.sin(i[2]) + yo * math.cos(i[2])

            if abs(dx) < car_radius and abs(dy) < AckermannCar.width / 2 + 1:
                return True

    return False

    

#analaytical expansion
def reeds_sheep_node_t(current_state, goal_state, env_map):

     # Get x, y, yaw of currentNode and goalNode
    start_x, start_y, start_yaw = current_state.trajectory_rollout[-1][0], current_state.trajectory_rollout[-1][1], current_state.trajectory_rollout[-1][2]
    goal_x, goal_y, goal_yaw = goal_state.trajectory_rollout[-1][0], goal_state.trajectory_rollout[-1][1], goal_state.trajectory_rollout[-1][2]


    radius = math.tan(AckermannCar.maxSteerAngle)/AckermannCar.wheelBase

    reeds_shepp_paths = rs.calc_all_paths(start_x, start_y, start_yaw, goal_x, goal_y, goal_yaw, radius, 1)
    # directions = reeds_shepp_paths.directions
    if not reeds_shepp_paths:
        return None

    
    # Find path with lowest cost considering non-holonomic constraints
    cost_queue = heapdict()
    directions = []
    for path in reeds_shepp_paths:
        #calculate steps
        steps = [Cost.path_resolution * delta for delta in path.directions]
        print("Path yaw", path.yaw)
        
        theta_trailer = get_trailer_yaw(path.yaw, current_state.state_id[3], steps )
        #need to calculate modify method to add trailer yaw
        cost_queue[path] = reeds_shepp_cost_t(current_state, path, theta_trailer)
        directions = path.directions



    while len(cost_queue)!=0:
        path = cost_queue.popitem()[0]
        steps = [Cost.path_resolution * delta for delta in path.directions]
        theta_trailer = get_trailer_yaw(path.yaw, current_state.state_id[3], steps )

        traj=[]
        traj = [[path.x[k],path.y[k],path.yaw[k], theta_trailer[k]] for k in range(len(path.x))]
        if not collision_t(traj, env_map):
            cost = reeds_shepp_cost_t(current_state, path, theta_trailer)
            return State(goal_state.state_id ,traj, None, directions, cost, index_t(current_state))
            
    return None

def path_cost_t(current_state, motion_primitive, simulation_length):

    # Previos Node Cost
    cost = current_state.cost

    # Distance cost
    if motion_primitive[1] == 1:
        cost += simulation_length 
    else:
        cost += simulation_length * Cost.reverse

    # Direction change cost
    if current_state.direction != motion_primitive[1]:
        cost += Cost.directionChange

    # Steering Angle Cost
    cost += motion_primitive[0] * Cost.steerAngle

    # Steering Angle change cost
    cost += abs(motion_primitive[0] - current_state.steeringAngle) * Cost.steerAngleChange
    
    #TODO check for current.state_id[2] and [3]
    cost += Cost.jack_knife_cost *  sum([ abs(rs.pi_2_pi(x - y)) for x, y in zip(current_state.state_id[2], current_state.state_id[3])]) 

    return cost

def get_kinematics_t(current_state, motion_primitive, env_map, simulation_length=4, step = 0.8 ):

    # Simulate node using given current Node and Motion Commands
    traj = []
    angle = rs.pi_2_pi(current_state.trajectory_rollout[-1][2] + motion_primitive[1] * step / TruckTrailer.wheelBase * math.tan(motion_primitive[0]))
    trailer_angle = rs.pi_2_pi(current_state.trajectory_rollout[-1][3] +
         motion_primitive[1] * step / TruckTrailer.RearToTrailerWheel * math.sin(current_state.trajectory_rollout[-1][2] - current_state.trajectory_rollout[-1][3]))


    traj.append([current_state.trajectory_rollout[-1][0] + motion_primitive[1] * step * math.cos(angle),
                current_state.trajectory_rollout[-1][1] + motion_primitive[1] * step * math.sin(angle),
                rs.pi_2_pi(angle + motion_primitive[1] * step / TruckTrailer.wheelBase * math.tan(motion_primitive[0])),
                 rs.pi_2_pi(trailer_angle + motion_primitive[1]* step / TruckTrailer.RearToTrailerWheel * math.sin(current_state.trajectory_rollout[-1][2] - current_state.trajectory_rollout[-1][3]))])


    for i in range(int((simulation_length/step))-1):
        traj.append([traj[i][0] + motion_primitive[1] * step * math.cos(traj[i][2]),
                    traj[i][1] + motion_primitive[1] * step * math.sin(traj[i][2]),
                    rs.pi_2_pi(traj[i][2] + motion_primitive[1] * step / TruckTrailer.wheelBase * math.tan(motion_primitive[0])),
                    rs.pi_2_pi(traj[i][3] + motion_primitive[1]* step / TruckTrailer.RearToTrailerWheel * math.sin(traj[i][2] - traj[i][3]))])

        print("Traj", traj[i][3])

    state_id = [round(traj[-1][0]/env_map.cordinate_resolution), \
                 round(traj[-1][1]/env_map.cordinate_resolution), \
                 round(traj[-1][2]/env_map.theta_resolution), \
                round(traj[-1][3]/env_map.theta_resolution)]

    # Check if node is valid
    if not is_state_valid(traj, state_id, env_map):
        return None

    # Calculate Cost of the node
    cost = path_cost_t(current_state, motion_primitive, simulation_length)

    return State(state_id, traj, motion_primitive[0], motion_primitive[1], cost, index_t(current_state))

def backtrack_t(start_state, goal_state, closed_set, plt):

    # Goal Node data
    start_state_id = index_t(start_state)
    current_state_id = goal_state.parent_index
    current_state = closed_set[current_state_id]

    x=[]
    y=[]
    theta=[]
    theta_trailer = []
    directions = []


    # Iterate till we reach start node from goal node
    while current_state_id != start_state_id:
        a, b, c, d = zip(*current_state.trajectory_rollout)
        x += a[::-1] 
        y += b[::-1] 
        theta += c[::-1]
        theta_trailer += d[::-1]

        directions = current_state.traversal_dir[::-1]
    
        print("theta trailer", theta_trailer)

        current_state_id = current_state.parent_index
        current_state = closed_set[current_state_id]

    return x[::-1], y[::-1], theta[::-1], theta_trailer[::-1], directions

def index_t(State):
    # Index is a tuple consisting grid index, used for checking if two nodes are near/same
    return tuple([State.state_id[0], State.state_id[1], State.state_id[2], State.state_id[3]])


def hybridAstar_trailer(start_pose, goal_pose, env_map, plt):

    print("I am here ")
    start_pose_id = [round(start_pose[0] / env_map.cordinate_resolution), \
                round(start_pose[1] / env_map.cordinate_resolution), \
                round(start_pose[2]/env_map.theta_resolution), \
                round(start_pose[3]/env_map.theta_resolution)]

    goal_pose_id = [round(goal_pose[0] / env_map.cordinate_resolution), \
                round(goal_pose[1] / env_map.cordinate_resolution), \
                round(goal_pose[2]/env_map.theta_resolution), \
                round(goal_pose[3]/env_map.theta_resolution)]

    motion_primitives = getMotionPrimitives()

    start_state = State(start_pose_id, [start_pose], 0, 1, 0, tuple(start_pose_id))
    goal_state = State(goal_pose_id, [goal_pose], 0, 1, 0, tuple(goal_pose_id))


    unconstrained_heuristics =  getUnconstrainedHeuristic(goal_state, env_map)
    # print("Unconstrained Heuristics", unconstrained_heuristics)

    open_set = {index_t(start_state):start_state}
    closed_set = {}

    cost_priority_queue = heapdict()

    cost_priority_queue[index_t(start_state)] = max(start_state.cost, Cost.hybrid_cost * unconstrained_heuristics[start_state.state_id[0]][start_state.state_id[1]])
    
    counter = 0

    # print("  cost_priority_queue ",  cost_priority_queue.popitem()[0])

    while True:
        counter += 1

        if not open_set:
            print("Im going out")
            return None

        current_state_id = cost_priority_queue.popitem()[0]
        current_state = open_set[current_state_id]
        print("Current state id ", current_state_id, counter)

        open_set.pop(current_state_id)
        closed_set[current_state_id] = current_state
        print("Closed set",closed_set)

        rs_node = reeds_sheep_node_t(current_state, goal_state, env_map)

        # If Reeds-Shepp Path is found exit
        if rs_node:
            closed_set[index_t(rs_node)] = rs_node
            break


        if current_state_id == index_t(goal_state):
            print(current_state.trajectory_rollout[-1])
            break

        for i in range(len(motion_primitives)):

            simulated_state = get_kinematics_t(current_state, motion_primitives[i], env_map)

        if not simulated_state:
                continue

        # Draw Simulated Node
        x,y,z =zip(*simulated_state.trajectory_rollout)
        plt.plot(x, y, linewidth=0.3, color='g')

        # Check if simulated node is already in closed set
        simulated_state_id = index_t(simulated_state)
        if simulated_state_id not in closed_set: 

            # Check if simulated node is already in open set, if not add it open set as well as in priority queue
            if simulated_state_id not in open_set:
                open_set[simulated_state_id] = simulated_state
                cost_priority_queue[simulated_state_id] = max(simulated_state.cost , Cost.hybridCost * unconstrained_heuristics[simulated_state.state_id[0]][simulated_state.state_id[1]])

                print("Check simulated node in open set")
            else:
                if simulated_state.cost < open_set[simulated_state_id].cost:
                    open_set[simulated_state_id] = simulated_state
                    cost_priority_queue[simulated_state_id] = max(simulated_state.cost , Cost.hybridCost * unconstrained_heuristics[simulated_state.state_id[0]][simulated_state.state_id[1]])

    x, y, yaw, yaw_t , directions= backtrack_t(start_state, goal_state, closed_set, plt)

    return x, y, yaw, yaw_t, directions