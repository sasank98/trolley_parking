#!/usr/bin/env python3

import numpy as np
import math
import heapq

import scipy.spatial.kdtree as kd
import matplotlib.pyplot as plt
from heapdict import heapdict

import CurvesGenerator.reeds_shepp as rs
from vehicles import AckermannCar 

"""
Class provides defination of state-vertex or node consider following parameters.
"""
class State:
    def __init__(self, state_id, trajectory_rollout, steering_angle, traversal_dir, cost, parent_index):
        self.state_id = state_id
        self.trajectory_rollout = trajectory_rollout
        self.steering_angle = steering_angle
        self.traversal_dir = traversal_dir
        self.cost = cost
        self.parent_index = parent_index

"""
Provides defination for holonomic state used for calculated unconstrained heuristics
"""
class unconstraintedState:
    def __init__(self, state_id, cost, parent_id):
        self.state_id = state_id
        self.cost = cost
        self.parent_id = parent_id

"""
Defines costs considered for pruning states after expansion
"""
class Cost:
    reverse = 10
    direction_change = 150
    steer_angle = 1
    steer_angle_change = 5
    hybrid_cost = 50

"""
Defines environment parameters like angle resolution coordinate resolution, obstacles
"""
class getEnvParameters:
    def __init__(self, env_xmin, env_ymin, env_xmax,
     env_ymax, cordinate_resolution, theta_resolution, obstacle_kdtree, obs_x, obs_y):
        self.env_xmin = env_xmin
        self.env_ymin = env_ymin
        self.env_xmax = env_xmax
        self.env_ymax = env_ymax
        self.cordinate_resolution = cordinate_resolution
        self.theta_resolution = theta_resolution
        self.obstacle_kdtree = obstacle_kdtree
        self.obs_x = obs_x
        self.obs_y = obs_y


"""
Calculates environment parameters
"""
def calculateEnvParameters(obs_x, obs_y, cordinate_resolution, theta_resolution):
    env_xmin = round(min(obs_x) / cordinate_resolution)
    env_ymin = round(min(obs_y) / cordinate_resolution)
    env_xmax = round(min(obs_x) / cordinate_resolution)
    env_ymax = round(min(obs_y) / cordinate_resolution)

    obstacle_kdtree = kd.KDTree([[x, y] for x, y in zip(obs_x, obs_y)])

    return getEnvParameters(env_xmin, env_ymin, env_xmax, env_ymax,cordinate_resolution,  theta_resolution, obstacle_kdtree, obs_x, obs_y)  

"""
Generates motion set for traversing with successive vertex
"""
def getMotionPrimitives():

    direction = 1 #Forward direction
    motion_primitives = []

    for primitive in np.arange(AckermannCar.maxSteerAngle, -(AckermannCar.maxSteerAngle/AckermannCar.steerPrecision), -AckermannCar.maxSteerAngle/AckermannCar.steerPrecision):
        motion_primitives.append([primitive, direction])
        motion_primitives.append([primitive, -direction])

    print("Motion Primitives", motion_primitives)
    return motion_primitives

"""
Get all the states of State vector
"""
def index(State):
    # Index is a tuple consisting grid index, used for checking if two nodes are near/same
    return tuple([State.state_id[0], State.state_id[1], State.state_id[2]])

"""
Build obstalces in environment as required in assignment
"""
def buildObstacles(obs_x, obs_y, coordinate_res):

    
    obs_x = [round(x / coordinate_res) for x in obs_x]
    obs_y = [round(y / coordinate_res) for y in obs_y]

    # Set all Grid locations to No Obstacle
    obstacles =[[False for i in range(max(obs_y))]for i in range(max(obs_x))]

    # Set Grid Locations with obstacles to True
    for x in range(max(obs_x)):
        for y in range(max(obs_y)):
            for i, j in zip(obs_x, obs_y):
                if math.hypot(i-x, j-y) <= 1/2:
                    obstacles[i][j] = True
                    break

    return obstacles

"""
Define motion set for unconstrained heuristics 
"""
def unconstrainedMotion():

    unconstrained_motion_command = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
    return unconstrained_motion_command


def eucledianCost(holonomicMotionCommand):
    # Compute Eucledian Distance between two nodes
    return math.hypot(holonomicMotionCommand[0], holonomicMotionCommand[1])

def uncontrainedStateId(unconstraintedState):
    return tuple([unconstraintedState.state_id[0], unconstraintedState.state_id[1]])

"""
Checks whether its within environment
"""
def is_unconstrained_state_valid(neighbour_state, obstacles, env_map):

    # Check if Node is out of map bounds
    if neighbour_state.state_id[0]<= env_map.env_xmin or \
       neighbour_state.state_id[0]>= env_map.env_xmax or \
       neighbour_state.state_id[1]<= env_map.env_ymin or \
       neighbour_state.state_id[1]>= env_map.env_xmax:
        return False

    # Check if Node on obstacle
    if obstacles[neighbour_state.state_id[0]][neighbour_state.state_id[1]]:
        return False

    return True


"""
Provides unconstrained heuristics
"""
def getUnconstrainedHeuristic(goal_state, env_map ):

    state_id = [round(goal_state.trajectory_rollout[-1][0]/env_map.cordinate_resolution), round(goal_state.trajectory_rollout[-1][1]/env_map.cordinate_resolution)]
    goal_state = unconstraintedState(state_id, 0, tuple(state_id))

    obstacles = buildObstacles(env_map.obs_x, env_map.obs_y, env_map.cordinate_resolution)

    unconstrained_motion_command = unconstrainedMotion()


    open_set = {uncontrainedStateId(goal_state): goal_state}
    closed_set = {}

    priority_queue =[]
    heapq.heappush(priority_queue, (goal_state.cost, uncontrainedStateId(goal_state)))

    while True:
        if not open_set:
            break

        _, current_state_id = heapq.heappop(priority_queue)
        current_state = open_set[current_state_id]
        open_set.pop(current_state_id)
        closed_set[current_state_id] = current_state

        for i in range(len(unconstrained_motion_command)):
            neighbour_state = unconstraintedState([current_state.state_id[0] + unconstrained_motion_command[i][0],\
                                      current_state.state_id[1] + unconstrained_motion_command[i][1]],\
                                      current_state.cost + eucledianCost(unconstrained_motion_command[i]), current_state_id)

            if not is_unconstrained_state_valid(neighbour_state, obstacles, env_map):
                continue

            neighbour_state_id = uncontrainedStateId(neighbour_state)

            if neighbour_state_id not in closed_set:            
                if neighbour_state_id in open_set:
                    if neighbour_state_id.cost < open_set[neighbour_state_id].cost:
                        open_set[neighbour_state_id].cost = neighbour_state.cost
                        open_set[neighbour_state_id].parent_id = neighbour_state.parent_id
                        # heapq.heappush(priorityQueue, (neighbourNode.cost, neighbourNodeIndex))
                else:
                    open_set[neighbour_state_id] = neighbour_state
                    heapq.heappush(priority_queue, (neighbour_state.cost, neighbour_state_id))

    unconstrained_cost = [[np.inf for i in range(max(env_map.obs_y))]for i in range(max(env_map.obs_x   ))]

    for nodes in closed_set.values():
        unconstrained_cost[nodes.state_id[0]][nodes.state_id[1]]=nodes.cost

    return unconstrained_cost

"""
Checks collision using KD-tree
"""
def collision(trajectory_rollout, map_env):

    car_radius = (AckermannCar.axleToFront + AckermannCar.axleToBack)/2 + 1
    dl = (AckermannCar.axleToFront - AckermannCar.axleToBack)/2
    for i in trajectory_rollout:
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

"""Check state validaty for environment limits
"""
def is_state_valid(trajectory_rollout, state_id, map_env):

    # Check if Node is out of map bounds
    if state_id[0]<=map_env.env_xmin or state_id[0]>=map_env.env_xmax or \
       state_id[1]<=map_env.env_ymin or state_id[1]>=map_env.env_ymax:
        return False

    # Check if Node is collidingmap_env with an obstacle
    if collision(trajectory_rollout, map_env):
        return False
    return True

"""Calculate path cost consider direction, reverse, steering change cost
"""
def path_cost(current_state, motion_primitive, simulation_length):

    # Previos Node Cost
    cost = current_state.cost

    # Distance cost
    if motion_primitive[1] == 1:
        cost += simulation_length 
    else:
        cost += simulation_length * Cost.reverse

    # Direction change cost
    if current_state.traversal_dir != motion_primitive[1]:
        cost += Cost.directionChange

    # Steering Angle Cost
    cost += motion_primitive[0] * Cost.steerAngle

    # Steering Angle change cost
    cost += abs(motion_primitive[0] - current_state.steeringAngle) * Cost.steerAngleChange

    return cost

"""
Update state using vehicle kinematics
"""
def get_kinematics(current_state, motion_primitive, env_map, simulation_length=4, step = 0.8 ):

        # Simulate node using given current Node and Motion Commands
    traj = []
    angle = rs.pi_2_pi(current_state.trajectory_rollout[-1][2] + motion_primitive[1] * step / AckermannCar.wheelBase * math.tan(motion_primitive[0]))
    traj.append([current_state.trajectory_rollout[-1][0] + motion_primitive[1] * step * math.cos(angle),
                current_state.trajectory_rollout[-1][1] + motion_primitive[1] * step * math.sin(angle),
                rs.pi_2_pi(angle + motion_primitive[1] * step / AckermannCar.wheelBase * math.tan(motion_primitive[0]))])
    for i in range(int((simulation_length/step))-1):
        traj.append([traj[i][0] + motion_primitive[1] * step * math.cos(traj[i][2]),
                    traj[i][1] + motion_primitive[1] * step * math.sin(traj[i][2]),
                    rs.pi_2_pi(traj[i][2] + motion_primitive[1] * step / AckermannCar.wheelBase * math.tan(motion_primitive[0]))])


    state_id = [round(traj[-1][0]/env_map.cordinate_resolution), \
                 round(traj[-1][1]/env_map.cordinate_resolution), \
                 round(traj[-1][2]/env_map.cordinate_resolution)]

    # Check if node is valid
    if not is_state_valid(traj, state_id, env_map):
        return None

    # Calculate Cost of the node
    cost = path_cost(current_state, motion_primitive, simulation_length)

    return State(state_id, traj, motion_primitive[0], motion_primitive[1], cost, index(current_state))


    
"""Calculates cost of Generated reeds_shepp path
"""
def reeds_shepp_cost(current_state, path):

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

    return cost


"""Generate reeds shepp paths from current state to goal
"""
def reeds_shepp_node(current_state, goal_state, env_map):

    # Get x, y, yaw of currentNode and goalNode
    start_x, start_y, start_yaw = current_state.trajectory_rollout[-1][0], current_state.trajectory_rollout[-1][1], current_state.trajectory_rollout[-1][2]
    goal_x, goal_y, goal_yaw = goal_state.trajectory_rollout[-1][0], goal_state.trajectory_rollout[-1][1], goal_state.trajectory_rollout[-1][2]

    # Instantaneous Radius of Curvature
    radius = math.tan(AckermannCar.maxSteerAngle)/AckermannCar.wheelBase

    #  Find all possible reeds-shepp paths between current and goal node
    reeds_shepp_paths = rs.calc_all_paths(start_x, start_y, start_yaw, goal_x, goal_y, goal_yaw, radius, 1)
    # directions = reeds_shepp_paths.directions
    # Check if reedsSheppPaths is empty
    if not reeds_shepp_paths:
        return None

    # Find path with lowest cost considering non-holonomic constraints
    cost_queue = heapdict()
    for path in reeds_shepp_paths:
        cost_queue[path] = reeds_shepp_cost(current_state, path)

    # Find first path in priority queue that is collision free
    while len(cost_queue)!=0:
        path = cost_queue.popitem()[0]
        traj=[]
        traj = [[path.x[k],path.y[k],path.yaw[k]] for k in range(len(path.x))]
        if not collision(traj, env_map):
            cost = reeds_shepp_cost(current_state, path)
            return State(goal_state.state_id ,traj, None, None, cost, index(current_state))
            
    return None
    
"""Back tracks the generated path to provide state vector to goal
"""
def backtrack(start_state, goal_state, closed_set, plt):

    # Goal Node data
    start_state_id = index(start_state)
    current_state_id = goal_state.parent_index
    current_state = closed_set[current_state_id]
    x=[]
    y=[]
    theta=[]

    # Iterate till we reach start node from goal node
    while current_state_id != start_state_id:
        a, b, c = zip(*current_state.trajectory_rollout)
        x += a[::-1] 
        y += b[::-1] 
        theta += c[::-1]
       
        current_state_id = current_state.parent_index
        current_state = closed_set[current_state_id]
    return x[::-1], y[::-1], theta[::-1]

"""Runs Hybrid astar
"""
def hybridAstar(start_pose, goal_pose, env_map, plt):

    start_pose_id = [round(start_pose[0] / env_map.cordinate_resolution), \
                  round(start_pose[1] / env_map.cordinate_resolution), \
                  round(start_pose[2]/env_map.theta_resolution)]

    goal_pose_id = [round(goal_pose[0] / env_map.cordinate_resolution), \
                  round(goal_pose[1] / env_map.cordinate_resolution), \
                  round(goal_pose[2]/env_map.theta_resolution)]

    motion_primitives = getMotionPrimitives()

    start_state = State(start_pose_id, [start_pose], 0, 1, 0, tuple(start_pose_id))
    goal_state = State(goal_pose_id, [goal_pose], 0, 1, 0, tuple(goal_pose_id))

    unconstrained_heuristics = getUnconstrainedHeuristic(goal_state, env_map)

    open_set = {index(start_state):start_state}
    closed_set = {}

    cost_priority_queue = heapdict()


    cost_priority_queue[index(start_state)] = max(start_state.cost, Cost.hybrid_cost * unconstrained_heuristics[start_state.state_id[0]][start_state.state_id[1]])
    counter = 0

    while True:
        counter += 1

        if not open_set:
            return None
        
        current_state_id = cost_priority_queue.popitem()[0]
        current_state = open_set[current_state_id]

        open_set.pop(current_state_id)
        closed_set[current_state_id] = current_state

        rs_node = reeds_shepp_node(current_state, goal_state, env_map)


           # Id Reeds-Shepp Path is found exit
        if rs_node:
            closed_set[index(rs_node)] = rs_node
            break

        if current_state_id == index(goal_state):
            print(current_state.trajectory_rollout[-1])
            break


    
          # Get all simulated Nodes from current node
        for i in range(len(motion_primitives)):
            #Kinematically valid next
            simulated_state = get_kinematics(current_state, motion_primitives[i], env_map)

            # Check if path is within map bounds and is collision free
            if not simulated_state:
                continue

            # Draw Simulated Node
            x,y,z =zip(*simulated_state.trajectory_rollout)
            plt.plot(x, y, linewidth=1, color='g')
            plt.show()

            # Check if simulated node is already in closed set
            simulated_state_id = index(simulated_state)
            if simulated_state_id not in closed_set: 

                # Check if simulated node is already in open set, if not add it open set as well as in priority queue
                if simulated_state_id not in open_set:
                    open_set[simulated_state_id] = simulated_state
                    cost_priority_queue[simulated_state_id] = max(simulated_state.cost , Cost.hybridCost * unconstrained_heuristics[simulated_state.state_id[0]][simulated_state.state_id[1]])
                else:
                    if simulated_state.cost < open_set[simulated_state_id].cost:
                        open_set[simulated_state_id] = simulated_state
                        cost_priority_queue[simulated_state_id] = max(simulated_state.cost , Cost.hybridCost * unconstrained_heuristics[simulated_state.state_id[0]][simulated_state.state_id[1]])

    x, y, yaw = backtrack(start_state, goal_state, closed_set, plt)

    return x, y, yaw