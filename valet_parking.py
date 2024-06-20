#!/usr/bin/env python3

from time import sleep
import numpy as np
import math

import matplotlib.pyplot as plt

from visualise import MapEnv, Arrow
import vehicles
from hybrid_astar import*
from truck_trailer_planning import*

ENV_WIDTH = 41
ENV_HEIGHT = 41
VEHICLE_TYPE = "diffdrive"
ARROW_LEN = 3


def main():


    start_pose = [3.0, 35.0, np.deg2rad(0)]
    goal_pose = [21.0, 5.0, np.deg2rad(0)]

    if VEHICLE_TYPE=="trucktrailer":
        start_pose = [10.0, 35.0, np.deg2rad(0), np.deg2rad(0)]
        goal_pose = [20.0, 5.0, np.deg2rad(0), np.deg2rad(0)]
        
    map = MapEnv(ENV_WIDTH, ENV_HEIGHT)
    map.buildEnv(VEHICLE_TYPE)

    map_env = calculateEnvParameters(map.envXCoord, map.envYCoord, 4, np.deg2rad(15.0))

    if VEHICLE_TYPE == "diffdrive":
        x, y, yaw = hybridAstar(start_pose, goal_pose, map_env, plt)

    if VEHICLE_TYPE=="car":
        x, y, yaw = hybridAstar(start_pose, goal_pose, map_env, plt)

    if VEHICLE_TYPE=="trucktrailer":
        x, y, yaw, yaw_t = hybridAstar_trailer(start_pose, goal_pose, map_env, plt)
    
    mng = plt.get_current_fig_manager()
 
 

    for k in range(len(x)):
        plt.cla()
        plt.xlim(min(map.envXCoord), max(map.envXCoord)) 
        plt.ylim(min(map.envYCoord), max(map.envYCoord))

    
        if VEHICLE_TYPE=="trucktrailer":            
            vehicles.drawTruck(x[k], y[k], yaw[k], yaw_t[k],  'black', )
            vehicles.drawTruck(start_pose[0], start_pose[1], start_pose[2], start_pose[3], 'green')
            vehicles.drawTruck(goal_pose[0], goal_pose[1], goal_pose[2],  goal_pose[3],'red')

        if VEHICLE_TYPE=="diffdrive":
            vehicles.drawDiffDrive(start_pose[0], start_pose[1], start_pose[2], 'green')
            vehicles.drawDiffDrive(goal_pose[0], goal_pose[1], goal_pose[2], 'red')
            vehicles.drawDiffDrive(x[k], y[k], yaw[k], 'grey' )

        if VEHICLE_TYPE=="car":
            vehicles.drawCar(start_pose[0], start_pose[1], start_pose[2], 'green')
            vehicles.drawCar(goal_pose[0], goal_pose[1], goal_pose[2], 'red')
            vehicles.drawCar(x[k], y[k], yaw[k], 'grey' )

        plt.arrow(x[k],y[k],  ARROW_LEN*math.cos(yaw[k]),  ARROW_LEN*math.sin(yaw[k]), width=0.05)

        plt.plot(map.envXCoord, map.envYCoord, "sk")
        plt.plot(x, y, linewidth=1.5, color='r', zorder=1)
        plt.axis('equal')
        plt.title("Valet Parking " + str(VEHICLE_TYPE))
        plt.pause(0.1)
        
    plt.show()




if __name__ == '__main__':
    main()