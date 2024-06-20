#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import math




class MapEnv:
    def __init__(self,width, height) -> None:
        self.width = width
        self.height = height
        self.envXCoord, self.envYCoord = [], []
        self.left_obs_anchor  = [self.width*0.12, self.width*0.05]
        self.right_obs_anchor  = [self.width*0.65, self.width*0.05]
        self.center_obs_anchor = [self.width*0.45, self.width*0.5]
        self.obs_width =  self.width*0.20
        self.obs_height =  self.width*0.1
        self.obs_delta =  self.width*0.25

    def buildEnv(self, vehicle_type):
        
        for row in range(self.height):
            for col in range(self.width):

                #Plotting along X border
                if row == 0:
                    self.envXCoord.append(col)
                    self.envYCoord.append(0)

                #Plotting along Y border
                if col == 0:
                    self.envXCoord.append(0)
                    self.envYCoord.append(row)

                #Plotting along X-Opposite border
                if row == (self.height-1):
                    self.envXCoord.append(col)
                    self.envYCoord.append(self.height-1)

                #Plotting along Y-Opposite border
                if col == (self.width-1):
                    self.envXCoord.append(self.width-1)
                    self.envYCoord.append(row)

                #Plotting right obstacle
                if col >= self.right_obs_anchor[0] and col <= (self.right_obs_anchor[0] + self.obs_width ):
                    if row >= self.right_obs_anchor[1] and row <= (self.right_obs_anchor[1] + self.obs_height):
                        self.envXCoord.append(col)
                        self.envYCoord.append(row)

                #Plotting center obstacle
                if col >= self.center_obs_anchor[0] and col <= (self.center_obs_anchor[0] + self.obs_width + self.width*0.20):
                    if row >= self.center_obs_anchor[1] and row <= (self.center_obs_anchor[1] + self.obs_height +self.obs_delta):
                        self.envXCoord.append(col)
                        self.envYCoord.append(row)
            

                if vehicle_type == "car" or "diffdrive":
                    #Plotting left obstacle
                    if col >= self.left_obs_anchor[0] and col <= (self.left_obs_anchor[0] + self.obs_width):
                        if row >= self.left_obs_anchor[1] and row <= (self.left_obs_anchor[1] + self.obs_height):
                            self.envXCoord.append(col)
                            self.envYCoord.append(row)

                   
                

class Arrow:
    def __init__(self, x, y, theta, L, c):
        angle = np.deg2rad(30)
        d = 0.3 * L
        w = 1

        x_start = x
        y_start = y
        x_end = x + L * np.cos(theta)
        y_end = y + L * np.sin(theta)

        theta_hat_L = theta + math.pi - angle
        theta_hat_R = theta + math.pi + angle

        x_hat_start = x_end
        x_hat_end_L = x_hat_start + d * np.cos(theta_hat_L)
        x_hat_end_R = x_hat_start + d * np.cos(theta_hat_R)

        y_hat_start = y_end
        y_hat_end_L = y_hat_start + d * np.sin(theta_hat_L)
        y_hat_end_R = y_hat_start + d * np.sin(theta_hat_R)

        plt.plot([x_start, x_end], [y_start, y_end], color=c, linewidth=w)
        plt.plot([x_hat_start, x_hat_end_L],
                 [y_hat_start, y_hat_end_L], color=c, linewidth=w)
        plt.plot([x_hat_start, x_hat_end_R],
                 [y_hat_start, y_hat_end_R], color=c, linewidth=w)
