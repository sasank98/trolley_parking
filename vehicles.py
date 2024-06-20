#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import math
from visualise import MapEnv, Arrow

class DiffDrive:
    wheelBase = 2.8 # meters
    axleToFront = 4.8 # L + 1
    axleToBack = 1 # m 
    width = 3.0
    WheelDistance = 0.7 * width
    TyreRadius = 0.5
    TyreWidth = 1.2
    ReartoFront = 4.5  # [m] distance from rear to vehicle front end of vehicle

    ReartoBack = 1.0  # [m] distance from rear to vehicle back end of vehicle
    WD = 1.3 * width  

def drawDiffDrive(x, y, yaw, color='grey'):
    car = np.array([[-DiffDrive.ReartoBack, -DiffDrive.ReartoBack, DiffDrive.ReartoFront, DiffDrive.ReartoFront, -DiffDrive.ReartoBack],
                    [DiffDrive.width / 2, -DiffDrive.width / 2, -DiffDrive.width / 2, DiffDrive.width / 2, DiffDrive.width / 2]])

    wheel = np.array([[-DiffDrive.TyreRadius, -DiffDrive.TyreRadius, DiffDrive.TyreRadius, DiffDrive.TyreRadius, -DiffDrive.TyreRadius],
                      [DiffDrive.TyreWidth / 4, -DiffDrive.TyreWidth / 4, -DiffDrive.TyreWidth / 4, DiffDrive.TyreWidth / 4, DiffDrive.TyreWidth / 4]])

    rlWheel = wheel.copy()
    rrWheel = wheel.copy()
    frWheel = wheel.copy()
    flWheel = wheel.copy()

    Rot1 = np.array([[math.cos(yaw), -math.sin(yaw)],
                     [math.sin(yaw), math.cos(yaw)]])

    frWheel = np.dot(Rot1, frWheel)
    flWheel = np.dot(Rot1, flWheel)

    frWheel += np.array([[DiffDrive.wheelBase], [-DiffDrive.WD / 2]])
    flWheel += np.array([[DiffDrive.wheelBase], [DiffDrive.WD / 2]])
    rrWheel[1, :] -= DiffDrive.WD / 2
    rlWheel[1, :] += DiffDrive.WD / 2

    frWheel = np.dot(Rot1, frWheel)
    flWheel = np.dot(Rot1, flWheel)

    rrWheel = np.dot(Rot1, rrWheel)
    rlWheel = np.dot(Rot1, rlWheel)
    car = np.dot(Rot1, car)

    frWheel += np.array([[x], [y]])
    flWheel += np.array([[x], [y]])
    rrWheel += np.array([[x], [y]])
    rlWheel += np.array([[x], [y]])
    car += np.array([[x], [y]])

    plt.plot(car[0, :], car[1, :], color)
    plt.plot(rrWheel[0, :], rrWheel[1, :], color)
    plt.plot(rlWheel[0, :], rlWheel[1, :], color)



class AckermannCar:    
    maxSteerAngle = 0.6
    steerPrecision = 10
    wheelBase = 2.8 # meters
    axleToFront = 3.8 # L + 1
    axleToBack = 1 # m 
    width = 3.0 # meters
    WheelDistance = 1.3 * width
    TyreRadius = 0.5
    TyreWidth = 1.2


def drawCar(x, y, yaw, color='grey'):
    car = np.array([[-AckermannCar.axleToBack, -AckermannCar.axleToBack, AckermannCar.axleToFront, AckermannCar.axleToFront, -AckermannCar.axleToBack],
                    [AckermannCar.width / 2, -AckermannCar.width / 2, -AckermannCar.width / 2, AckermannCar.width / 2, AckermannCar.width / 2]])
    
    wheel = np.array([[-AckermannCar.TyreRadius, -AckermannCar.TyreRadius, AckermannCar.TyreRadius, AckermannCar.TyreRadius, -AckermannCar.TyreRadius],
                      [AckermannCar.TyreWidth / 4, -AckermannCar.TyreWidth / 4, -AckermannCar.TyreWidth / 4, AckermannCar.TyreWidth / 4, AckermannCar.TyreWidth / 4]])

    rlWheel = wheel.copy()
    rrWheel = wheel.copy()
    frWheel = wheel.copy()
    flWheel = wheel.copy()

    steer = 0
    Rot1 = np.array([[math.cos(yaw), -math.sin(yaw)],
                     [math.sin(yaw), math.cos(yaw)]])

    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    frWheel = np.dot(Rot2, frWheel)
    flWheel = np.dot(Rot2, flWheel)

    frWheel += np.array([[AckermannCar.wheelBase], [-AckermannCar.WheelDistance / 2]])
    flWheel += np.array([[AckermannCar.wheelBase], [AckermannCar.WheelDistance / 2]])
    rrWheel[1, :] -= AckermannCar.WheelDistance / 2
    rlWheel[1, :] += AckermannCar.WheelDistance / 2

    frWheel = np.dot(Rot1, frWheel)
    flWheel = np.dot(Rot1, flWheel)

    rrWheel = np.dot(Rot1, rrWheel)
    rlWheel = np.dot(Rot1, rlWheel)
    car = np.dot(Rot1, car)

    frWheel += np.array([[x], [y]])
    flWheel += np.array([[x], [y]])
    rrWheel += np.array([[x], [y]])
    rlWheel += np.array([[x], [y]])
    car += np.array([[x], [y]])

    plt.plot(car[0, :], car[1, :], color)
    plt.plot(frWheel[0, :], frWheel[1, :], color)
    plt.plot(rrWheel[0, :], rrWheel[1, :], color)
    plt.plot(flWheel[0, :], flWheel[1, :], color)
    plt.plot(rlWheel[0, :], rlWheel[1, :], color)

class TruckTrailer:
    maxSteerAngle = 0.6
    width = 3.0
    wheelBase = 3.0
    WD = 0.7 * width  
    axleWidth = 1.75
    d1 = 5.0
    length_t = 8.0 # Length from truck axle to trailer axle
    axleToFront = 4.5 # L + 1
    axleToBack = 1.0 # m 
    axleTrailerToBack = 9.0
    axleTrailerToFront = 1.0
    TR = 0.5  # [m] tyre radius
    TW = 1.0  # [m] tyre width

    width = 3.0  # [m] width of vehicle
    wheelBase = 3.5  # [m] wheel base: rear to front steer
    WD = 1.3 * width  # [m] distance between left-right wheels
    ReartoFront = 4.5  # [m] distance from rear to vehicle front end of vehicle
    ReartoBack = 1.0  # [m] distance from rear to vehicle back end of vehicle

    RearToTrailerWheel = 8.0  # [m] rear to trailer wheel
    RearToTrailerFront = -3.0  # [m] distance from rear to vehicle front end of trailer
    RearToTrailerBack = 9.0  # [m] distance from rear to vehicle back end of trailer
    TyreRadius = 0.5  # [m] tyre radius
    TyreWidth = 1.2  # [m] tyre width
    maxSteerAngle = 0.6  # [rad] maximum steering angle




def drawTruck(x, y, yaw, yawt, color='grey'):
    car = np.array([[-TruckTrailer.ReartoBack, -TruckTrailer.ReartoBack, TruckTrailer.ReartoFront, TruckTrailer.ReartoFront, -TruckTrailer.ReartoBack],
                    [TruckTrailer.width / 2, -TruckTrailer.width / 2, -TruckTrailer.width / 2, TruckTrailer.width / 2, TruckTrailer.width / 2]])

    trail = np.array([[-TruckTrailer.RearToTrailerBack, -TruckTrailer.RearToTrailerBack, TruckTrailer.RearToTrailerFront, TruckTrailer.RearToTrailerFront, -TruckTrailer.RearToTrailerBack],
                      [TruckTrailer.width / 2, -TruckTrailer.width / 2, -TruckTrailer.width / 2, TruckTrailer.width / 2, TruckTrailer.width / 2]])

    wheel = np.array([[-TruckTrailer.TyreRadius, -TruckTrailer.TyreRadius, TruckTrailer.TyreRadius, TruckTrailer.TyreRadius, -TruckTrailer.TyreRadius],
                      [TruckTrailer.TyreWidth / 4, -TruckTrailer.TyreWidth / 4, -TruckTrailer.TyreWidth / 4, TruckTrailer.TyreWidth / 4, TruckTrailer.TyreWidth / 4]])

    rlWheel = wheel.copy()
    rrWheel = wheel.copy()
    frWheel = wheel.copy()
    flWheel = wheel.copy()
    rltWheel = wheel.copy()
    rrtWheel = wheel.copy()

    steer =0
    Rot1 = np.array([[math.cos(yaw), -math.sin(yaw)],
                     [math.sin(yaw), math.cos(yaw)]])

    Rot2 = np.array([[math.cos(steer), -math.sin(steer)],
                     [math.sin(steer), math.cos(steer)]])

    Rot3 = np.array([[math.cos(yawt), -math.sin(yawt)],
                     [math.sin(yawt), math.cos(yawt)]])

    frWheel = np.dot(Rot2, frWheel)
    flWheel = np.dot(Rot2, flWheel)

    frWheel += np.array([[TruckTrailer.wheelBase], [-TruckTrailer.WD / 2]])
    flWheel += np.array([[TruckTrailer.wheelBase], [TruckTrailer.WD / 2]])
    rrWheel[1, :] -= TruckTrailer.WD / 2
    rlWheel[1, :] += TruckTrailer.WD / 2

    frWheel = np.dot(Rot1, frWheel)
    flWheel = np.dot(Rot1, flWheel)

    rrWheel = np.dot(Rot1, rrWheel)
    rlWheel = np.dot(Rot1, rlWheel)
    car = np.dot(Rot1, car)

    rltWheel += np.array([[-TruckTrailer.RearToTrailerWheel], [TruckTrailer.WD / 2]])
    rrtWheel += np.array([[-TruckTrailer.RearToTrailerWheel], [-TruckTrailer.WD / 2]])

    rltWheel = np.dot(Rot3, rltWheel)
    rrtWheel = np.dot(Rot3, rrtWheel)
    trail = np.dot(Rot3, trail)

    frWheel += np.array([[x], [y]])
    flWheel += np.array([[x], [y]])
    rrWheel += np.array([[x], [y]])
    rlWheel += np.array([[x], [y]])
    rrtWheel += np.array([[x], [y]])
    rltWheel += np.array([[x], [y]])
    car += np.array([[x], [y]])
    trail += np.array([[x], [y]])

    plt.plot(car[0, :], car[1, :], color)
    plt.plot(trail[0, :], trail[1, :], color)
    plt.plot(frWheel[0, :], frWheel[1, :], color)
    plt.plot(rrWheel[0, :], rrWheel[1, :], color)
    plt.plot(flWheel[0, :], flWheel[1, :], color)
    plt.plot(rlWheel[0, :], rlWheel[1, :], color)
    plt.plot(rrtWheel[0, :], rrtWheel[1, :], color)
    plt.plot(rltWheel[0, :], rltWheel[1, :], color)
