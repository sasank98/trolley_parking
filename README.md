# Valet Parking Motion Planning

Uses Hybrid A star for kino-dynamic constrained motion planning for parking valet between the two other vehicles shown as black blocks.

Uses motion primivites trajectory rollout to check cost of obstacles with Reeds Shep Path curve generator, backtracks using Hybrid AStar considering holonomic and non holonomic constraints. 

Objectives include to park following

- Differential drive delivery robot.
- Ackermann Steering Car
- Truck Trailer      

## Instructions to run the Project

change the hyper-parameters in the valet_parking.py file accordingly

```
    ENV_WIDTH = 41
    ENV_HEIGHT = 41
    VEHICLE_TYPE = "diffdrive"
    ARROW_LEN = 3
```

run the valet_parking.py file

```
python3 valet_parking.py
```



To the run the valet-parking please use python3 valet_parking.py
set the specific vehicle you wish to run 

1. 

![](./diffdrive.gif)


2.

![](./car.gif)


1. 

![](truck_trailer.gif)