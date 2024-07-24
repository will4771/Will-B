import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, quad
from scipy.optimize import minimize
from typing import List, Tuple
import random

fig, ax = plt.subplots()
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.set_aspect('equal')

def normalize_angle(angle: float) -> float:
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle

def Initial_incident_Road() -> Tuple[List[float], List[float], float, float]:   
    Incident_Road_0 = []

    x0 = np.random.uniform(25, 50)
    y0 = np.random.uniform(25, 50)

    Angle_out = np.random.uniform(-np.pi, np.pi)
    Angle_in = normalize_angle(Angle_out + np.pi)
    delta_x = np.cos(Angle_in)
    delta_y = np.sin(Angle_in)

    Theta = normalize_angle(np.pi / 2 - Angle_out)
    delta_x1 = np.cos(Theta) * 2
    delta_y1 = np.sin(Theta) * 2
    
    def Coord(x0, y0):
        t = np.linspace(0, 2, 21)
        x_coords = list(x0 + t * delta_x)
        y_coords = list(y0 + t * delta_y)
        return x_coords, y_coords
    
    Number_of_Incident_Roads = random.randint(1, 2)  # 1, 2 or 3 new lanes
    Incident_Road_0.append(Coord(x0, y0))

    for _ in range(Number_of_Incident_Roads):
        x0 = x0 + delta_x1 + 1 * delta_x
        y0 = y0 + delta_y1 + 1 * delta_y
        Incident_Road_0.append(Coord(x0, y0))

    Incident_Road_0.append(Angle_in)
    Incident_Road_0.append(Angle_out)
    Incident_Road_0.append(Number_of_Incident_Roads + 1)
    return Incident_Road_0

def Incident_Plotter_Green(Incident_roads):  # has to be Incident_roads[i] inputted
    Count = Incident_roads[-1]
    for i in range(Count):
        plt.plot(Incident_roads[i][0], Incident_roads[i][1], color='green')

def clothoid_ode_rhs(state, s, kappa0, kappa1):
    x, y, theta = state[0], state[1], state[2]
    return [np.cos(theta), np.sin(theta), kappa0 + kappa1 * s]

def eval_clothoid(x0, y0, theta0, kappa0, kappa1, s):
    return odeint(clothoid_ode_rhs, [x0, y0, theta0], s, args=(kappa0, kappa1))

def Helper_Path_Gen(x0: float, y0: float, theta0: float, L: float) -> Tuple[List[float], List[float], float]:
    kappa0, kappa1 = np.random.uniform(0.025, 0.025), np.random.uniform(0.025, 0.025)
    s = np.linspace(0, L, 1000)

    sol = eval_clothoid(x0, y0, theta0, kappa0, kappa1, s)

    xs, ys, thetas = sol[:, 0], sol[:, 1], sol[:, 2]
    return xs, ys, thetas[-1]

def Incident_Road_Gen(Number_of_incident_Roads: int) -> List:
    Roads = []
    Helper_paths = []
    Roads.append(Initial_incident_Road())

    i = 1  # Start from 1 to avoid index error
    while i < Number_of_incident_Roads:
        temp = []
        
        Helper_paths.append(Helper_Path_Gen(Roads[i-1][1][0][0], Roads[i-1][1][1][0], Roads[i-1][-2], 10)) ###< 10 can't do 4 incident
        
        x0 = Helper_paths[i-1][0][-1]
        y0 = Helper_paths[i-1][1][-1]
        Angle_in = Helper_paths[i-1][2]
        Angle_out = normalize_angle(Helper_paths[i-1][2] - np.pi)

        delta_x = np.cos(Angle_out)
        delta_y = np.sin(Angle_out)

        Theta = normalize_angle(np.pi / 2 - Angle_out)
        delta_x1 = np.cos(Theta) *2
        delta_y1 = np.sin(Theta) * 2
        
        def Coord(x0, y0):
            t = np.linspace(0, 2, 21)
            x_coords = list(x0 - t * delta_x)
            y_coords = list(y0 - t * delta_y)
            return x_coords, y_coords
        
        Number_of_Incident_Roads = random.randint(1, 2)  # 1, 2 or 3 new lanes
        temp.append(Coord(x0, y0))

        if Number_of_Incident_Roads == 2:
            temp.append(Coord((x0 + delta_x1 + 1 * delta_x),(y0 + delta_y1 + 1 * delta_y)))
            temp.append(Coord((x0 - delta_x1 - 1 * delta_x),(y0 - delta_y1 - 1 * delta_y)))
        else:
            temp.append(Coord((x0 + delta_x1 + 1 * delta_x),(y0 + delta_y1 + 1 * delta_y)))
               

        temp.append(Angle_in)
        temp.append(Angle_out)
        temp.append(Number_of_Incident_Roads + 1)

        Roads.append(temp)
        
        should_pop = False
    
        for j in range(len(Roads) - 1):
            if (np.abs(Roads[-1][1][0][0] - Roads[j][1][0][0]) <= 2 and np.abs(Roads[-1][1][1][0] - Roads[j][1][1][0]) <= 2):
                Roads.pop()
                Helper_paths.pop()
                should_pop = True
                break
                
        if not should_pop:
            i += 1
    
    return Roads

Incident_Roads = Incident_Road_Gen(4)

for i in range(len(Incident_Roads)):
    Incident_Plotter_Green(Incident_Roads[i])


plt.plot(Incident_Roads[0][0][0][0],Incident_Roads[0][0][1][0], 'ro')
plt.plot(Incident_Roads[0][1][0][0],Incident_Roads[0][1][1][0], 'bo')

plt.show()

