import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def Para_plotter_blue(x, y):
    plt.plot(x, y, color='blue')

def Para_plotter_green(x, y):
    plt.plot(x, y, color='green')

def Inital_Incident_Road_Generator():
    x0 = np.random.uniform(25, 50)
    y0 = np.random.uniform(25, 50)
    Heading = np.random.uniform(np.pi / 3, 2 * np.pi / 3)

    delta_x = np.cos(Heading)
    delta_y = np.sin(Heading)

    t = np.linspace(0, 5, 100)
    x = x0 - t * delta_x 
    y = y0 - t * delta_y

    return [x, y, Heading]

def clothoid_ode_rhs(state, s, kappa0, kappa1):
    x, y, theta = state[0], state[1], state[2]
    return np.array([np.cos(theta), np.sin(theta), kappa0 + kappa1 * s])

def eval_clothoid(x0, y0, theta0, kappa0, kappa1, s):
    return odeint(clothoid_ode_rhs, np.array([x0, y0, theta0]), s, (kappa0, kappa1))

def Helper_Path_Gen(a, b, c):
    x0, y0, theta0 = a, b, c
    L = 10
    kappa0, kappa1 = np.random.uniform(0.025, 0.005), np.random.uniform(0.025, 0.005)
    s = np.linspace(0, L, 1000)

    sol = eval_clothoid(x0, y0, theta0, kappa0, kappa1, s)

    xs, ys, thetas = sol[:, 0], sol[:, 1], sol[:, 2] 
    return xs, ys, thetas[-1]

def Incident_Road_Gen(Helper_path):
    x0, y0, heading = Helper_path[0][-1], Helper_path[1][-1], Helper_path[2]
    delta_x = np.cos(heading)
    delta_y = np.sin(heading)

    t = np.linspace(0, 5, 100)
    x = x0 + t * delta_x 
    y = y0 + t * delta_y

    return [x, y, heading + np.pi]

import numpy as np

def Incident_Map_Gen(Num):
    Incident_Roads, Helper_Paths = [], []
    Incident_Roads.append(Inital_Incident_Road_Generator())
    
    i = 1
    while i <= Num:
        Helper_Paths.append(Helper_Path_Gen(Incident_Roads[i-1][0][0], Incident_Roads[i-1][1][0], Incident_Roads[i-1][2]))
        Incident_Roads.append(Incident_Road_Gen(Helper_Paths[i-1]))
        should_pop = False
        for j in range(len(Incident_Roads) - 1):
            if np.abs(Incident_Roads[-1][0][0] - Incident_Roads[j][0][0]) <= 2.5 and np.abs(Incident_Roads[-1][1][0] - Incident_Roads[j][1][0]) <= 2.5:
                Incident_Roads.pop()
                Helper_Paths.pop()
                should_pop = True
                break
        
        if not should_pop:
            i += 1
        
    return Incident_Roads, Helper_Paths

# Assuming Inital_Incident_Road_Generator, Helper_Path_Gen, and Incident_Road_Gen are defined elsewhere

def Incident_Map_Plot(Incident_Roads, Helper_paths):
    for i in range(len(Incident_Roads)):
        Para_plotter_blue(Incident_Roads[i][0], Incident_Roads[i][1])
    for j in range(len(Helper_paths)):
        Para_plotter_green(Helper_paths[j][0], Helper_paths[j][1])


Incident_Roads, Helper_paths = Incident_Map_Gen(4) ### 4 + 1 at the momment

Incident_Map_Plot(Incident_Roads, Helper_paths)

plt.show()

print(Incident_Roads[-1][0][0])
