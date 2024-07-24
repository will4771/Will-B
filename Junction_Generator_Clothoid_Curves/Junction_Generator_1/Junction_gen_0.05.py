### Improved all the code from 0.03 version.

### Step 1 just makes our boundry for the junction (trivial square)
### Step 2 makes our incident roads {1. has min distacnes between each new incident road. 2. can't spawn to close to the corners}
### Step 3 makes our helper roads


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

## Step 1 setting our boundary for the junction 
## Boundaries (square) x : {1, 11}, y : {1,11}

fig, ax = plt.subplots()
square = patches.Rectangle((1, 1), 10, 10, edgecolor='red', facecolor='none')
ax.add_patch(square)
plt.xlim(0, 12)
plt.ylim(0, 12)
plt.gca().set_aspect('equal', adjustable='box')  # Ensures equal aspect ratio

## Step 2Incident Roads :  
boundary = [[1, 11], [1, 11]]

def Incident_gen(boundary, N, E, S, W): 
    Z = True
    while Z:
        edge = random.randint(0, 3)
        if edge == 0:  # West
            x1, y1 = boundary[0][0], random.uniform(2, 10)
            x2, y2 = x1 - 1, y1
            if all(np.abs(y1 - w) >= 2.5 for w in W):
                W.append(y1)
                Z = False
        elif edge == 1:  # East
            x1, y1 = boundary[0][1], random.uniform(2, 10)
            x2, y2 = x1 + 1, y1
            if all(np.abs(y1 - n) >= 2.5 for n in N):
                N.append(y1)
                Z = False
        elif edge == 2:  # North
            x1, y1 = random.uniform(2, 10), boundary[1][1]
            x2, y2 = x1, y1 + 1
            if all(np.abs(x1 - e) >= 2.5 for e in E):
                E.append(x1)
                Z = False
        elif edge == 3:  # South
            x1, y1 = random.uniform(2, 10), boundary[1][0]
            x2, y2 = x1, y1 - 1
            if all(np.abs(x1 - s) >= 2.5 for s in S):
                S.append(x1)
                Z = False

    return [x1, y1], [x2, y2], N, E, S, W

def plot_line_2coords_incident(P1, P2):
    plt.plot([P1[0], P2[0]], [P1[1], P2[1]], color='blue')

global Incident_Roads

Incident_Roads = []
N, E, S, W = [], [], [], []

for i in range(8):
    P1, P2, N, E, S, W = Incident_gen(boundary, N, E, S, W)
    plot_line_2coords_incident(P1, P2)
    Incident_Roads.append(P1)



## Step 3 Creating the Helper Roads.

def plot_line_2coords(P1, P2):
    plt.plot([P1[0], P2[0]], [P1[1], P2[1]], color='green')

def Helper_Road_Gen_Linear(incident_roads):
    for i in range(len(incident_roads)):
        for j in range(i + 1, len(incident_roads)):
            plot_line_2coords(incident_roads[i], incident_roads[j])


def hermite_curve(t, P0, P1, M0, M1):
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    return h00*P0 + h10*M0 + h01*P1 + h11*M1



def Parametric_Hermite_spline(P0,P1):

    P0 = np.array(P0)
    P1 = np.array(P1)
    theta0 = 0
    theta1 = np.pi / 2 ## has to change for each side ...

    T0 = np.array([np.cos(theta0), np.sin(theta0)])
    T1 = np.array([np.cos(theta1), np.sin(theta1)])

    d = np.linalg.norm(P1 - P0)
    
    M0 = d * T0
    M1 = d * T1

    t_values = np.linspace(0, 1, 100)
    curve_points = np.array([hermite_curve(t, P0, P1, M0, M1) for t in t_values])

    
    plt.plot(curve_points[:, 0], curve_points[:, 1])


def Helper_Road_Gen_Smooth(incident_roads):
    for i in range(len(incident_roads)):
        for j in range(i + 1, len(incident_roads)):
            Parametric_Hermite_spline(incident_roads[i], incident_roads[j])






Helper_Road_Gen_Smooth(Incident_Roads)

plt.show()

