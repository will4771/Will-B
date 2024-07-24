import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, quad
from scipy.optimize import minimize
from typing import List, Tuple



def normalize_angle(angle: float) -> float:
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle

def Initial_incident_Road() -> Tuple[List[float], List[float], float, float]:   
    x0 = np.random.uniform(25, 50)
    y0 = np.random.uniform(25, 50)

    x_coords = []
    y_coords = []

    Angle_out = np.random.uniform(-np.pi, np.pi)
    Angle_in = normalize_angle(Angle_out + np.pi)

    delta_x = np.cos(Angle_in)
    delta_y = np.sin(Angle_in)
    t = np.linspace(0, 2, 21)

    x_coords = list(x0 + t * delta_x)
    y_coords = list(y0 + t * delta_y)

    return x_coords, y_coords, Angle_in, Angle_out

def Incident_Plotter(x_points: List[float], y_points: List[float]) -> None:
    plt.plot(x_points, y_points, color='blue')

def Incident_Plotter_Green(x_points: List[float], y_points: List[float]) -> None:
    plt.plot(x_points, y_points, color='blue')

def clothoid_ode_rhs(state, s, kappa0, kappa1):
    x, y, theta = state[0], state[1], state[2]
    return [np.cos(theta), np.sin(theta), kappa0 + kappa1 * s]

def eval_clothoid(x0, y0, theta0, kappa0, kappa1, s):
    return odeint(clothoid_ode_rhs, [x0, y0, theta0], s, args=(kappa0, kappa1))

def Helper_Path_Gen(x0: float, y0: float, theta0: float, L: float) -> Tuple[List[float], List[float], float]:
    kappa0, kappa1 = np.random.uniform(0.005, 0.025), np.random.uniform(0.005, 0.025)
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
        Helper_paths.append(Helper_Path_Gen(Roads[i-1][0][0], Roads[i-1][1][0], Roads[i-1][3], 7))
        
        delta_x = np.cos(Helper_paths[i-1][2])
        delta_y = np.sin(Helper_paths[i-1][2])
        t = np.linspace(0, 2, 21)
        
        x_coords = list(Helper_paths[i-1][0][-1] + t * delta_x)
        y_coords = list(Helper_paths[i-1][1][-1] + t * delta_y)
        
        Roads.append([x_coords, y_coords, Helper_paths[i-1][2], normalize_angle(Helper_paths[i-1][2] - np.pi)])
        
        should_pop = False
        for j in range(len(Roads) - 1):
            if np.abs(Roads[-1][0][0] - Roads[j][0][0]) <= 2 and np.abs(Roads[-1][1][0] - Roads[j][1][0]) <= 2:
                Roads.pop()
                Helper_paths.pop()
                should_pop = True
                break
        if not should_pop:
            i += 1
    
    return Roads

def Clothoid_Curve(a, b, c, d, heading0, heading1):
    x0, y0, x1, y1 = a, b, c, d
    v0, v1 = heading0, heading1 
    Delta_x, Delta_y = x1 - x0, y1 - y0
    Delta_v = normalize_angle(v1 - v0)
    phi = np.arctan2(Delta_y, Delta_x)
    r = np.sqrt(Delta_x**2 + Delta_y**2)
    Delta_Phi =  normalize_angle(v0 - phi)


    def G(A):
        def integrand(tau):
            return np.sin(A * tau**2 + (Delta_v - A) * tau + Delta_Phi)
        integral = quad(integrand, 0, 1)[0]
        return integral

    def dG(A):
        def integrand(tau):
            return np.cos(A * tau**2 + (Delta_v - A) * tau + Delta_Phi) * (tau**2 - tau)
        integral = quad(integrand, 0, 1)[0]
        return integral

    def newton(f, df, x0):
        iterates = [x0]
        for i in range(10):
            iterates.append(iterates[-1] - f(iterates[-1]) / df(iterates[-1]))
        return iterates

    A = newton(G, dG, 1)[-1]

    

    def H(A):
        def integrand(tau):
            return np.sin(A * tau**2 + (Delta_v - A) * tau + Delta_Phi + (np.pi/2))
        integral = quad(integrand, 0, 1)[0]
        return integral


    L = r / H(A)
    k0 = (Delta_v - A) / L
    k1 = (2 * A) / L**2

    def clothoid_curve(x0, y0, theta0, L, kappa0, kappa1, num_points=1000):
        s_vals = np.linspace(0, L, num_points)
        x_vals = np.zeros(num_points)
        y_vals = np.zeros(num_points)
        for i, s in enumerate(s_vals):
            def integrand_x(tau):
                return np.cos(0.5 * kappa1 * tau**2 + kappa0 * tau + theta0)
            def integrand_y(tau):
                return np.sin(0.5 * kappa1 * tau**2 + kappa0 * tau + theta0)
            x_vals[i] = x0 + quad(integrand_x, 0, s)[0]
            y_vals[i] = y0 + quad(integrand_y, 0, s)[0]
        return [x_vals, y_vals]

    return clothoid_curve(x0, y0, v0, L, k0, k1)
   
def Connection_road_plotter(Connection_Roads):
    for road in Connection_Roads:
        plt.plot(road[0], road[1], color='red')

def Connection_road_gen(Incident_roads,Incident_road_index): ## incidnt_road index choses which road you want to generate paths from!
    Roads, I = Incident_roads,Incident_road_index

    temp = Roads[I]
    Roads.pop(I)
    Roads.insert(0,temp)
    Connection_Roads = []

    for j in range(1,len(Incident_Roads)):
        Connection_Roads.append(Clothoid_Curve(Incident_Roads[0][0][0],Incident_Roads[0][1][0],Incident_Roads[j][0][0],Incident_Roads[j][1][0],Incident_Roads[0][3],Incident_Roads[j][2]))

    return Connection_Roads

def Incident_Boundry_Road_gen(a,b,c,d):
    
    x0 = a
    y0 = b
    Angle_in = c
    Angle_out = d

    Theta_L = Angle_out + np.pi / 2
    Theta_R = Angle_out - np.pi / 2

    delta_L_x = np.cos(Theta_L)
    delta_L_y = np.sin(Theta_L)

    delta_R_x = np.cos(Theta_R)
    delta_R_y = np.sin(Theta_R)



    delta_x = np.cos(Angle_in)
    delta_y = np.sin(Angle_in)
    t = np.linspace(0, 2, 21)

    x_coords_L = list( (x0 + (delta_L_x)/2) + t * delta_x)
    y_coords_L = list( (y0 + (delta_L_y)/2) + t * delta_y)

    x_coords_R = list( (x0 + (delta_R_x)/2) + t * delta_x)
    y_coords_R = list( (y0 + (delta_R_y)/2) + t * delta_y)

    return  [x_coords_L,y_coords_L],[x_coords_R, y_coords_R],Angle_in , Angle_out

def Connection_Road_Boundry_gen(Inc_1,Inc_2):

    xL,yL = Clothoid_Curve(Inc_1[0][0][0],Inc_1[0][1][0],Inc_2[1][0][0],Inc_2[1][1][0],Inc_1[3],Inc_2[2])
    xR,yR = Clothoid_Curve(Inc_1[1][0][0],Inc_1[1][1][0],Inc_2[0][0][0],Inc_2[0][1][0],Inc_1[3],Inc_2[2])

    return [xR,yR], [xL,yL]
    
def Connection_boundry_plotter(Connection_Boundry, i ): ### i is which element in connection boundry list
    plt.plot(Connection_Boundry[i][0][0],Connection_Boundry[i][0][1],color = 'green')
    plt.plot(Connection_Boundry[i][1][0],Connection_Boundry[i][1][1],color = 'green')



Incident_Roads = Incident_Road_Gen(3)

### Incident road Center line plot
#for i in range(len(Incident_Roads)):                              
    #Incident_Plotter(Incident_Roads[i][0], Incident_Roads[i][1])

Connection_Roads = []
   
for i in range(len(Incident_Roads)):
    Connection_Roads = Connection_Roads + (Connection_road_gen(Incident_Roads,i))

#Connection_road_plotter(Connection_Roads)

Incident_Roads_boundry = []

for i in range(len(Incident_Roads)):
    Incident_Roads_boundry.append(Incident_Boundry_Road_gen(Incident_Roads[i][0][0],Incident_Roads[i][1][0],Incident_Roads[i][2],Incident_Roads[i][3]))
    Incident_Plotter_Green(Incident_Roads_boundry[i][0][0],Incident_Roads_boundry[i][0][1])
    Incident_Plotter_Green(Incident_Roads_boundry[i][1][0],Incident_Roads_boundry[i][1][1])

Connection_Boundry = [] ##########[ , ....[[XR,YR],[XL,YL], ....]


for i in range(len(Incident_Roads)):
    for j in range(len(Incident_Roads)):
        if i == j:
            break
        Connection_Boundry.append(Connection_Road_Boundry_gen(Incident_Roads_boundry[i],Incident_Roads_boundry[j]))

for i in range(len(Connection_Boundry)):
    Connection_boundry_plotter(Connection_Boundry, i)






plt.show()
