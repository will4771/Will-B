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

def Initial_incident_Road() -> Tuple[List[List[float]], List[List[float]], float, float, int]:
    Incident_Road_Temp = []
    
    x0 = np.random.uniform(25, 50)
    y0 = np.random.uniform(25, 50)
    t = np.linspace(0, 2, 21)

    Angle_out = np.random.uniform(-np.pi, np.pi)
    Angle_in = normalize_angle(Angle_out + np.pi)

    Theta_L = Angle_out + np.pi / 2
    Theta_R = Angle_out - np.pi / 2

    delta_L_x = np.cos(Theta_L)
    delta_L_y = np.sin(Theta_L)

    delta_R_x = np.cos(Theta_R)
    delta_R_y = np.sin(Theta_R)

    delta_x = np.cos(Angle_in)
    delta_y = np.sin(Angle_in)
    
    x_coords = list(x0 + t * delta_x)
    y_coords = list(y0 + t * delta_y)

    Incident_Road_Temp.append([x_coords, y_coords])

    Number_of_Incident_Roads = random.randint(1, 2)  # 1, 2 or 3 new lanes

    if Number_of_Incident_Roads == 1:
        Incident_Road_Temp.append([list((x0 + 2 * delta_L_x ) + t * delta_x), list((y0 + 2* delta_L_y ) + t * delta_y)])
    if Number_of_Incident_Roads == 2:
        Incident_Road_Temp.append([list((x0 + 2 * delta_L_x ) + t * delta_x), list((y0 + 2 * delta_L_y ) + t * delta_y)])
        Incident_Road_Temp.append([list((x0 + 2 * delta_R_x ) + t * delta_x), list((y0 + 2 * delta_R_y ) + t * delta_y)])

    Incident_Road_Temp.append(Angle_in)
    Incident_Road_Temp.append(Angle_out)
    Incident_Road_Temp.append(Number_of_Incident_Roads + 1)

    return Incident_Road_Temp

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
        t = np.linspace(0, 2, 21)
        
        Helper_paths.append(Helper_Path_Gen(Roads[i-1][1][0][0], Roads[i-1][1][1][0], Roads[i-1][-2], 10)) ###< 10 can't do 4 incident
        
        x0 = Helper_paths[i-1][0][-1]
        y0 = Helper_paths[i-1][1][-1]
        Angle_in = normalize_angle(Helper_paths[i-1][2])
        Angle_out = normalize_angle(Helper_paths[i-1][2] - np.pi)

        Theta_L = Angle_out + np.pi / 2
        Theta_R = Angle_out - np.pi / 2

        delta_L_x = np.cos(Theta_L)
        delta_L_y = np.sin(Theta_L)

        delta_R_x = np.cos(Theta_R)
        delta_R_y = np.sin(Theta_R)

        delta_x = np.cos(Angle_in)
        delta_y = np.sin(Angle_in)
        
        x_coords = list(x0 + t * delta_x)
        y_coords = list(y0 + t * delta_y)

        temp.append([x_coords, y_coords])

        Number_of_Incident_Roads = random.randint(1, 2)  # 1, 2 or 3 new lanes

        if Number_of_Incident_Roads == 1:
            temp.append([list((x0 + 2 * delta_L_x ) + t * delta_x), list((y0 + 2* delta_L_y ) + t * delta_y)])
        if Number_of_Incident_Roads == 2:
            temp.append([list((x0 + 2 * delta_L_x ) + t * delta_x), list((y0 + 2 * delta_L_y ) + t * delta_y)])
            temp.append([list((x0 + 2 * delta_R_x ) + t * delta_x), list((y0 + 2 * delta_R_y ) + t * delta_y)])

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

def Connection_road_gen(Incident_roads,Incident_road_index): ## incidnt_road index choses which road you want to generate paths from!
    Roads, I = Incident_roads,Incident_road_index

    
    Connection_Roads = []

    for j in range(1,len(Incident_Roads)):
        if I ==j:
            break
        
        if Roads[i][-1] == 3 and Roads[j][-1] == 3:
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][0][0],Roads[I][1][1][0],Roads[j][2][0][0],Roads[j][2][1][0],Roads[I][-2],Roads[j][-3]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][0][0],Roads[I][0][1][0],Roads[j][0][0][0],Roads[j][0][1][0],Roads[I][-2],Roads[j][-3]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][2][0][0],Roads[I][2][1][0],Roads[j][1][0][0],Roads[j][1][1][0],Roads[I][-2],Roads[j][-3]))

        if Roads[i][-1] == 3 and Roads[j][-1] == 2 :
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][0][0],Roads[I][1][1][0],Roads[j][0][0][0],Roads[j][0][1][0],Roads[I][-2],Roads[j][-3]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][0][0],Roads[I][0][1][0],Roads[j][0][0][0],Roads[j][0][1][0],Roads[I][-2],Roads[j][-3]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][2][0][0],Roads[I][2][1][0],Roads[j][1][0][0],Roads[j][1][1][0],Roads[I][-2],Roads[j][-3]))

        if Roads[i][-1] == 2 and Roads[j][-1] == 3 :
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][0][0],Roads[I][1][1][0],Roads[j][2][0][0],Roads[j][2][1][0],Roads[I][-2],Roads[j][-3]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][0][0],Roads[I][0][1][0],Roads[j][0][0][0],Roads[j][0][1][0],Roads[I][-2],Roads[j][-3]))
        if Roads[i][-1] == 2 and Roads[j][-1] == 2 : ###correct
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][0][0],Roads[I][1][1][0],Roads[j][0][0][0],Roads[j][0][1][0],Roads[I][-2],Roads[j][-3]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][0][0],Roads[I][0][1][0],Roads[j][1][0][0],Roads[j][1][1][0],Roads[I][-2],Roads[j][-3]))   
    return Connection_Roads

def Connection_road_plotter(Connection_Roads):
    for road in Connection_Roads:
        plt.plot(road[0], road[1], color='white')

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

    x_coords_L = list( (x0 + (delta_L_x)) + t * delta_x)
    y_coords_L = list( (y0 + (delta_L_y)) + t * delta_y)

    x_coords_R = list( (x0 + (delta_R_x)) + t * delta_x)
    y_coords_R = list( (y0 + (delta_R_y)) + t * delta_y)

    return  [x_coords_L,y_coords_L],[x_coords_R, y_coords_R],Angle_in , Angle_out

def Incident_Plotter_boun(x_points: List[float], y_points: List[float]) -> None:
    plt.plot(x_points, y_points, color='green')

def Connection_Road_Boundry_gen(Inc_1,Inc_2):

    xL,yL = Clothoid_Curve(Inc_1[0][0][0],Inc_1[0][1][0],Inc_2[1][0][0],Inc_2[1][1][0],Inc_1[-1],Inc_2[-2])
    xR,yR = Clothoid_Curve(Inc_1[1][0][0],Inc_1[1][1][0],Inc_2[0][0][0],Inc_2[0][1][0],Inc_1[-1],Inc_2[-2])

    return [xR,yR], [xL,yL]
    
def Connection_boundry_plotter(Connection_Boundry, i ): ### i is which element in connection boundry list
    plt.plot(Connection_Boundry[i][0][0],Connection_Boundry[i][0][1],color = 'white')
    plt.plot(Connection_Boundry[i][1][0],Connection_Boundry[i][1][1],color = 'white')

def Connection_Boundry_gen(Incident_boundry,Incident_road_index,Incident_Roads): ## incidnt_road index choses which road you want to generate paths from!
    Roads, I = Incident_boundry,Incident_road_index

    
    Connection_Roads = []

    for j in range(len(Incident_Roads)):
        if I ==j:
            break
        
        if Incident_Roads[i][-1] == 3 and Incident_Roads[j][-1] == 3:
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][0][0][0],Roads[I][1][0][1][0],Roads[j][2][1][0][0],Roads[j][2][1][1][0],Roads[I][0][-1],Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][1][0][0],Roads[I][1][1][1][0],Roads[j][2][0][0][0],Roads[j][2][0][1][0],Roads[I][0][-1],Roads[j][0][-2]))

            Connection_Roads.append(Clothoid_Curve(Roads[I][0][0][0][0],Roads[I][0][0][1][0],Roads[j][0][1][0][0],Roads[j][0][1][1][0],Roads[I][0][-1],Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][1][0][0],Roads[I][0][1][1][0],Roads[j][0][0][0][0],Roads[j][0][0][1][0],Roads[I][0][-1],Roads[j][0][-2]))
            
            Connection_Roads.append(Clothoid_Curve(Roads[I][2][0][0][0],Roads[I][2][0][1][0],Roads[j][1][1][0][0],Roads[j][1][1][1][0],Roads[I][0][-1],Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][2][1][0][0],Roads[I][2][1][1][0],Roads[j][1][0][0][0],Roads[j][1][0][1][0],Roads[I][0][-1],Roads[j][0][-2]))
            
        if Incident_Roads[i][-1] == 3 and Incident_Roads[j][-1] == 2 :
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][0][0][0],Roads[I][1][0][1][0],Roads[j][0][1][0][0],Roads[j][0][1][1][0],Roads[I][0][-1],Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][1][0][0],Roads[I][1][1][1][0],Roads[j][0][0][0][0],Roads[j][0][0][1][0],Roads[I][0][-1],Roads[j][0][-2]))

            Connection_Roads.append(Clothoid_Curve(Roads[I][0][0][0][0],Roads[I][0][0][1][0],Roads[j][0][1][0][0],Roads[j][0][1][1][0],Roads[I][0][-1],Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][1][0][0],Roads[I][0][1][1][0],Roads[j][0][0][0][0],Roads[j][0][0][1][0],Roads[I][0][-1],Roads[j][0][-2]))

            Connection_Roads.append(Clothoid_Curve(Roads[I][2][0][0][0],Roads[I][2][0][1][0],Roads[j][1][1][0][0],Roads[j][1][1][1][0],Roads[I][0][-1],Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][2][1][0][0],Roads[I][2][1][1][0],Roads[j][1][0][0][0],Roads[j][1][0][1][0],Roads[I][0][-1],Roads[j][0][-2]))
            

        if Incident_Roads[i][-1] == 2 and Incident_Roads[j][-1] == 3 :
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][0][0][0],Roads[I][1][0][1][0],Roads[j][2][1][0][0],Roads[j][2][1][1][0],Roads[I][0][-1],Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][1][0][0],Roads[I][1][1][1][0],Roads[j][2][0][0][0],Roads[j][2][0][1][0],Roads[I][0][-1],Roads[j][0][-2]))

            Connection_Roads.append(Clothoid_Curve(Roads[I][0][0][0][0],Roads[I][0][0][1][0],Roads[j][0][1][0][0],Roads[j][0][1][1][0],Roads[I][0][-1],Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][1][0][0],Roads[I][0][1][1][0],Roads[j][0][0][0][0],Roads[j][0][0][1][0],Roads[I][0][-1],Roads[j][0][-2]))

            
        if Incident_Roads[i][-1] == 2 and Incident_Roads[j][-1] == 2 : ###correct
            
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][0][0][0],Roads[I][1][0][1][0],Roads[j][0][1][0][0],Roads[j][0][1][1][0],Roads[I][0][-1],Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][1][0][0],Roads[I][1][1][1][0],Roads[j][0][0][0][0],Roads[j][0][0][1][0],Roads[I][0][-1],Roads[j][0][-2]))
            
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][0][0][0],Roads[I][0][0][1][0],Roads[j][1][1][0][0],Roads[j][1][1][1][0],Roads[I][0][-1],Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][1][0][0],Roads[I][0][1][1][0],Roads[j][1][0][0][0],Roads[j][1][0][1][0],Roads[I][0][-1],Roads[j][0][-2]))

            
    return Connection_Roads    
            
    
            
    








Incident_Roads = Incident_Road_Gen(4)

#for i in range(len(Incident_Roads)):
#    Incident_Plotter_Green(Incident_Roads[i])

Connection_Roads = []

for i in range(len(Incident_Roads)):
    Connection_Roads = Connection_Roads + (Connection_road_gen(Incident_Roads,i))

#Connection_road_plotter(Connection_Roads)


Incident_Roads_boundry = []

for i in range(len(Incident_Roads)):
    temp = []
    for j in range(Incident_Roads[i][-1]):
        temp.append(Incident_Boundry_Road_gen(Incident_Roads[i][j][0][0],Incident_Roads[i][j][1][0],Incident_Roads[i][-3],Incident_Roads[i][-2]))
    Incident_Roads_boundry.append(temp)

for j in range(len(Incident_Roads)):
    for i in range(len(Incident_Roads_boundry[j])):
        Incident_Plotter_boun(Incident_Roads_boundry[j][i][0][0],Incident_Roads_boundry[j][i][0][1])
        Incident_Plotter_boun(Incident_Roads_boundry[j][i][1][0],Incident_Roads_boundry[j][i][1][1])


Connection_Boundry = [] ##########[ , ....[[XR,YR],[XL,YL], ....]


for i in range(len(Incident_Roads)):
    Connection_Boundry = Connection_Boundry + ( Connection_Boundry_gen(Incident_Roads_boundry,i,Incident_Roads))   

Connection_road_plotter(Connection_Boundry)




print(Incident_Roads[0][-1])

plt.plot(Incident_Roads_boundry[0][0][0][0][0],Incident_Roads_boundry[0][0][0][1][0], 'ro')

plt.plot(Incident_Roads_boundry[0][0][1][0][0],Incident_Roads_boundry[0][0][1][1][0], 'bo')



plt.plot(Incident_Roads_boundry[1][0][0][0][0],Incident_Roads_boundry[1][0][0][1][0], 'yo')

plt.plot(Incident_Roads_boundry[1][0][1][0][0],Incident_Roads_boundry[1][0][1][1][0], 'go')
















plt.show()

