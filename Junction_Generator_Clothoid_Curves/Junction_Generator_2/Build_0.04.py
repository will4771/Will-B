import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.integrate import quad
from scipy.optimize import minimize

def Para_plotter_blue(x, y):
    plt.plot(x, y, color='blue')

def Para_plotter_green(x, y):
    plt.plot(x, y, color='green')

def Inital_Incident_Road_Generator():
    x0 = np.random.uniform(25, 50)
    y0 = np.random.uniform(25, 50)
    Heading = np.random.uniform(-np.pi /2 ,np.pi/2)

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

def Incident_Map_Plot(Incident_Roads, Helper_paths):
    for i in range(len(Incident_Roads)):
        Para_plotter_blue(Incident_Roads[i][0], Incident_Roads[i][1])
    #for j in range(len(Helper_paths)):
        #Para_plotter_green(Helper_paths[j][0], Helper_paths[j][1])

def Clothoid_Curve(a,b,c,d,heading0,heading1): #x0,y0,x1,x2,heading0,heading1
    x0 = a
    y0 = b
    x1 = c
    y1 = d

    v0 = heading0
    v1 = heading1 - np.pi

    Delta_x = x1 - x0
    Delta_y = y1 - y0
    Delta_v = v1 - v0

    def Ga(vars):
        L, A = vars

        def Integrand_x(tau):
            return np.cos(A * tau**2 + (Delta_v - A) * tau + v0)

        def Integrand_y(tau):
            return np.sin(A * tau**2 + (Delta_v - A) * tau + v0)

        integral_x = quad(Integrand_x, 0, 1)[0]
        integral_y = quad(Integrand_y, 0, 1)[0]

        Eq1 = Delta_x - L * integral_x
        Eq2 = Delta_y - L * integral_y

        return Eq1**2 + Eq2**2  # Squared sum to minimize L

    # Initial guess for the variables (L, A)
    initial_guess = [1, 1]

    # Define bounds for L and A
    bounds = [(None, None), (None, None)]  # L >= 0, A can be any value

    # Perform minimization
    result = minimize(Ga, initial_guess, bounds=bounds)
    L, A = result.x

    

    # Function to find kappa0 and kappa1
    def Kappa_finder(L, A):
        kappa0 = (Delta_v - A) / L
        kappa1 = 2 * A / L**2
        return kappa0, kappa1

    k0, k1 = Kappa_finder(L, A)

    # Function to generate clothoid curve
    def clothoid_curve(x0, y0, theta0, L, kappa, kappa0, num_points=1000):
        s_vals = np.linspace(0, L, num_points)
        x_vals = np.zeros(num_points)
        y_vals = np.zeros(num_points)

        for i, s in enumerate(s_vals):
            def integrand_x(tau):
                return np.cos(0.5 * kappa0 * tau**2 + kappa * tau + theta0)

            def integrand_y(tau):
                return np.sin(0.5 * kappa0 * tau**2 + kappa * tau + theta0)

            x_vals[i] = x0 + quad(integrand_x, 0, s)[0]
            y_vals[i] = y0 + quad(integrand_y, 0, s)[0]

        return [x_vals, y_vals]


    x_vals, y_vals = clothoid_curve(x0, y0, v0, L, k0, k1)

    return x_vals, y_vals

def Connection_road_plotter(Connection_Roads):
    for i in range(len(Connection_Roads)):
        plt.plot(Connection_Roads[i][0],Connection_Roads[i][1],color='red' )



def angle_fix(angle):
    if angle >= np.pi:
        return  angle - np.pi
    elif angle < np.pi:
        return angle + np.pi

def Connection_road_gen(Incident):
    Connection_Roads = []
    
    
    
    
    Connection_Roads.append(Clothoid_Curve(Incident_Roads[0][0][0],Incident_Roads[0][1][0],Incident_Roads[1][0][0],Incident_Roads[1][1][0], Incident_Roads[0][2], angle_fix(Helper_paths[0][2]))) 
    Connection_Roads.append(Clothoid_Curve(Incident_Roads[0][0][0],Incident_Roads[0][1][0],Incident_Roads[3][0][0],Incident_Roads[3][1][0], Incident_Roads[0][2], angle_fix(Helper_paths[2][2]))) 
    Connection_Roads.append(Clothoid_Curve(Incident_Roads[0][0][0],Incident_Roads[0][1][0],Incident_Roads[2][0][0],Incident_Roads[2][1][0],Incident_Roads[0][2], angle_fix(Helper_paths[1][2])))
    Connection_Roads.append(Clothoid_Curve(Incident_Roads[1][0][0],Incident_Roads[1][1][0],Incident_Roads[2][0][0],Incident_Roads[2][1][0],Helper_paths[0][2] - np.pi ,Helper_paths[1][2] - np.pi))     
    Connection_Roads.append(Clothoid_Curve(Incident_Roads[2][0][0],Incident_Roads[2][1][0],Incident_Roads[3][0][0],Incident_Roads[3][1][0],Helper_paths[1][2] - np.pi ,Helper_paths[2][2] - np.pi))     


    return Connection_Roads 



Incident_Roads, Helper_paths = Incident_Map_Gen(3) ### 4 + 1 at the momment

Incident_Map_Plot(Incident_Roads, Helper_paths)

Connection_Roads = Connection_road_gen(Incident_Roads)

Connection_road_plotter(Connection_Roads)



plt.plot(Incident_Roads[0][0][0],Incident_Roads[0][1][0], 'ro')
plt.plot(Incident_Roads[1][0][0],Incident_Roads[1][1][0], 'bo')

print(Helper_paths[0][2],Helper_paths[1][2],)
plt.show()

