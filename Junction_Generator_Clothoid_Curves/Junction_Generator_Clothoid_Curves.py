import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, quad
from scipy.optimize import minimize
from typing import List, Tuple
import random
from matplotlib.patches import FancyBboxPatch

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
    # Initialize variables
    Incident_Road_Temp = []
    Number_of_Incident_Roads = 0
    x0 = np.random.uniform(25, 50)  # Random starting x-coordinate
    y0 = np.random.uniform(25, 50)  # Random starting y-coordinate
    t = np.linspace(0, 2, 21)  # Time vector

    Angle_out = np.random.uniform(-np.pi, np.pi)  # Random angle
    Angle_in = normalize_angle(Angle_out + np.pi)  # Adjusted angle

    Theta_L = Angle_out + np.pi / 2  # Left angle
    Theta_R = Angle_out - np.pi / 2  # Right angle

    delta_L_x = np.cos(Theta_L)  # x-direction cosine for left
    delta_L_y = np.sin(Theta_L)  # y-direction sine for left

    delta_R_x = np.cos(Theta_R)  # x-direction cosine for right
    delta_R_y = np.sin(Theta_R)  # y-direction sine for right

    delta_x = np.cos(Angle_in)  # x-direction cosine for incident road
    delta_y = np.sin(Angle_in)  # y-direction sine for incident road
    
    x_coords = list(x0 + t * delta_x)  # x-coordinates of the road
    y_coords = list(y0 + t * delta_y)  # y-coordinates of the road

    Incident_Road_Temp.append([x_coords, y_coords])  # Append primary road coordinates

    # Determine number of incident roads (1, 2, or 3)
    Number_of_Incident_Roads = random.randint(Min_num_of_Incident_Roads - 1, Max_num_Incident_Roads - 1)

    # Partition width
    Partition_of_Inicident_Roads = np.random.uniform(Partition_of_Inicident_Roads_min, Partition_of_Inicident_Roads_max)

    # Randomly choose whether to use partition
    k = random.randint(0, 1)
    
    if k == 1:
        A = np.random.uniform(lane_width_min, lane_width_max) + Partition_of_Inicident_Roads
        l = True
    else:
        A = np.random.uniform(lane_width_min, lane_width_max)
        l = False

    # Append additional incident roads based on the number
    if Number_of_Incident_Roads == 1:
        Incident_Road_Temp.append([list((x0 + A * delta_L_x) + t * delta_x), list((y0 + A * delta_L_y) + t * delta_y)])
    elif Number_of_Incident_Roads == 2:
        Incident_Road_Temp.append([list((x0 + A * delta_L_x) + t * delta_x), list((y0 + A * delta_L_y) + t * delta_y)])
        Incident_Road_Temp.append([list((x0 + A * delta_R_x) + t * delta_x), list((y0 + A * delta_R_y) + t * delta_y)])
    else:
        Incident_Road_Temp.append([list((x0) + t * delta_x), list((y0) + t * delta_y)])

    Incident_Road_Temp.append(l)  # Add partition flag
    
    if k == 1:
        Incident_Road_Temp.append(A - Partition_of_Inicident_Roads)  # Lane width with partition
    else:
        Incident_Road_Temp.append(A)  # Lane width without partition

    Incident_Road_Temp.append(Angle_in)  # Incoming angle
    Incident_Road_Temp.append(Angle_out)  # Outgoing angle
    Incident_Road_Temp.append(Number_of_Incident_Roads + 1)  # Total number of incident roads
    
    return Incident_Road_Temp


def Incident_Plotter_Green(Incident_roads):  # Takes a list of incident roads
    Count = Incident_roads[-1]  # Extract the number of incident roads from the last element
    for i in range(Count):
        # Plot each incident road with green color
        plt.plot(Incident_roads[i][0], Incident_roads[i][1], color='green')

def clothoid_ode_rhs(state, s, kappa0, kappa1):
    x, y, theta = state[0], state[1], state[2]
    # Return derivatives for clothoid curve
    return [np.cos(theta), np.sin(theta), kappa0 + kappa1 * s]

def eval_clothoid(x0, y0, theta0, kappa0, kappa1, s):
    # Solve the ODE for the clothoid curve
    return odeint(clothoid_ode_rhs, [x0, y0, theta0], s, args=(kappa0, kappa1))

def Helper_Path_Gen(x0: float, y0: float, theta0: float, L: float) -> Tuple[List[float], List[float], float]:
    kappa0, kappa1 = Curvature_of_Incident_Road_placment, Curvature_of_Incident_Road_placment
    s = np.linspace(0, L, 1000)  # Generate s values
    
    sol = eval_clothoid(x0, y0, theta0, kappa0, kappa1, s)  # Compute clothoid path

    xs, ys, thetas = sol[:, 0], sol[:, 1], sol[:, 2]  # Extract coordinates and angles
    return xs, ys, thetas[-1]  # Return path and final angle

def Incident_Road_Gen(Number_of_incident_Roads: int) -> List:
    Roads = []  # List to store road details
    Helper_paths = []  # List to store helper paths
    Roads.append(Initial_incident_Road())  # Add initial incident road
    Number_of_Incident_Roads = 0

    i = 1  # Start from 1 to avoid index error
    while i < Number_of_incident_Roads:
        temp = []  # Temporary storage for a new incident road
        t = np.linspace(0, 2, 21)  # Time vector
        
        # Generate helper path based on the last road's end point and angle
        Helper_paths.append(Helper_Path_Gen(Roads[i-1][1][0][0], Roads[i-1][1][1][0], Roads[i-1][-2], Length_of_Roads))
        
        x0 = Helper_paths[i-1][0][-1]  # x-coordinate of the new road start
        y0 = Helper_paths[i-1][1][-1]  # y-coordinate of the new road start
        Angle_in = normalize_angle(Helper_paths[i-1][2])  # Incoming angle
        Angle_out = normalize_angle(Helper_paths[i-1][2] - np.pi)  # Outgoing angle

        Theta_L = Angle_out + np.pi / 2  # Left angle
        Theta_R = Angle_out - np.pi / 2  # Right angle

        delta_L_x = np.cos(Theta_L)  # x-component for left road
        delta_L_y = np.sin(Theta_L)  # y-component for left road

        delta_R_x = np.cos(Theta_R)  # x-component for right road
        delta_R_y = np.sin(Theta_R)  # y-component for right road

        delta_x = np.cos(Angle_in)  # x-component for the incident road
        delta_y = np.sin(Angle_in)  # y-component for the incident road
        
        x_coords = list(x0 + t * delta_x)  # x-coordinates of the new road
        y_coords = list(y0 + t * delta_y)  # y-coordinates of the new road

        temp.append([x_coords, y_coords])  # Append new road coordinates

        # Determine partition and lane width
        Partition_of_Inicident_Roads = np.random.uniform(Partition_of_Inicident_Roads_min, Partition_of_Inicident_Roads_max)
        k = random.randint(0, 1)
        if k == 1:
            A = np.random.uniform(lane_width_min, lane_width_max) + Partition_of_Inicident_Roads
            l = True
        else:
            A = np.random.uniform(lane_width_min, lane_width_max)
            l = False

        # Generate additional incident roads if needed
        Number_of_Incident_Roads = random.randint(Min_num_of_Incident_Roads - 1, Max_num_Incident_Roads - 1)
        if Number_of_Incident_Roads == 1:
            temp.append([list((x0 + A * delta_L_x) + t * delta_x), list((y0 + A * delta_L_y) + t * delta_y)])
        elif Number_of_Incident_Roads == 2:
            temp.append([list((x0 + A * delta_L_x) + t * delta_x), list((y0 + A * delta_L_y) + t * delta_y)])
            temp.append([list((x0 + A * delta_R_x) + t * delta_x), list((y0 + A * delta_R_y) + t * delta_y)])
        else:
            temp.append([list((x0) + t * delta_x), list((y0) + t * delta_y)])

        temp.append(l)  # Add partition flag
        temp.append(A - Partition_of_Inicident_Roads if k == 1 else A)  # Add lane width
        temp.append(Angle_in)  # Add incoming angle
        temp.append(Angle_out)  # Add outgoing angle
        temp.append(Number_of_Incident_Roads + 1)  # Add total number of incident roads

        Roads.append(temp)  # Append new road to the list

        # Check for proximity with existing roads and remove if too close
        should_pop = False
        for j in range(len(Roads) - 1):
            if (np.abs(Roads[-1][1][0][0] - Roads[j][1][0][0]) <= Min_seperation_of_Incident_roads and
                np.abs(Roads[-1][1][1][0] - Roads[j][1][1][0]) <= Min_seperation_of_Incident_roads):
                Roads.pop()
                Helper_paths.pop()
                should_pop = True
                break
                
        if not should_pop:
            i += 1  # Increment index if road is not removed
    
    return Roads  # Return the list of roads

def Clothoid_Curve(a, b, c, d, heading0, heading1):
    x0, y0, x1, y1 = a, b, c, d  # Start and end coordinates
    v0, v1 = heading0, heading1  # Start and end headings
    Delta_x, Delta_y = x1 - x0, y1 - y0  # Difference in coordinates
    Delta_v = normalize_angle(v1 - v0)  # Change in heading
    phi = np.arctan2(Delta_y, Delta_x)  # Angle of the line connecting start and end
    r = np.sqrt(Delta_x**2 + Delta_y**2)  # Distance between start and end
    Delta_Phi = normalize_angle(v0 - phi)  # Difference between initial heading and line angle

    # Function to compute the integral for G(A)
    def G(A):
        def integrand(tau):
            return np.sin(A * tau**2 + (Delta_v - A) * tau + Delta_Phi)
        integral = quad(integrand, 0, 1)[0]
        return integral

    # Function to compute the derivative of the integral for G(A)
    def dG(A):
        def integrand(tau):
            return np.cos(A * tau**2 + (Delta_v - A) * tau + Delta_Phi) * (tau**2 - tau)
        integral = quad(integrand, 0, 1)[0]
        return integral

    # Newton's method to find the root of G(A) derivative
    def newton(f, df, x0):
        iterates = [x0]
        for i in range(10):
            iterates.append(iterates[-1] - f(iterates[-1]) / df(iterates[-1]))
        return iterates

    A = newton(G, dG, 1)[-1]  # Find optimal A

    # Function to compute the integral for H(A)
    def H(A):
        def integrand(tau):
            return np.sin(A * tau**2 + (Delta_v - A) * tau + Delta_Phi + (np.pi/2))
        integral = quad(integrand, 0, 1)[0]
        return integral

    L = r / H(A)  # Length of the clothoid
    k0 = (Delta_v - A) / L  # Initial curvature
    k1 = (2 * A) / L**2  # Curvature rate

    # Generate clothoid curve
    def clothoid_curve(x0, y0, theta0, L, kappa0, kappa1, num_points=1000):
        s_vals = np.linspace(0, L, num_points)  # Parameter values
        x_vals = np.zeros(num_points)  # x coordinates
        y_vals = np.zeros(num_points)  # y coordinates
        for i, s in enumerate(s_vals):
            def integrand_x(tau):
                return np.cos(0.5 * kappa1 * tau**2 + kappa0 * tau + theta0)
            def integrand_y(tau):
                return np.sin(0.5 * kappa1 * tau**2 + kappa0 * tau + theta0)
            x_vals[i] = x0 + quad(integrand_x, 0, s)[0]
            y_vals[i] = y0 + quad(integrand_y, 0, s)[0]
        return [x_vals, y_vals]

    return clothoid_curve(x0, y0, v0, L, k0, k1)  # Return the clothoid curve

def Connection_road_gen(Incident_roads, Incident_road_index):
    """
    Generates connection roads between a specified incident road and other incident roads.

    Parameters:
    Incident_roads (list): List of roads where each road contains data about its segments and characteristics.
    Incident_road_index (int): Index of the road from which to generate paths.

    Returns:
    list: List of generated connection roads.
    """
    
    Roads, I = Incident_roads, Incident_road_index

    Connection_Roads = []  # List to store the generated connection roads

    for j in range(len(Incident_roads)):
        if I == j:  # Skip the road if it's the same as the incident road
            break
        
        # Case where both roads have type 3
        if Roads[I][-1] == 3 and Roads[j][-1] == 3:
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][0][0], Roads[I][1][1][0], Roads[j][2][0][0], Roads[j][2][1][0], Roads[I][-2], Roads[j][-3]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][0][0], Roads[I][0][1][0], Roads[j][0][0][0], Roads[j][0][1][0], Roads[I][-2], Roads[j][-3]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][2][0][0], Roads[I][2][1][0], Roads[j][1][0][0], Roads[j][1][1][0], Roads[I][-2], Roads[j][-3]))

        # Case where incident road is type 3 and other road is type 2
        if Roads[I][-1] == 3 and Roads[j][-1] == 2:
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][0][0], Roads[I][1][1][0], Roads[j][0][0][0], Roads[j][0][1][0], Roads[I][-2], Roads[j][-3]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][0][0], Roads[I][0][1][0], Roads[j][0][0][0], Roads[j][0][1][0], Roads[I][-2], Roads[j][-3]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][2][0][0], Roads[I][2][1][0], Roads[j][1][0][0], Roads[j][1][1][0], Roads[I][-2], Roads[j][-3]))

        # Case where incident road is type 3 and other road is type 1
        if Roads[I][-1] == 3 and Roads[j][-1] == 1:
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][0][0], Roads[I][1][1][0], Roads[j][0][0][0], Roads[j][0][1][0], Roads[I][-2], Roads[j][-3]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][0][0], Roads[I][0][1][0], Roads[j][0][0][0], Roads[j][0][1][0], Roads[I][-2], Roads[j][-3]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][2][0][0], Roads[I][2][1][0], Roads[j][0][0][0], Roads[j][0][1][0], Roads[I][-2], Roads[j][-3]))

        # Case where incident road is type 2 and other road is type 3
        if Roads[I][-1] == 2 and Roads[j][-1] == 3:
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][0][0], Roads[I][1][1][0], Roads[j][2][0][0], Roads[j][2][1][0], Roads[I][-2], Roads[j][-3]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][0][0], Roads[I][0][1][0], Roads[j][0][0][0], Roads[j][0][1][0], Roads[I][-2], Roads[j][-3]))

        # Case where both roads have type 2
        if Roads[I][-1] == 2 and Roads[j][-1] == 2:
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][0][0], Roads[I][1][1][0], Roads[j][0][0][0], Roads[j][0][1][0], Roads[I][-2], Roads[j][-3]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][0][0], Roads[I][0][1][0], Roads[j][1][0][0], Roads[j][1][1][0], Roads[I][-2], Roads[j][-3]))   

        # Case where incident road is type 2 and other road is type 1
        if Roads[I][-1] == 2 and Roads[j][-1] == 1:
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][0][0], Roads[I][1][1][0], Roads[j][0][0][0], Roads[j][0][1][0], Roads[I][-2], Roads[j][-3]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][0][0], Roads[I][0][1][0], Roads[j][0][0][0], Roads[j][0][1][0], Roads[I][-2], Roads[j][-3]))
    
        # Case where both roads have type 1
        if Roads[I][-1] == 1 and Roads[j][-1] == 1:
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][0][0], Roads[I][0][1][0], Roads[j][0][0][0], Roads[j][0][1][0], Roads[I][-2], Roads[j][-3]))

        # Case where incident road is type 1 and other road is type 3
        if Roads[I][-1] == 1 and Roads[j][-1] == 3:
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][0][0], Roads[I][0][1][0], Roads[j][2][0][0], Roads[j][2][1][0], Roads[I][-2], Roads[j][-3]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][0][0], Roads[I][0][1][0], Roads[j][0][0][0], Roads[j][0][1][0], Roads[I][-2], Roads[j][-3]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][0][0], Roads[I][0][1][0], Roads[j][1][0][0], Roads[j][1][1][0], Roads[I][-2], Roads[j][-3]))

        # Case where incident road is type 1 and other road is type 2
        if Roads[I][-1] == 1 and Roads[j][-1] == 2:
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][0][0], Roads[I][0][1][0], Roads[j][0][0][0], Roads[j][0][1][0], Roads[I][-2], Roads[j][-3]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][0][0], Roads[I][0][1][0], Roads[j][1][0][0], Roads[j][1][1][0], Roads[I][-2], Roads[j][-3]))   

    return Connection_Roads  # Return the list of generated connection roads

def Connection_road_plotter(Connection_Roads):
    for road in Connection_Roads:
        plt.plot(road[0], road[1], color='white')

def Incident_Boundry_Road_gen(a, b, c, d, e):
    """
    Generate the coordinates for the boundaries of a road segment based on the incident and outgoing angles.

    Parameters:
    a (float): x-coordinate of the starting point.
    b (float): y-coordinate of the starting point.
    c (float): Angle of incidence (in radians).
    d (float): Angle of departure (in radians).
    e (float): Lane width.

    Returns:
    tuple: Two lists of coordinates for the left and right boundaries of the road segment,
           and the angles of incidence and departure.
           ([x_coords_L, y_coords_L], [x_coords_R, y_coords_R], Angle_in, Angle_out)
    """

    # Starting coordinates
    x0 = a
    y0 = b
    
    # Angles of incidence and departure
    Angle_in = c
    Angle_out = d

    # Compute the angles for the left and right boundaries based on the outgoing angle
    Theta_L = Angle_out + np.pi / 2
    Theta_R = Angle_out - np.pi / 2

    # Calculate the deltas for the left boundary
    delta_L_x = np.cos(Theta_L)
    delta_L_y = np.sin(Theta_L)

    # Calculate the deltas for the right boundary
    delta_R_x = np.cos(Theta_R)
    delta_R_y = np.sin(Theta_R)

    # Calculate the deltas for the incident angle
    delta_x = np.cos(Angle_in)
    delta_y = np.sin(Angle_in)

    # Generate 21 evenly spaced points from 0 to 2
    t = np.linspace(0, 2, 21)

    # Define the lane width
    lane_width = e

    # Calculate the coordinates for the left boundary of the road
    x_coords_L = list((x0 + (lane_width / 2) * delta_L_x) + t * delta_x)
    y_coords_L = list((y0 + (lane_width / 2) * delta_L_y) + t * delta_y)

    # Calculate the coordinates for the right boundary of the road
    x_coords_R = list((x0 + (lane_width / 2) * delta_R_x) + t * delta_x)
    y_coords_R = list((y0 + (lane_width / 2) * delta_R_y) + t * delta_y)

    # Return the coordinates for the left and right boundaries along with the incident and outgoing angles
    return [x_coords_L, y_coords_L], [x_coords_R, y_coords_R], Angle_in, Angle_out


def Incident_Plotter_boun(x_points: List[float], y_points: List[float]) -> None:
    plt.plot(x_points, y_points, color='red')

def Connection_Road_Boundry_gen(Inc_1, Inc_2): ### BEST CODE TO UNDERSTAND THE CLOTHOID CURVE FUNCTION
    """
    Generate the boundary coordinates for a connecting road segment between two incident road segments.

    Parameters:
    Inc_1 (list): List containing the boundary coordinates and angles of the first incident road segment.
    Inc_2 (list): List containing the boundary coordinates and angles of the second incident road segment.

    Returns:
    tuple: Two lists of coordinates for the right and left boundaries of the connecting road segment.
           ([xR, yR], [xL, yL])
    """

    # Generate the left boundary coordinates for the connecting road segment
    xL, yL = Clothoid_Curve(
        Inc_1[0][0][0],  # x-coordinate of the start point of the left boundary of the first incident road
        Inc_1[0][1][0],  # y-coordinate of the start point of the left boundary of the first incident road
        Inc_2[1][0][0],  # x-coordinate of the end point of the right boundary of the second incident road
        Inc_2[1][1][0],  # y-coordinate of the end point of the right boundary of the second incident road
        Inc_1[-1],       # Angle of incidence of the first incident road
        Inc_2[-2]        # Angle of departure of the second incident road
    )

    # Generate the right boundary coordinates for the connecting road segment
    xR, yR = Clothoid_Curve(
        Inc_1[1][0][0],  # x-coordinate of the start point of the right boundary of the first incident road
        Inc_1[1][1][0],  # y-coordinate of the start point of the right boundary of the first incident road
        Inc_2[0][0][0],  # x-coordinate of the end point of the left boundary of the second incident road
        Inc_2[0][1][0],  # y-coordinate of the end point of the left boundary of the second incident road
        Inc_1[-1],       # Angle of incidence of the first incident road
        Inc_2[-2]        # Angle of departure of the second incident road
    )

    # Return the coordinates for the right and left boundaries of the connecting road segment
    return [xR, yR], [xL, yL]
    
def Connection_boundry_plotter(Connection_Boundry, i ): ### i is which element in connection boundry list
    plt.plot(Connection_Boundry[i][0][0],Connection_Boundry[i][0][1],color = 'green')
    plt.plot(Connection_Boundry[i][1][0],Connection_Boundry[i][1][1],color = 'green')

def Boundry_plotter(Connection_Roads):
    for road in Connection_Roads:
        plt.plot(road[0], road[1], color='green')  

def Connection_Boundry_gen(Incident_boundry, Incident_road_index, Incident_Roads):
    """
    Generate the connection roads between different incident road segments based on their types and indices.

    Parameters:
    Incident_boundry (list): A list of road boundaries.
    Incident_road_index (int): Index of the current incident road segment.
    Incident_Roads (list): A list containing details of all incident roads, including their type.

    Returns:
    list: A list of generated connection roads.
    """

    Roads, I = Incident_boundry, Incident_road_index
    Connection_Roads = []

    for j in range(len(Incident_Roads)):
        if I == j:
            continue  # Skip the same road index

        # Case when both roads are of type 3
        if Incident_Roads[I][-1] == 3 and Incident_Roads[j][-1] == 3:
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][0][0][0], Roads[I][1][0][1][0], Roads[j][2][1][0][0], Roads[j][2][1][1][0], Roads[I][0][-1], Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][1][0][0], Roads[I][1][1][1][0], Roads[j][2][0][0][0], Roads[j][2][0][1][0], Roads[I][0][-1], Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][0][0][0], Roads[I][0][0][1][0], Roads[j][0][1][0][0], Roads[j][0][1][1][0], Roads[I][0][-1], Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][1][0][0], Roads[I][0][1][1][0], Roads[j][0][0][0][0], Roads[j][0][0][1][0], Roads[I][0][-1], Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][2][0][0][0], Roads[I][2][0][1][0], Roads[j][1][1][0][0], Roads[j][1][1][1][0], Roads[I][0][-1], Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][2][1][0][0], Roads[I][2][1][1][0], Roads[j][1][0][0][0], Roads[j][1][0][1][0], Roads[I][0][-1], Roads[j][0][-2]))

        # Case when the current road is of type 3 and the other road is of type 2
        elif Incident_Roads[I][-1] == 3 and Incident_Roads[j][-1] == 2:
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][0][0][0], Roads[I][1][0][1][0], Roads[j][0][1][0][0], Roads[j][0][1][1][0], Roads[I][0][-1], Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][1][0][0], Roads[I][1][1][1][0], Roads[j][0][0][0][0], Roads[j][0][0][1][0], Roads[I][0][-1], Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][0][0][0], Roads[I][0][0][1][0], Roads[j][0][1][0][0], Roads[j][0][1][1][0], Roads[I][0][-1], Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][1][0][0], Roads[I][0][1][1][0], Roads[j][0][0][0][0], Roads[j][0][0][1][0], Roads[I][0][-1], Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][2][0][0][0], Roads[I][2][0][1][0], Roads[j][1][1][0][0], Roads[j][1][1][1][0], Roads[I][0][-1], Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][2][1][0][0], Roads[I][2][1][1][0], Roads[j][1][0][0][0], Roads[j][1][0][1][0], Roads[I][0][-1], Roads[j][0][-2]))

        # Case when the current road is of type 3 and the other road is of type 1
        elif Incident_Roads[I][-1] == 3 and Incident_Roads[j][-1] == 1:
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][0][0][0], Roads[I][1][0][1][0], Roads[j][0][1][0][0], Roads[j][0][1][1][0], Roads[I][0][-1], Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][1][0][0], Roads[I][1][1][1][0], Roads[j][0][0][0][0], Roads[j][0][0][1][0], Roads[I][0][-1], Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][0][0][0], Roads[I][0][0][1][0], Roads[j][0][1][0][0], Roads[j][0][1][1][0], Roads[I][0][-1], Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][1][0][0], Roads[I][0][1][1][0], Roads[j][0][0][0][0], Roads[j][0][0][1][0], Roads[I][0][-1], Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][2][0][0][0], Roads[I][2][0][1][0], Roads[j][0][1][0][0], Roads[j][0][1][1][0], Roads[I][0][-1], Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][2][1][0][0], Roads[I][2][1][1][0], Roads[j][0][0][0][0], Roads[j][0][0][1][0], Roads[I][0][-1], Roads[j][0][-2]))

        # Case when the current road is of type 2 and the other road is of type 3
        elif Incident_Roads[I][-1] == 2 and Incident_Roads[j][-1] == 3:
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][0][0][0], Roads[I][1][0][1][0], Roads[j][2][1][0][0], Roads[j][2][1][1][0], Roads[I][0][-1], Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][1][0][0], Roads[I][1][1][1][0], Roads[j][2][0][0][0], Roads[j][2][0][1][0], Roads[I][0][-1], Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][0][0][0], Roads[I][0][0][1][0], Roads[j][0][1][0][0], Roads[j][0][1][1][0], Roads[I][0][-1], Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][1][0][0], Roads[I][0][1][1][0], Roads[j][0][0][0][0], Roads[j][0][0][1][0], Roads[I][0][-1], Roads[j][0][-2]))

        # Case when the current road is of type 2 and the other road is of type 1
        elif Incident_Roads[I][-1] == 2 and Incident_Roads[j][-1] == 1:
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][0][0][0], Roads[I][1][0][1][0], Roads[j][0][1][0][0], Roads[j][0][1][1][0], Roads[I][0][-1], Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][1][0][0], Roads[I][1][1][1][0], Roads[j][0][0][0][0], Roads[j][0][0][1][0], Roads[I][0][-1], Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][0][0][0], Roads[I][0][0][1][0], Roads[j][0][1][0][0], Roads[j][0][1][1][0], Roads[I][0][-1], Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][1][0][0], Roads[I][0][1][1][0], Roads[j][0][0][0][0], Roads[j][0][0][1][0], Roads[I][0][-1], Roads[j][0][-2]))

        # Case when both roads are of type 2
        elif Incident_Roads[I][-1] == 2 and Incident_Roads[j][-1] == 2:
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][0][0][0], Roads[I][1][0][1][0], Roads[j][0][1][0][0], Roads[j][0][1][1][0], Roads[I][0][-1], Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][1][1][0][0], Roads[I][1][1][1][0], Roads[j][0][0][0][0], Roads[j][0][0][1][0], Roads[I][0][-1], Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][0][0][0], Roads[I][0][0][1][0], Roads[j][1][1][0][0], Roads[j][1][1][1][0], Roads[I][0][-1], Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][1][0][0], Roads[I][0][1][1][0], Roads[j][1][0][0][0], Roads[j][1][0][1][0], Roads[I][0][-1], Roads[j][0][-2]))

        # Case when both roads are of type 1
        elif Incident_Roads[I][-1] == 1 and Incident_Roads[j][-1] == 1:
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][0][0][0], Roads[I][0][0][1][0], Roads[j][0][1][0][0], Roads[j][0][1][1][0], Roads[I][0][-1], Roads[j][0][-2]))
            Connection_Roads.append(Clothoid_Curve(Roads[I][0][1][0][0], Roads[I][0][1][1][0], Roads[j][0][0][0][0], Roads[j][0][0][1][0], Roads[I][0][-1], Roads[j][0][-2]))

    return Connection_Roads


def Boundry_gen(Incident_boundry_Roads,Incident_Roads):
    Boundry = []
    Roads = Incident_boundry_Roads

    for k in range(len(Incident_Roads) - 1):
        a = Incident_Roads[k][-1] ### either 1,2,3
        b = Incident_Roads[k+1][-1]

        if a == 3 and b == 3:
            Boundry.append(Clothoid_Curve(Roads[k][1][0][0][0], Roads[k][1][0][1][0], Roads[k+1][2][1][0][0], Roads[k+1][2][1][1][0], Roads[k][0][-1], Roads[k+1][0][-2]))

        if a == 3 and b == 2: 
            Boundry.append(Clothoid_Curve(Roads[k][1][0][0][0], Roads[k][1][0][1][0], Roads[k+1][0][1][0][0], Roads[k+1][0][1][1][0], Roads[k][0][-1], Roads[k+1][0][-2]))

        if a == 3 and b == 1:
            Boundry.append(Clothoid_Curve(Roads[k][1][0][0][0], Roads[k][1][0][1][0], Roads[k+1][0][1][0][0], Roads[k+1][0][1][1][0], Roads[k][0][-1], Roads[k+1][0][-2]))



        if a == 2 and b == 3:
            Boundry.append(Clothoid_Curve(Roads[k][1][0][0][0], Roads[k][1][0][1][0], Roads[k+1][2][1][0][0], Roads[k+1][2][1][1][0], Roads[k][0][-1], Roads[k+1][0][-2]))

        if a == 2 and b == 2:
            Boundry.append(Clothoid_Curve(Roads[k][1][0][0][0], Roads[k][1][0][1][0], Roads[k+1][0][1][0][0], Roads[k+1][0][1][1][0], Roads[k][0][-1], Roads[k+1][0][-2]))

        if a == 2 and b == 1:
            Boundry.append(Clothoid_Curve(Roads[k][1][0][0][0], Roads[k][1][0][1][0], Roads[k+1][0][1][0][0], Roads[k+1][0][1][1][0], Roads[k][0][-1], Roads[k+1][0][-2]))


        if a == 1 and b == 3:
            Boundry.append(Clothoid_Curve(Roads[k][0][0][0][0], Roads[k][0][0][1][0], Roads[k+1][2][1][0][0], Roads[k+1][2][1][1][0], Roads[k][0][-1], Roads[k+1][0][-2]))
        
        if a == 1 and b == 2:
            Boundry.append(Clothoid_Curve(Roads[k][0][0][0][0], Roads[k][0][0][1][0], Roads[k+1][0][1][0][0], Roads[k+1][0][1][1][0], Roads[k][0][-1], Roads[k+1][0][-2]))

        if a == 1 and b == 1 : 
            Boundry.append(Clothoid_Curve(Roads[k][0][0][0][0], Roads[k][0][0][1][0], Roads[k+1][0][1][0][0], Roads[k+1][0][1][1][0], Roads[k][0][-1], Roads[k+1][0][-2]))


    ### my poor programming skills has lead to this being the best soluiton i had at the time, im sure there is a more elegant solution avaliable 
    i = len(Incident_boundry_Roads) - 1 
    j = 0 

    a = Incident_Roads[i][-1] ### either 1,2,3
    b = Incident_Roads[j][-1]

    if a == 3 and b == 3:
            Boundry.append(Clothoid_Curve(Roads[i][1][0][0][0], Roads[i][1][0][1][0], Roads[j][2][1][0][0], Roads[j][2][1][1][0], Roads[i][0][-1], Roads[j][0][-2]))
    
    if a == 3 and b == 2: 
        Boundry.append(Clothoid_Curve(Roads[i][1][0][0][0], Roads[i][1][0][1][0], Roads[j][0][1][0][0], Roads[j][0][1][1][0], Roads[i][0][-1], Roads[j][0][-2]))

    if a == 3 and b == 1:
        Boundry.append(Clothoid_Curve(Roads[i][1][0][0][0], Roads[i][1][0][1][0], Roads[j][0][1][0][0], Roads[j][0][1][1][0], Roads[i][0][-1], Roads[j][0][-2]))


    if a == 2 and b == 3:
        Boundry.append(Clothoid_Curve(Roads[i][1][0][0][0], Roads[i][1][0][1][0], Roads[j][2][1][0][0], Roads[j][2][1][1][0], Roads[i][0][-1], Roads[j][0][-2]))

    if a == 2 and b == 2:
        Boundry.append(Clothoid_Curve(Roads[i][1][0][0][0], Roads[i][1][0][1][0], Roads[j][0][1][0][0], Roads[j][0][1][1][0], Roads[i][0][-1], Roads[j][0][-2]))

    if a == 2 and b == 1:
        Boundry.append(Clothoid_Curve(Roads[i][1][0][0][0], Roads[i][1][0][1][0], Roads[j][0][1][0][0], Roads[j][0][1][1][0], Roads[i][0][-1], Roads[j][0][-2]))

    
    if a == 1 and b == 3:
        Boundry.append(Clothoid_Curve(Roads[i][0][0][0][0], Roads[i][0][0][1][0], Roads[j][2][1][0][0], Roads[j][2][1][1][0], Roads[i][0][-1], Roads[j][0][-2]))
        
    if a == 1 and b == 2:
        Boundry.append(Clothoid_Curve(Roads[i][0][0][0][0], Roads[i][0][0][1][0], Roads[j][0][1][0][0], Roads[j][0][1][1][0], Roads[i][0][-1], Roads[j][0][-2]))

    if a == 1 and b == 1 : 
        Boundry.append(Clothoid_Curve(Roads[i][0][0][0][0], Roads[i][0][0][1][0], Roads[j][0][1][0][0], Roads[j][0][1][1][0], Roads[i][0][-1], Roads[j][0][-2]))

    
    for w in range(len(Incident_Roads)):
        if Incident_Roads[w][-1] == 3 and Incident_Roads[w][-5] == True:  ##and part ensures we dont connect lanes with no incident 
            Boundry.append(Clothoid_Curve(Roads[w][0][0][0][0], Roads[w][0][0][1][0], Roads[w][1][1][0][0], Roads[w][1][1][1][0], Roads[w][0][-1], Roads[w][0][-2]))
            Boundry.append(Clothoid_Curve(Roads[w][2][0][0][0], Roads[w][2][0][1][0], Roads[w][0][1][0][0], Roads[w][0][1][1][0], Roads[w][0][-1], Roads[w][0][-2]))

        if Incident_Roads[w][-1] == 2 and  Incident_Roads[w][-5] == True :
            Boundry.append(Clothoid_Curve(Roads[w][0][0][0][0], Roads[w][0][0][1][0], Roads[w][1][1][0][0], Roads[w][1][1][1][0], Roads[w][0][-1], Roads[w][0][-2]))

    return Boundry

   

def compute_tangent_angles(x, y):
    # Ensure the inputs are numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Compute the first derivative (dy/dx) using central differences
    dy = np.gradient(y, x)
    
    # Compute the tangent angles in radians
    tangent_angles = np.arctan(dy)
    
    # Bound the angles within [-pi, pi]
    tangent_angles = np.mod(tangent_angles + np.pi, 2 * np.pi) - np.pi
    
    return tangent_angles

def plot_rotated_rectangle(ax, center, width, height, angle, corner_radius):
    # Create a FancyBboxPatch with the specified dimensions and corner radius
    box = FancyBboxPatch(
        (center[0] - width / 2, center[1] - height / 2), width, height,
        boxstyle=f"round,pad=0,rounding_size={corner_radius}",
        linewidth=1, edgecolor='w', facecolor='white'
    )
    
    # Apply rotation by setting the transformation
    transform = plt.matplotlib.transforms.Affine2D().rotate_around(center[0], center[1], angle) + ax.transData
    box.set_transform(transform)
    ax.add_patch(box)

def plot_rotated_rectangle(ax,center, width, height, angle, corner_radius):
    # Create a FancyBboxPatch with the specified dimensions and corner radius
    box = FancyBboxPatch(
        (center[0] - width / 2, center[1] - height / 2), width, height,
        boxstyle=f"round,pad=0,rounding_size={corner_radius}",
        linewidth=1, edgecolor='w', facecolor='white'
    )
    
    # Apply rotation by setting the transformation
    transform = plt.matplotlib.transforms.Affine2D().rotate_around(center[0], center[1], angle) + ax.transData
    box.set_transform(transform)
    ax.add_patch(box)

def vehicle_Generator(Incident_Roads, ax):
    vehicles = []
    Connection_Roads = []
    i = random.randint(0, len(Incident_Roads) - 1)

    number_of_lanes = Incident_Roads[i][-1]

    Connection_Roads = Connection_road_gen(Incident_Roads, i)
    
    for m in range(len(Connection_Roads)):
        Connection_Roads[m].append(compute_tangent_angles(Connection_Roads[m][0], Connection_Roads[m][1]))

    for _ in range(250):  # Use _ for throwaway loop variable
        z = random.randint(0, len(Connection_Roads) - 1)  # Select a lane index
        k = random.randint(0, len(Connection_Roads[z][0]) - 1)  # Select a point index

        vehicles.append([Connection_Roads[z][0][k], Connection_Roads[z][1][k], Connection_Roads[z][2][k]])

        should_pop = False
        
        for j in range(len(vehicles) - 1):
            if (np.abs(vehicles[-1][0] - vehicles[j][0]) <= 3.5 and np.abs(vehicles[-1][1] - vehicles[j][1]) <= 4.5):
                vehicles.pop()
                should_pop = True
                break

    for i in range(len(vehicles)):
        plot_rotated_rectangle(ax, (vehicles[i][0], vehicles[i][1]), 2, 3, vehicles[i][2] - np.pi/2, 0.3)  # Plotting the vehicles






def main():    

    Incident_Roads = Incident_Road_Gen(Num_Incident_Roads)
    Connection_Roads = []

    for i in range(len(Incident_Roads)):
        Connection_Roads = Connection_Roads + (Connection_road_gen(Incident_Roads,i))

    if Show_Connection_path == True:
        Connection_road_plotter(Connection_Roads)

    ### adding tangent vectors of the connection curves

    for i in range(len(Connection_Roads)):
        Connection_Roads[i].append(compute_tangent_angles(Connection_Roads[i][0],Connection_Roads[i][1]))

    
    if Add_veh:
        vehicle_Generator(Incident_Roads,ax)
    
    Incident_Roads_boundry = []

    for i in range(len(Incident_Roads)):
        temp = []
        for j in range(Incident_Roads[i][-1]):
            temp.append(Incident_Boundry_Road_gen(Incident_Roads[i][j][0][0],Incident_Roads[i][j][1][0],Incident_Roads[i][-3],Incident_Roads[i][-2],Incident_Roads[i][-4]))
        Incident_Roads_boundry.append(temp)

    for j in range(len(Incident_Roads)):
        for i in range(len(Incident_Roads_boundry[j])):
            Incident_Plotter_boun(Incident_Roads_boundry[j][i][0][0],Incident_Roads_boundry[j][i][0][1])
            Incident_Plotter_boun(Incident_Roads_boundry[j][i][1][0],Incident_Roads_boundry[j][i][1][1])

    Connection_Boundry = []
    Junction_Boundry = []

    for i in range(len(Incident_Roads)):
        Connection_Boundry = Connection_Boundry +  Connection_Boundry_gen(Incident_Roads_boundry,i,Incident_Roads)  

    
    Junction_Boundry =  Junction_Boundry  +  Boundry_gen(Incident_Roads_boundry,Incident_Roads)  

     

    if Show_Lane_Boundries == True:
        Connection_road_plotter(Connection_Boundry)

    if Show_Junction_Boundry == True : 
        Boundry_plotter(Junction_Boundry)

    


    plt.show()

    return Incident_Roads,Connection_Roads,Incident_Roads_boundry,Connection_Boundry

### Para 

Num_Incident_Roads = 3

Curvature_of_Incident_Road_placment = 0.020   ### recommend 0.020

Min_num_of_Incident_Roads,Max_num_Incident_Roads  = 1, 2                ### max 3 min 2... 3, 3 would gen only 3 lane junctions

Length_of_Roads = 12                     ### 8 is min, max 12  NB // if one wants increased length, vary the Curvature of the Incident Road Placment accordingly
                       
lane_width_min, lane_width_max = 2.5, 2.8

Partition_of_Inicident_Roads_min, Partition_of_Inicident_Roads_max = 0.95, 1     ### can set both to zero for no randomness

Min_seperation_of_Incident_roads = 2          ### if = 2 would be a 2 by 2 area of exclusion 

Show_Lane_Boundries = True

Show_Junction_Boundry = False

Show_Connection_path = False

Add_veh = True  ### add cars  (does have bug in it, works half the time...)
main()