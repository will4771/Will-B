### Step 1 manual input SQUARE!!
### Step 2 makes random Incident Roads for a square ### new part creating min / min distance between new incident roads now 2.5 +
#  changed the random int to generate from 2,10 so not to be so close to corners
### Step 3 only creates straight lines between the incident roads



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

## Step 1 setting our boundry for the junction 
## Boundries (square) x : {1, 11}, y : {1,11}


fig, ax = plt.subplots()
square = patches.Rectangle((1, 1), 10, 10, edgecolor='red', facecolor='none')
ax.add_patch(square)
plt.xlim(0, 12)
plt.ylim(0, 12)
plt.gca().set_aspect('equal', adjustable='box')  # Ensures equal aspect ratio


## Incidident Roads :  BASIC N = 3 CASE 
boundry = [[1,11],[1,11]]




def Incident_gen(boundry,N,E,S,W): ## IR incident Road lists
    
  
    
    Z = True

    edge = random.randint(0,3)

    if edge == 0:
        while Z == True:
            x1 = boundry[0][0]
            y1 = random.uniform(2,10)
            x2 = x1 - 1
            y2 = y1
            Direction = 0 ## WEST
            for i in range(len(W)):
                if np.abs(y1 - W[i]) < 2.5:
                    Z = True
                else:
                    Z = False

        W.append(y1)            
        return [x1,y1,Direction],[x2,y2,Direction],N,E,S,W
                    
    if edge == 1:
        while Z == True:
            x1 = boundry[0][1]
            y1 = random.uniform(2,10)
            x2= x1 + 1 
            y2 = y1
            Direction = 1 ##North
            for i in range(len(N)):
                if np.abs(y1 - N[i]) < 2.5:
                    Z = True
                else:
                    Z = False
                    
        N.append(y1)
        return [x1,y1,Direction],[x2,y2,Direction],N,E,S,W

    if edge == 2:
        while Z == True:
            x1 = random.uniform(2,10)
            y1 = boundry[1][1]
            x2 = x1
            y2 = y1 + 1
            Direction = 2 ## EAST
            for i in range(len(E)):
                if np.abs(x1 - E[i]) < 2.5:
                    Z = True
                else:
                    Z = False
        E.append(x1)            
        return [x1,y1,Direction],[x2,y2,Direction],N,E,S,W
    if edge == 3:
        while Z == True:
            x1 = random.uniform(2,10)
            y1 = boundry[1][0]
            x2 = x1
            y2 = y1 - 1
            Direction = 3 ## South
            for i in range(len(S)):
                if np.abs(x1 - S[i]) < 2.5:
                    Z = True
                else:
                    Z = False
        S.append(x1)            
        return [x1,y1,Direction],[x2,y2,Direction],N,E,S,W        
    
    

def plot_line_2coords_incident(P1,P2):
    x = np.linspace(P1[0],P2[0],100)
    y = np.linspace(P1[1],P2[1],100)
    
    plt.plot(x,y,color='blue')
        
global Incident_Roads
Incident_Roads= []
N = [0]
E = [0]
S = [0]
W = [0]

for i in range(7):
    a = Incident_gen(boundry,N,E,S,W)
    b = a[0]
    c = a[1]
    N = a[2]
    E = a[3]
    S = a[4]
    W = a[5]
    
    plot_line_2coords_incident(b,c)
    Incident_Roads.append(b)

print(Incident_Roads)

    



## Step 3 Creating the Helper Roads.



def plot_line_2coords(P1,P2):
    x = np.linspace(P1[0],P2[0],100)
    y = np.linspace(P1[1],P2[1],100)
    
    plt.plot(x,y,color='green')


def Helper_Road_Gen(list):
    for i in range(len(list)):
        for j in range(len(list)):
            plot_line_2coords(list[i],list[j])

Helper_Road_Gen(Incident_Roads)

plt.show()