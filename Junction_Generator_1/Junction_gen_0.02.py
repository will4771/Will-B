### Step 1 manual input SQUARE!!
### Step 2 makes random Incident Roads for a square ### new part is random incident roads
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


def Incident_gen(boundry):
    edge = random.randint(0,3)

    if edge == 0:
        x1 = boundry[0][0]
        y1 = random.uniform(1,11)
        x2 = x1 - 1
        y2 = y1
    
    if edge == 1:
        x1 = boundry[0][1]
        y1 = random.uniform(1,11)
        x2= x1 + 1 
        y2 = y1
    

    if edge == 2:
        x1 = random.uniform(1,11)
        y1 = boundry[1][1]
        x2 = x1
        y2 = y1 + 1
    

    if edge == 3:
        x1 = random.uniform(1,11)
        y1 = boundry[1][0]
        x2 = x1
        y2 = y1 - 1
    
    print(x1,y1,x2,y2)
    return [x1,y1],[x2,y2]
def plot_line_2coords_incident(P1,P2):
    x = np.linspace(P1[0],P2[0],100)
    y = np.linspace(P1[1],P2[1],100)
    
    plt.plot(x,y,color='blue')
        

Incident_Roads= []

for i in range(4):
    a = Incident_gen(boundry)
    b = a[0]
    c = a[1]
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