### Step 1,2 manual input
### Step 3 only creates straight lines between the incident roads





import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

## Step 1 setting our boundry for the junction 
## Boundries (square) x : {1, 11}, y : {1,11}


fig, ax = plt.subplots()
square = patches.Rectangle((1, 1), 10, 10, edgecolor='red', facecolor='none')
ax.add_patch(square)
plt.xlim(0, 12)
plt.ylim(0, 12)
plt.gca().set_aspect('equal', adjustable='box')  # Ensures equal aspect ratio


## Incidident Roads :  BASIC N = 3 CASE 

I1_x = np.linspace(0,1,11) ## I1 x-axis vector
I1_y = np.full(11, 6) ## I1 y-axis vector
plt.plot(I1_x,I1_y,linestyle='-', linewidth=2,color='blue') ## I1 Vector - y(t) = (t,6) t in [0,1] ## start point in boundy (1,6)

I2_x = np.linspace(11,12,11) ## I2 x-axis vector
I2_y = np.full(11, 6) ## I2 y-axis vector
plt.plot(I2_x,I2_y,linestyle='-', linewidth=2,color='blue') ## I2 Vector - y(t) = (12-t,6) t in [0,1] ## start point in boundy (11,6)


I3_x = np.full(11, 6) ## I3 x-axis vector
I3_y = np.linspace(0,1,11) ## I3 y-axis vector
plt.plot(I3_x,I3_y,linestyle='-', linewidth=2,color='blue') ## I3 Vector - y(t) = (6,t) t in [0,1] ## start point in boundy (6,1)




## Step 3 Creating the Helper Roads.

Incident_Roads = [ [1,6],[11,6],[6,1] ] ### start points

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