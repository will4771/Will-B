'''
You start at the cell (rStart, cStart) of an rows x cols grid facing east. The northwest corner is at the first row and column in the grid, and the southeast corner is at the last row and column.

You will walk in a clockwise spiral shape to visit every position in this grid. Whenever you move outside the grid's boundary, we continue our walk outside the grid (but may return to the grid boundary later.). Eventually, we reach all rows * cols spaces of the grid.

Return an array of coordinates representing the positions of the grid in the order you visited them.

'''

class Solution:
    def spiralMatrixIII(self, rows: int, cols: int, rStart: int, cStart: int) -> List[List[int]]:

        coord = [ [rStart, cStart] ]

        ans = [ [rStart, cStart] ]
        
        k = 0 # index of coordinates
        count = 1 # amount of step to take in a direction
        direction = 1   # or -1 

        while len(ans) < rows*cols :  

            for i in range(count):  # Completes the row dirctions (count times) in direction 1 or -1
                coord.append([coord[k][0], coord[k][1] + direction ] ) 

                k+= 1

                if 0 <= coord[k][0] < rows  and 0 <= coord[k][1] < cols: # checks if is in soultion boundry
                    ans.append(coord[k])

            for i in range(count): # Completes the column dirctions (count times) in direction 1 or -1
                coord.append([coord[k][0] + direction, coord[k][1]] )

                k += 1

                if 0 <= coord[k][0] < rows and 0 <= coord[k][1] < cols:  # checks if is in soultion boundry
                    ans.append(coord[k])

            direction = direction * -1  
            count +=1 
                
            
        return ans
                


            


