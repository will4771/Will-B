'''
The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)

P   A   H   N
A P L S I I G
Y   I   R
And then read line by line: "PAHNAPLSIIGYIR"

Write the code that will take a string and make this conversion given a number of rows:

string convert(string s, int numRows);

'''








class Solution:
    def convert(self, s, numRows):
        if numRows == 1 or numRows >= len(s):
            return s
        
        # Create an array of empty strings for each row
        rows = [''] * numRows
        current_row = 0
        direction = -1  # -1 means moving up, 1 means moving down
        
        for char in s:
            rows[current_row] += char
            
            # Change direction if we reach the top or bottom row
            if current_row == 0 or current_row == numRows - 1:
                direction *= -1
            
            current_row += direction
        
        # Join all rows to form the final string
        return ''.join(rows)


