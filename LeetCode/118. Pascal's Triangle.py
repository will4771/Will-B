'''

Given an integer numRows, return the first numRows of Pascal's triangle.

'''

class Solution(object):
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        if numRows == 1:
            return [[1]]
        else:
            res = [[1]]
            for i in range(1, numRows):
                prev_row = res[-1]
                new_row = [1] + [prev_row[j] + prev_row[j+1] for j in range(len(prev_row)-1)] + [1]
                res.append(new_row)
        
        return res
