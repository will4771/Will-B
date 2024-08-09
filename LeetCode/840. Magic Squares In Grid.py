'''
A 3 x 3 magic square is a 3 x 3 grid filled with distinct numbers from 1 to 9 such that each row, column, and both diagonals all have the same sum.

Given a row x col grid of integers, how many 3 x 3 contiguous magic square subgrids are there?

Note: while a magic square can only contain numbers from 1 to 9, grid may contain numbers up to 15.

'''


from typing import List

class Solution:
    def numMagicSquaresInside(self, grid: List[List[int]]) -> int:
        if len(grid[0]) < 3 or len(grid) < 3:
            return 0

        def isMagicSquare(x, y):
            s = set()
            for i in range(3):
                for j in range(3):
                    num = grid[x + i][y + j]
                    if num < 1 or num > 9 or num in s:
                        return False
                    s.add(num)

            if sum(grid[x][y:y + 3]) != 15:
                return False
            if sum(grid[x + 1][y:y + 3]) != 15:
                return False
            if sum(grid[x + 2][y:y + 3]) != 15:
                return False

            if grid[x][y] + grid[x + 1][y] + grid[x + 2][y] != 15:
                return False
            if grid[x][y + 1] + grid[x + 1][y + 1] + grid[x + 2][y + 1] != 15:
                return False
            if grid[x][y + 2] + grid[x + 1][y + 2] + grid[x + 2][y + 2] != 15:
                return False

            if grid[x][y] + grid[x + 1][y + 1] + grid[x + 2][y + 2] != 15:
                return False
            if grid[x][y + 2] + grid[x + 1][y + 1] + grid[x + 2][y] != 15:
                return False

            return True

        res = 0
        for i in range(len(grid) - 2):
            for j in range(len(grid[0]) - 2):
                if isMagicSquare(i, j):
                    res += 1

        return res
