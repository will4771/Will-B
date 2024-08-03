'''

You are given two integer arrays of equal length target and arr. In one step, 

you can select any non-empty subarray of arr and reverse it. You are allowed to make any number of steps.


'''

class Solution:
    def canBeEqual(self, target: List[int], arr: List[int]) -> bool:
        if sorted(target) == sorted(arr):
            return True
        else:
            return False
        

 