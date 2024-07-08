"""
Given an integer array arr, return true if there are three consecutive odd numbers in the array. 

Otherwise, return false.

"""

class Solution(object):
    
    def threeConsecutiveOdds(self, arr):
        for i in range((len(arr)-2)):
            if arr[i] % 2 and arr[i+1] % 2 and arr[i+2] %2 :
                return True
        return False
       

        



      
        