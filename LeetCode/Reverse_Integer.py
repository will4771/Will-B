'''
Given a signed 32-bit integer x, return x with its digits reversed. If reversing x causes the value to go 
outside the signed 32-bit integer range [-231, 231 - 1], then return 0.

'''

class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        Positive = True
        if x < 0:
            Positive = False
            x = -x
        
        x_str = str(x)
        x_str = x_str[::-1]
        res = int(x_str)
        
        if Positive == False:
            res = -res
        if res >= (2**31) -1 or res <= -(2**31) :
            return 0
        return res