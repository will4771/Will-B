'''
Given an integer x, return true if x is a palindrom, and false otherwise.
'''

class Solution(object):
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        y = str(x)
        if y == y[::-1]:
            return True
        return False