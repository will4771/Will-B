'''
A distinct string is a string that is present only once in an array.

Given an array of strings arr, and an integer k, return the kth distinct string present in arr. 

If there are fewer than k distinct strings, return an empty string "".

Note that the strings are considered in the order in which they appear in the array.
'''



from typing import List

class Solution:
    def kthDistinct(self, arr: List[str], k: int) -> str:
        distinct_count = {}
        
        
        for string in arr:
            if string in distinct_count:
                distinct_count[string] += 1
            else:
                distinct_count[string] = 1
        
        distinct_strings = [string for string in arr if distinct_count[string] == 1]
        
        if k <= len(distinct_strings):
            return distinct_strings[k - 1]
        else:
            return ""
