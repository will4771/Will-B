'''
Given an integer array nums, find the subarray with the largest sum, and return its sum.
'''

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        
        current_sum = nums[0]
        max_sum = nums[0]
        
       
        for num in nums[1:]:
            # Update the current subarray sum to include the current element
            # or start a new subarray starting at the current element, whichever is larger
            current_sum = max(num, current_sum + num)

            max_sum = max(max_sum, current_sum)
        
        return max_sum
