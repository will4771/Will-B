'''
You are given the array nums consisting of n positive integers. You computed the sum of all non-empty 
continuous subarrays from the array and then sorted them in non-decreasing order, creating a new array 
of n * (n + 1) / 2 numbers.

Return the sum of the numbers from index left to index right (indexed from 1), inclusive, 
in the new array. Since the answer can be a huge number return it modulo 109 + 7.


'''


class Solution:
    def rangeSum(self, nums: List[int], n: int, left: int, right: int) -> int:

        sub_sums = []
        

        for i in range(0,len(nums)):
            for j in range(i + 1, len(nums) + 1):
                sub_sums.append(sum(nums[i:j]))


        sub_sums = sorted(sub_sums)

        return (sum(sub_sums[left-1:right])) % (10**9 + 7)
        