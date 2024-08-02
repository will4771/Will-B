'''

A swap is defined as taking two distinct positions in an array and swapping the values in them.

A circular array is defined as an array where we consider the first element and the last element to be adjacent.

Given a binary circular array nums, return the minimum number of swaps required to group all 1's present in the array together at any location.

'''


class Solution(object):
    def minSwaps(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        length = sum(nums)
        new_nums = nums* 2
        num = 0
        n = len(nums)

        if length == 0 or length == n:
            return 0

        current_sum = sum(new_nums[:length])
        if current_sum == length:
            return 0

        for i in range(1,n):
            current_sum = current_sum - new_nums[i - 1] + new_nums[i + length - 1]
            num = max(num,current_sum)

        return length - num