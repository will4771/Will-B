'''
Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.

The overall run time complexity should be O(log (m+n)).
'''

class Solution(object):

    def median(self, lst):
        n = len(lst)
        return (lst[n // 2 - 1] + lst[n // 2]) / 2.0 if n % 2 == 0 else lst[n // 2]

    def findMedianSortedArrays(self, nums1, nums2):
        if len(nums1) == 0:
            return self.median(nums2)
        
        if len(nums2) == 0:
            return self.median(nums1)

        for num in nums2:
            for j in range(len(nums1)):
                if num <= nums1[j]:
                    nums1.insert(j, num)
                    break
            else:
                nums1.append(num)

        return self.median(nums1)
    

solution = Solution()

# Test cases
print(solution.findMedianSortedArrays([1, 3], [2]))  # Output: 2.0
print(solution.findMedianSortedArrays([1, 2], [3, 4]))  # Output: 2.5
print(solution.findMedianSortedArrays([0, 0], [0, 0]))  # Output: 0.0
print(solution.findMedianSortedArrays([], [1]))  # Output: 1.0
print(solution.findMedianSortedArrays([2], []))  # Output: 2.0