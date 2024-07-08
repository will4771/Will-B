"""
Given a string s, find the length of the longest substringwithout repeating characters.

"""

class Solution(object):
    def lengthOfLongestSubstring(self, s):
        Output = 0
        Check_list = []
        count = 0
        n = len(s)

        for i in range(n):
            if s[i] not in Check_list:
                Check_list.append(s[i])
                count += 1
                Output = max(Output, count)  # Update Output here to check the maximum count
            else:
                while s[i] in Check_list:
                    Check_list.pop(0)  # Remove the first character until s[i] is not in Check_list
                    count -= 1  # Decrease the count as we are removing characters
                Check_list.append(s[i])
                count += 1
                Output = max(Output, count)  # Update Output here to check the maximum count

        return Output
