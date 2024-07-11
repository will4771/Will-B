'''
You are given a string s that consists of lower case English letters and brackets.

Reverse the strings in each pair of matching parentheses, starting from the innermost one.

Your result should not contain any brackets.
'''

class Solution(object):
    def reverseParentheses(self, s):
        stack = []
        s = list(s)
        
        for i, char in enumerate(s):
            if char == '(':
                stack.append(i)
            elif char == ')':
                start = stack.pop()
                end = i
                s[start + 1:end] = s[start + 1:end][::-1]
        return ''.join([char for char in s if char not in '()'])
            