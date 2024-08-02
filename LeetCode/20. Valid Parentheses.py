'''

Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.
Every close bracket has a corresponding open bracket of the same type.
 
'''

class Solution:
    def isValid(self, s: str) -> bool:

        if len(s) == 0:
            return True

        if s[0] == ")" or s[0] == "}" or s[0] == "]":
            return False

        t = []

        for i in range(len(s)):
            if s[i] == "(" or s[i] == "{" or s[i] == "[":
                t.append(s[i])
            elif s[i] == ")" or s[i] == "}" or s[i] == "]":
                if len(t) == 0:
                    return False
                top = t.pop()
                if (s[i] == ")" and top != "(") or (s[i] == "}" and top != "{") or (s[i] == "]" and top != "["):
                    return False

        if len(t) == 0:
            return True
        else:
            return False
