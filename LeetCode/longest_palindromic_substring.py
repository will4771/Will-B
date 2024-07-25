'''
Given a string s, return the longest palindromic substring in s.

'''

class Solution(object):
    def Is_Palindromic(self, lis):
        return lis == lis[::-1]

    def longestPalindrome(self, s):
        if self.Is_Palindromic(s):
            return s

        Res = "a"
        for i in range(len(s)):
            for j in range(i + 1, len(s) + 1):
                if self.Is_Palindromic(s[i:j]):
                    if len(s[i:j]) > len(Res):
                        Res = s[i:j]
        return Res

# Test cases
def run_tests():
    sol = Solution()

    # Test case 1: Single character string
    assert sol.longestPalindrome("a") == "a", "Test case 1 failed"

    # Test case 2: Simple palindrome
    assert sol.longestPalindrome("aba") == "aba", "Test case 2 failed"

    # Test case 3: Entire string is a palindrome
    assert sol.longestPalindrome("racecar") == "racecar", "Test case 3 failed"

    # Test case 4: Longest palindrome in the middle
    assert sol.longestPalindrome("babad") in ["bab", "aba"], "Test case 4 failed"

    # Test case 5: Longest palindrome at the end
    assert sol.longestPalindrome("cbbd") == "bb", "Test case 5 failed"

    # Test case 6: No palindrome longer than 1 character
    assert sol.longestPalindrome("abc") in ["a", "b", "c"], "Test case 6 failed"

    # Test case 7: String with spaces
    assert sol.longestPalindrome("a man a plan a canal panama") in ["a man a plan a canal panama", " a ", "ana", " a p "], "Test case 7 failed"

    # Test case 8: Mixed case string
    assert sol.longestPalindrome("Aba") in ["A", "b", "a"], "Test case 8 failed"

    # Test case 9: Empty string
    assert sol.longestPalindrome("") == "", "Test case 9 failed"

    # Test case 10: Long string with repeating characters
    assert sol.longestPalindrome("aaaaa") == "aaaaa", "Test case 10 failed"

    print("All test cases passed!")

# Run the test cases
run_tests()
