'''
Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.

You have the following three operations permitted on a word:

Insert a character
Delete a character
Replace a character


'''


class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        
        # Create a DP table with size (m+1) x (n+1)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize the DP table
        for i in range(m + 1):
            dp[i][0] = i  # Cost of converting any prefix of word1 to an empty word2
        for j in range(n + 1):
            dp[0][j] = j  # Cost of converting an empty word1 to any prefix of word2
        
        # Fill the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]  # Characters match, no new operation needed
                else:
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,     # Deletion
                        dp[i][j - 1] + 1,     # Insertion
                        dp[i - 1][j - 1] + 1  # Replacement
                    )
        
        # The result is in the bottom-right corner of the DP table
        return dp[m][n]