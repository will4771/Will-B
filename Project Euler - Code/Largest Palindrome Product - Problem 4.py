'''A palindromic number reads the same both ways. The largest palindrome made from the product of two 
2-digit numbers is 

9009 = 91 * 99

Find the largest palindrome made from the product of two 3-digit numbers.

'''


def palindrome(a):
    a_s = str(a)
    if a_s == a_s[::-1]:
        return True

a = list(range(0,1000))
b = list(range(0,1000))


temp = 0 
greatest = 0
for i in range(len(a)):
    for j in range(len(b)):
        temp = a[i] * b[j]

        if palindrome(temp):
            if temp > greatest:
                greatest = temp

print(greatest)