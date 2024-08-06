'''
 Pythagorean triplet is a set of three natural numbers, a,b,c, for which, a^2 + b^2 = c^2

There exists exactly one Pythagorean triplet for which a + b + c = 1000

Find the product abc

'''

for m in range(3, 100):
    for n in range(m + 1, 100):
        a = n**2 - m**2
        b = 2 * m * n
        c = n**2 + m**2
        
        if a + b + c == 1000:
            print(a * b * c)
            break
