import numpy as np
print(np.array([1, 2]))

def fabo(n):
    if n <= 2:
        return n
    else:
        return fabo(n-1) + fabo(n-2)
