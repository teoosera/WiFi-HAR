import numpy as np

def test_function(n):
    print(f"Running test_function with n={n}")
    arr = np.arange(n)
    print(f"Generated array: {arr}")
    return arr
