import math

def standardDeviation(arr):
    """
    Calculates the standard deviation of an array.
    
    Parameters:
        arr: List or NumPy array of numerical values.
    
    Returns:
        Standard deviation as a float.
    """
    mean = sum(arr) / len(arr)
    variance = sum((x - mean) ** 2 for x in arr) / len(arr)
    return math.sqrt(variance)