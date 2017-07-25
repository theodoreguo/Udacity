""" quiz materials for feature scaling clustering """

### FYI, the most straightforward implementation might 
### throw a divide-by-zero error, if the min and max
### values are the same
### but think about this for a second--that means that every
### data point has the same value for that feature!  
### why would you rescale it?  Or even use it at all?
import numpy

def featureScaling(arr):
    arr = numpy.array(arr, dtype=float)
    max = numpy.max(arr)
    min = numpy.min(arr)
    result = []
    
    for i in arr:
        result.append((i - min) / (max - min))

    return result

# tests of your feature scaler--line below is input data
data = [115, 140, 175]
print featureScaling(data)