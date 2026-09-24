import numpy as np

# Define the input vector
x = np.array([-25, 67, 0, 88.88, -4.7, -11.27, 5.6])

# Apply the ReLU formula: f(x) = max(0, x)
y = np.maximum(0, x)

# Output the result
print(y)