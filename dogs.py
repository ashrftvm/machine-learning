import numpy as np
import matplotlib.pyplot as plt


greyhounds = 500
labs = 500

# 28 and 24 inches with plus or minus 4 inches by generating random numbers
grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'])
plt.show()