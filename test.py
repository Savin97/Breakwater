import numpy as np
import random
import matplotlib.pyplot as plt

days_to_earnings = [n for n in range(31) ]
list_to_plot=[]
for day in days_to_earnings:
    # days_to_earnings >= 30  → 0
    # days_to_earnings <= 0   → 100
    base = 1 - np.clip(day / 30 , 0, 1)
    # Non-linear pressure near earnings
    base = base ** 1.5
    print(base, "\n-----------\n")
    list_to_plot.append((day, base))

plt.plot(list_to_plot)
plt.show()