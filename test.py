import build.zspline as m
import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([0.01, 0.75, 1.2, 1.4, 0.2, 0.8, 0.6, 0, 0.5, 0.25])

Z = m.Z_spline(2, X, y)

test = np.linspace(1, 10, 900)


plt.plot(test, Z(test))
plt.scatter(X, y)
plt.show()