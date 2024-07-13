import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = np.array([1993, 1995, 1996, 1997, 1998, 1999, 2002]).reshape(-1, 1)
y = np.array([77, 88, 94, 85, 91, 98, 90]).reshape(-1, 1)

reg = LinearRegression().fit(X, y)
print(reg.score(X, y))

print(reg.coef_)
print(reg.intercept_)

print(1.37640449 + (-2659.87640449))

plt.scatter(X, y)
plt.plot(X,  reg.predict(X))
plt.show()

