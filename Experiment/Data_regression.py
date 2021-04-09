import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data = pd.read_csv('advertising.csv')
x = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Showing the dataset
plt.figure(figsize=(9,12))
plt.subplot(311)
plt.plot(data['TV'], y, 'ro')
plt.title('TV')
plt.grid()
plt.subplot(312)
plt.plot(data['Radio'], y, 'go')
plt.title('Radio')
plt.grid()
plt.subplot(313)
plt.plot(data['Newspaper'], y, 'bo')
plt.title('Newspaper')
plt.grid()
plt.tight_layout()
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
linreg = LinearRegression()
model = linreg.fit(x_train, y_train)
print(model)
print(linreg.coef_)
print(linreg.intercept_)

y_hat = linreg.predict(np.array(x_test))
mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error
rmse = np.sqrt(mse)  # Root Mean Squared Error
print(mse, rmse)

t = np.arange(len(x_test))
plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
plt.legend(loc='upper right')
plt.grid()
plt.show()

# todo 应该把原理搞清楚
