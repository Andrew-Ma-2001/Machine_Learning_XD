import pandas as pd
import matplotlib.pyplot as plt
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