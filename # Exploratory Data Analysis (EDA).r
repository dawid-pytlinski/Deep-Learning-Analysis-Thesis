# Exploratory Data Analysis (EDA)
print("\nPodstawowe statystyki:")
print(data.describe())

print("\nInformacje o danych:")
data.info()

plt.figure(figsize=(12, 6))
plt.plot(data.index, data['price'], label='Price', color='blue')
plt.title('Bitcoin Price Over Time')
plt.xlabel('Historical date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid()
plt.show()

data['month'] = data.index.month
monthly_avg = data.groupby('month')['price'].mean()
plt.figure(figsize=(10, 5))
monthly_avg.plot(kind='bar', color='orange', alpha=0.7)
plt.title('Average Monthly Bitcoin Price')
plt.xlabel('Month')
plt.ylabel('Average Price (USD)')
plt.grid(axis='y')
plt.show()

data['price'].plot(kind='hist', bins=50, figsize=(10, 5), title='Price Distribution', color='blue')
plt.xlabel('Price')
plt.grid()
plt.show()

decomposition = seasonal_decompose(data['price'], model='additive', period=30)

plt.figure(figsize=(14, 8))

plt.subplot(4, 1, 1)
plt.plot(decomposition.observed, label='Observed', color='blue')
plt.title('Observed')
plt.grid()


plt.subplot(4, 1, 2)
plt.plot(decomposition.trend, label='Trend', color='orange')
plt.title('Trend')
plt.grid()


plt.subplot(4, 1, 3)
plt.plot(decomposition.seasonal, label='Seasonal', color='green')
plt.title('Seasonal')
plt.grid()


plt.subplot(4, 1, 4)
plt.plot(decomposition.resid, label='Residual', color='red')
plt.title('Residual')
plt.grid()


plt.tight_layout()
plt.show()



