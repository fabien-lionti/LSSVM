
## Quick Start

```python
# preparing the training data

n = 1000
mu = 0
sigma_ = 0.5
noise = np.random.normal(mu, sigma_, n).reshape(n,1)
X = np.array(np.linspace(1,4*np.pi,n)).reshape(n,1)
Y = np.sin(X) + noise
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Training

lssvr = LSSVR(gamma=10, kernel_name="rbf", sigma=1)
lssvr.fit(X_train, y_train)
y_pred = lssvr.predict(X_test)
rms = lssvr.rmse(y_pred,y_test)
print('Error : '+str(rms))

plt.figure(figsize=(10,10))
plt.plot(X_train, y_train, '.')
plt.plot(X_test, y_pred, '.')
plt.show()
```




