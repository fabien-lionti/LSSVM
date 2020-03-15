import numpy as np

class LSSVM:

    def __init__(self, gamma, kernel_name, **kernel_params):
        self.gamma = gamma
        self.x = None
        self.y = None
        self.alpha = None
        self.bias = None
        self.kernel = self.get_kernel(kernel_name)
        self.kernel_params = kernel_params

class LSSVR:

    def __init__(self, gamma, kernel_name, **kernel_params):
        self.gamma = gamma
        self.x = None
        self.y = None
        self.alpha = None
        self.bias = None
        self.kernel = self.get_kernel(kernel_name)
        self.kernel_params = kernel_params

    def rbf(self, x, y):
        sigma = self.kernel_params['sigma']
        if x.ndim > 1:
            x2 = sum((x ** 2).T).reshape(x.shape[0], 1)
            y2 = sum((y ** 2).T).reshape(y.shape[0], 1)
        else:
            x2 = x * x
            y2 = y * y
        norm = y2[:, ...] + x2.T - 2 * np.dot(y, x.T)
        return np.exp(-norm / (2 * sigma ** 2))

    def polynomial(self, x, y):
        d = self.kernel_params['d']
        return (np.dot(x, y.T) + 1) ** d

    def linear(self, x, y):
        return np.dot(x, y.T)

    def tanh(self, x, y):
        k = self.kernel_params['k']
        phi = self.kernel_params['phi']
        return np.tanh(k * np.dot(x, y.T) + phi)

    def get_kernel(self, kernel_name):
        kernel_list = {
            "rbf": self.rbf,
            "polynomial": self.polynomial,
            "linear": self.linear,
            "tanh": self.tanh
        }

        try:
            return kernel_list[kernel_name]

        except KeyError:
            print('Undefined kernel name \n Available kernels : ' + str(kernel_list.keys))
            return None

    def get_params(self, x, y):
        N = len(x)
        I = np.eye(N)
        V1 = np.ones((N, 1))
        K = self.kernel(x, x)
        A = np.block([[0, V1.T], [V1, K + I / self.gamma]])
        Ainv = np.linalg.pinv(A)
        B = np.block([[np.zeros(1)], [y]])
        params = np.dot(Ainv, B)
        bias = params[0]
        alpha = params[1:]
        return alpha, bias

    def rmse(self, Y_train, Y_test):
        N = len(Y_train)
        residual = (Y_train - Y_test)**2
        RMSE = np.sqrt(np.sum(residual/N))
        return RMSE

    def fit(self, x, y):
        self.x = x
        self.alpha, self.bias = self.get_params(x, y)
        return

    def predict(self, x_test):
        K = self.kernel(self.x, x_test)
        output = np.dot(K, self.alpha) + self.bias
        return output

def test():

    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    # preparing the training data

    n = 1000
    mu = 0
    sigma_ = 0.5
    noise = np.random.normal(mu, sigma_, n).reshape(n,1)
    X = np.array(np.linspace(1,4*np.pi,n)).reshape(n,1)
    Y = np.sin(X) + noise
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Training

    lssvr = LSSVR(gamma=100, kernel_name="rbf", sigma=1)
    lssvr.fit(X_train, y_train)
    y_pred = lssvr.predict(X_test)
    rms = lssvr.rmse(y_pred,y_test)
    print('Error : '+str(rms))

    plt.figure(figsize=(10,10))
    plt.plot(X_train, y_train, '.')
    plt.plot(X_test, y_pred, '.')
    plt.show()