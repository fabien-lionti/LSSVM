import numpy as np

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

    def fit(self, x, y):
        self.x = x
        self.alpha, self.bias = self.get_params(x, y)
        return

    def predict(self, x_test):
        K = self.kernel(self.x, x_test)
        output = np.dot(K, self.alpha) + self.bias
        return output