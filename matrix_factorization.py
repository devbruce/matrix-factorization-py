import numpy as np


__all__ = ['MatrixFactorization']


class MatrixFactorization:
    
    def __init__(self, R:np.ndarray, K:int, alpha:float, beta:float=0.02):
        """
        Args:
            R (np.ndarray): User-Item rating matrix
            K (int): Number of latent dimensions
            alpha (float): Learning rate
            beta (float): Regularization parameter
        """
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        
        # Create a training samples(Not np.nan indices)
        self.train_ds = np.argwhere(~np.isnan(self.R))
        
    def train(self, epochs:int):
        # Initialize user and item latent feature matrixes
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        
        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[~np.isnan(self.R)])
        
        # Perform stochastic gradient descent for number of iterations
        hist = []
        for epoch in range(1, epochs+1):
            mse = self.sgd()
            hist.append(mse)
            print(f'Epoch: {epoch:3} | MSE: {mse:.4f}')
        return hist

    def sgd(self) -> np.float64:
        """
        Update weights with Stochastic Gradient Descent
        
        Returns:
            np.float64: Total mean squared error
        """
        train_ds = self.train_ds.copy()
        np.random.shuffle(train_ds)
        
        total_mse = 0.0
        for i, j in train_ds:  # user i, item j
            label = self.R[i, j]
            pred = self.get_pred_rating(i, j)
            e = label - pred
            
            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])
            
            # Create copy of row of P since we need to update it but use older values for update on Q
            P_i = self.P[i, :].copy()
            
            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * P_i - self.beta * self.Q[j,:])
            
            total_mse += np.square(e)
        total_mse = np.sqrt(total_mse)
        return total_mse

    def get_pred_rating(self, i:int, j:int) -> np.float64:
        """
        Get the predicted rating of user i and item j
        """
        pred = self.b + self.b_u[i] + self.b_i[j] + (self.P[i, :] @ self.Q[j, :].T)
        return pred
    
    def get_pred_full_matrix(self) -> np.ndarray:
        """
        Get the full matrix using the resultant biases, P and Q
        """
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + (self.P @ self.Q.T)


if __name__ == '__main__':
    np.random.seed(1234)

    R = np.array([
        [     5,      3, np.nan, 1],
        [     4, np.nan, np.nan, 1],
        [     1,      1, np.nan, 5],
        [     1, np.nan, np.nan, 4],
        [np.nan,      1,      5, 4],
    ])

    mf = MatrixFactorization(
        R=R,
        K=2,        # Number of latent features
        alpha=0.1,  # Learning Rate
        beta=0.01,
    )

    print('\n====== Training Matrix Factorization ======')
    hist = mf.train(epochs=20)
    print()

    pred_matrix = mf.get_pred_full_matrix()
    print('====== Predicted full matrix ======')
    print(pred_matrix.round(2))
    print()
    