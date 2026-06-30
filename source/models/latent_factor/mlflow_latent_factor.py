import numpy as np
import pandas as pd
from loguru import logger
from mlflow.pyfunc import PythonModel

class LatentFactorModel(PythonModel):
    " Define a class for Latent Factor Model. "
    def __init__(self, k, var_name='power', var_P='farmId', var_U='periodId', warm_start=False, verbose=False, learning_rate=None, random_state=42):
        """
        Initialize the LatentFactorModel object.
        """

        assert isinstance(k, int), "Input k must be an integer."
        assert isinstance(var_name, str), "Input var_name must be a string."
        assert isinstance(var_P, str), "Input var_P must be a string."
        assert isinstance(var_U, str), "Input var_U must be a string."
        assert isinstance(warm_start, bool), "Input warm_start must be a boolean."
        assert isinstance(verbose, bool), "Input verbose must be a boolean."
        assert isinstance(learning_rate, float) or learning_rate is None, "Input learning_rate must be a float or None."
        assert isinstance(random_state, int), "Input random_state must be an integer."

        # Initialize parameters
        self.k = k  # Number of latent factors
        self.var_name = var_name  # Target variable
        self.var_P = var_P  # Variable for matrix P
        self.var_U = var_U  # Variable for matrix U
        self.warm_start = warm_start  # Warm start
        self.verbose = verbose
        self.learning_rate=learning_rate  # Learning rate for SGD
        self.random_state = random_state

    def initialize_matrix(self, n, k, warm_start):
        """
        Initialize a matrix with random values or warm start.
        """
        # Set random seed
        np.random.seed(self.random_state)
        # Initialize matrix
        matrix = np.random.rand(n, k)
        # Warm start with average values
        if warm_start:
            avg_values = np.mean(matrix, axis=1)
            matrix[:, 0] = avg_values
        return matrix

    def least_squares(self, list_ids, X, lambda_reg):
        """
        Perform least squares optimization.
        """
        indices = list_ids[:, 0].astype(int)
        X_subset = X[indices]
        # Compute XtX and XtY
        XtX = np.dot(X_subset.T, X_subset) + lambda_reg * np.eye(X_subset.shape[1])
        XtY = np.dot(X_subset.T, list_ids[:, 1])
        return np.linalg.solve(XtX, XtY).T  # Solve the linear system

    def _alternating_least_squares(self, U, P, n_epochs, train_df, valid_df, lambda_reg_U, lambda_reg_P):
        """
        Perform Alternating Least Squares (ALS) optimization.
        """

        assert isinstance(P, np.ndarray), "Input P must be a numpy array."
        assert isinstance(n_epochs, int), "Input n_epochs must be an integer."
        assert isinstance(train_df, pd.DataFrame), "Input train_df must be a pandas DataFrame."
        assert isinstance(valid_df, pd.DataFrame), "Input valid_df must be a pandas DataFrame."
        assert isinstance(lambda_reg_U, float), "Input lambda_reg_U must be a float."
        assert isinstance(lambda_reg_P, float), "Input lambda_reg_P must be a float."

        train_errors, valid_errors = [], []
        for epoch in range(n_epochs):
            # Update U and P
            # --- Atualizar U explicitamente por índice ---
            for u_idx, group in train_df.groupby(self.var_U):
                # Extrai a matriz X e os targets para aquele período específico
                X_vals = group[[self.var_P, self.var_name]].values
                # Atualiza apenas a linha correspondente em U
                U[u_idx] = self.least_squares(X_vals, P, lambda_reg_P)
                
            # --- Atualizar P explicitamente por índice ---
            for p_idx, group in train_df.groupby(self.var_P):
                # Extrai a matriz X e os targets para aquela variável(farm) específica
                X_vals = group[[self.var_U, self.var_name]].values
                # Atualiza apenas a linha correspondente em P
                P[p_idx] = self.least_squares(X_vals, U, lambda_reg_U)
            # Compute training and validation errors
            train_preds = self.predict(U, P, train_df)
            train_error = self.compute_rmse(train_df[self.var_name].values, train_preds)
            train_errors.append(train_error)
            valid_preds = self.predict(U, P, valid_df)
            valid_error = self.compute_rmse(valid_df[self.var_name].values, valid_preds)
            valid_errors.append(valid_error)
            if self.verbose:
                logger.info('____________________')
                logger.info(f' Iteration: {epoch}')
                logger.info(f' mse training error: {train_error}')
                logger.info(f' mse validation error {valid_error}')
        return U, P


    def _stochastic_gradient_descent(self, P, U, n_epochs, train_df, valid_df, lambda_reg_U, lambda_reg_P):
        """
        Perform Stochastic Gradient Descent (SGD) optimization.
        """

        assert isinstance(P, np.ndarray), "Input P must be a numpy array."
        assert isinstance(U, np.ndarray), "Input U must be a numpy array."
        assert isinstance(n_epochs, int), "Input n_epochs must be an integer."
        assert isinstance(train_df, pd.DataFrame), "Input train_df must be a pandas DataFrame."
        assert isinstance(valid_df, pd.DataFrame), "Input valid_df must be a pandas DataFrame."
        assert isinstance(lambda_reg_U, float), "Input lambda_reg_U must be a float."
        assert isinstance(lambda_reg_P, float), "Input lambda_reg_P must be a float."

        train_errors, valid_errors = [], []
        for epoch in range(n_epochs):
            for row in train_df.itertuples():
                # Update U and P for each row
                u, p, target = row.__getattribute__(self.var_U), row.__getattribute__(self.var_P), row.__getattribute__(self.var_name)  # Get row values
                error = target - np.dot(U[u], P[p])  # Compute error
                U[u] += self.learning_rate * (error * P[p] - lambda_reg_U * U[u])  # Update U
                P[p] += self.learning_rate * (error * U[u] - lambda_reg_P * P[p])  # Update P
            train_preds = self.predict(U, P, train_df)  # Compute training predictions
            train_error = self.compute_rmse(train_df[self.var_name].values, train_preds)  # Compute training error
            train_errors.append(train_error)
            valid_preds = self.predict(U, P, valid_df)  # Compute validation predictions
            valid_error = self.compute_rmse(valid_df[self.var_name].values, valid_preds)  # Compute validation error
            valid_errors.append(valid_error)
            if self.verbose:
                logger.info('____________________')
                logger.info(f' Iteration: {epoch}')
                logger.info(f' mse training error: {train_error}')
                logger.info(f' mse validation error {valid_error}')
        return U, P

    def factorization(self, train_df, valid_df, solver='als', n_epochs=5, lambda_reg_P=50, lambda_reg_U=50):
        """
        Perform matrix factorization using ALS or SGD.
        """
        n_p = len(train_df[self.var_P].unique())  # Number of unique values for variable P
        n_u = len(train_df[self.var_U].unique())  # Number of unique values for variable U
        P = self.initialize_matrix(n_p, self.k, self.warm_start)  # Initialize matrix P
        U = self.initialize_matrix(n_u, self.k, self.warm_start)  # Initialize matrix U
        if solver == 'als':
            U, P = self._alternating_least_squares(U=U, P=P, n_epochs = n_epochs,
                                                   train_df=train_df, valid_df=valid_df,
                                                   lambda_reg_U =lambda_reg_U, lambda_reg_P=lambda_reg_P)
        elif solver == 'sgd':
            U, P = self._stochastic_gradient_descent(P=P, U=U, n_epochs=n_epochs,
                                                    train_df=train_df, valid_df=valid_df,
                                                    lambda_reg_U=lambda_reg_U, lambda_reg_P=lambda_reg_P)
        return U, P

    def predict(self, U, P, df):
        """
        Make predictions using matrices U and P.
        """
        predictions = np.zeros(len(df))
        for i, row in enumerate(df.itertuples()):
            u, p = row.__getattribute__(self.var_U), row.__getattribute__(self.var_P)  # Get row values
            predictions[i] = np.dot(U[u], P[p])  # Compute prediction
        return predictions

    @staticmethod
    def compute_rmse(targets, predictions):
        """
        Compute the Root Mean Squared Error (RMSE) between targets and predictions.
        """
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        return rmse
