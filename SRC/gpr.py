import copy
import numpy as np
import torch
import gpytorch
from scipy.stats import kurtosis
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from SRC.utils import *
import cvxpy as cp

class ExactGPModel(gpytorch.models.ExactGP):
    '''
    Subclass used to build an exact GP Model inheriting from the ExactGP model.

    Args:
        X_train (np.ndarray): Design features matrix of dimensions (n_cases, d),
                              where d is the number of design parameters
        Y_train (np.ndarray): Matrix of size (p, q) where p is the number of operating conditions
                              and q is the number of retained POD coefficients.
        likelihood (gpytorch.likelihoods): One dimensional likelihood from gpytorch.likelihoods  
        mean (gpytorch.means:) Mean function gpytorch.means
        kernel (gpytorch.kernels): Kernel function from gpytorch.kernels
    '''
    
    def __init__(self, X_train: float, Y_train: float, likelihood, mean, kernel):
        super().__init__(X_train, Y_train, likelihood)
        
        self.mean_module = mean
        self.covar_module = kernel
            
    def forward(self, x):
        '''
        Returns the multivariate distribution given the input x
        
        Args:
            x (Tensor[float]): Input to the multivariate distribution
        '''

        mean_x = self.mean_module(x)
        kernel_x = self.covar_module(x)

        return MultivariateNormal(mean_x, kernel_x)

class GPR(ROM):
    '''
    Class used for building a GPR-based ROM.
    
    Args:
        X (np.ndarray): 2D matrix of size (n_feature * n_points, n_cases or n, p), 
                        with n_cases being the number of operating condition explored
        X_species (np.ndarray): 2D matrix of size (n_feature_species * n_points, n_cases or n, p), 
                        with n_cases being the number of operating condition explored
        X_flame (np.ndarray): 2D matrix of size (n_feature_flame * n_points, n_cases or n, p), 
                        with n_cases being the number of operating condition explored
        n_features (int): number of features in the dataset (temperature, velocity, etc.)
        n_features_species (int): number of species features in the dataset (temperature, ch4 mass fraction, etc.)
        n_features_flame (int): number of species features in the dataset (temperature, ch4 mass fraction, etc.)
        P (np.ndarray): Design features matrix of dimensions (p, d),
                        where p is the number of operating conditions and d is the number of design parameters
        gpr_type (str): If ```SingleTask```, a GPR model is trained for each of the r latent dimensions;
                        if ```MultiTask```, one GPR is trained for all the r latend dimensions together.
                        ```SingleTask``` is the only one implemented
    '''

    def __init__(self, X: float, X_species: float, X_flame: float, n_features: int, n_features_species: int, n_features_flame: int, P: float, gpr_type='SingleTask'):
        super().__init__(X, n_features)
        self.X = X
        self.X_species = X_species
        self.X_flame = X_flame                          # 20260612 D
        self.n_features = n_features
        self.n_features_species = n_features_species
        self.n_features_flame = n_features_flame
        self.n_points = X.shape[0] // self.n_features
        self.n_points_species = X_species.shape[0] // self.n_features_species
        self.n_points_flame = X_flame.shape[0] // self.n_features_flame
        self.P = P
        self.gpr_type = gpr_type

        if gpr_type == 'MultiTask': 
            raise NotImplementedError('GPR: ```SingleTask``` is the only mode implemented...')
        if P.shape[0] != X.shape[1]:
            raise Exception(f'The number of parameters ({P.shape[0]}) is different' \
                            f' from the number of columns of X ({X.shape[1]})')
            exit()


    def _train_loop(self, model, likelihood, P0_torch, Vr_torch):
        
        model.train()
        likelihood.train()
    
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        mll = ExactMarginalLogLikelihood(likelihood, model)    # defines loss function for optimization
        
        # initialize training control parameters
        loss_old = 1e10
        e = 1e10
        j = 0

        while (e > self.rel_error) and (j < self.max_iter):
            optimizer.zero_grad()
            output = model(P0_torch)
            loss = -mll(output, Vr_torch)

            # optimization loop
            loss.backward()
            optimizer.step()

            # update control parameters
            e = torch.abs(loss - loss_old).item()
            loss_old = loss
            j += 1

            # for printing specific objects in log file
            if self.verbose == True:
                if self.gpr_type == 'SingleTask':
                    noise_avg = np.mean(model.likelihood.noise.detach().numpy())

                    # to print the lengthscale
                    if hasattr(model.covar_module, ('base_kernel')):
                        if hasattr(model.covar_module.base_kernel, ('kernels')):
                            if len(model.covar_module.base_kernel.kernels) < 1:  
                                ls = model.covar_module.base_kernel.lengthscale.detach()                                    
                            else:
                                ls = []
                                for k, lss in enumerate(model.covar_module.base_kernel.kernels):
                                    ls.append(lss.lengthscale.detach().numpy() if lss.lengthscale is not None else f'no lengthscale for kernel #{k}')
                        else:
                            ls = model.covar_module.lengthscale.detach().numpy()  if model.covar_module.lengthscale is not None else 'no lengthscale for kernel'
                    else:
                        if hasattr(model.covar_module, 'lengthscale'):
                            ls = model.covar_module.lengthscale.detach().numpy()  if model.covar_module.lengthscale is not None else 'no lengthscale for kernel'

                    if self.log is not None:
                        if j % self.max_iter == 200: print(f'   -> Iter {j:d}/{self.max_iter:d} - Loss: {loss.item():.2e} - Mean noise: {noise_avg:.2e} - Lengthscale: {ls}', file=self.log, flush=True)
                    else:
                        if j % self.max_iter == 200: print(f'   -> Iter {j:d}/{self.max_iter:d} - Loss: {loss.item():.2e} - Mean noise: {noise_avg:.2e} - Lengthscale: {ls}')
                else:
                    raise NotImplementedError('GPR._train_loop: ```SingleTask``` is the only mode implemented...')
                
        print(f'   -> Finished at iter {j:d}/{self.max_iter:d} - Loss: {loss.item():.2e} - Mean noise: {noise_avg:.2e} - Lengthscale: {ls}', file=self.log, flush=True)
        
        Vr_sigma = output.stddev.detach().numpy()
        
        return model, likelihood, Vr_sigma

    def scale_data(self, scale_type='std', axis_cnt=1):
        '''
        Return the scaled data matrix.
        
        Args:
            scale_type (str): Type of scaling. Default: ```std```.
                              The list of scaling methods includes
                              ['std', 'none', 'pareto', 'vast', 'range', 'level', 'max', 'variance',
                              'median', 'poisson', 'vast_2', 'vast_3', 'vast_4', 'l2-norm']
            axis_cnt (int): Axis used to compute the centering coefficient. If ``` None```, 
                            the centering coefficient is a scalar. Default: ```1```
        '''
        
        if axis_cnt == 0:
            X_cnt = np.zeros((self.n_features, self.X.shape[1]))
            Xc = np.zeros_like(self.X)
            for i in range(self.n_features):
                for j in range(self.X.shape[1]):
                    x0 = self.X[i*self.n_points:(i+1)*self.n_points, j]
                    X_cnt[i, j] = np.average(x0, axis=axis_cnt)
                    Xc[i*self.n_points:(i+1)*self.n_points, j] = x0 - X_cnt[i, j]
        else:
            X_cnt = np.average(self.X, axis=axis_cnt, keepdims=True)
            Xc = self.X - X_cnt

        # X_cnt = np.average(self.X, axis=axis_cnt, keepdims=True)
        # Xc = self.X - X_cnt
        
        X_scl = np.zeros((self.X.shape[0], 1))
        
        for i in range(self.n_features):
            x0 = Xc[i*self.n_points:(i+1)*self.n_points, :]
            
            if scale_type == 'std':
                X_scl[i*self.n_points:(i+1)*self.n_points, 0] = np.std(x0)
            
            elif scale_type == 'none':
                X_scl[i*self.n_points:(i+1)*self.n_points, 0] = 1.
            
            elif scale_type == 'pareto':
                X_scl[i*self.n_points:(i+1)*self.n_points, 0] = np.sqrt(np.std(x0))
            
            elif scale_type == 'vast':
                scl_factor = np.std(x0)**2/np.average(x0)
                X_scl[i*self.n_points:(i+1)*self.n_points, 0] = scl_factor
            
            elif scale_type == 'range':
                scl_factor = np.max(x0) - np.min(x0)
                X_scl[i*self.n_points:(i+1)*self.n_points, 0] = scl_factor
                
            elif scale_type == 'level':
                X_scl[i*self.n_points:(i+1)*self.n_points, 0] = np.average(x0)
                
            elif scale_type == 'max':
                X_scl[i*self.n_points:(i+1)*self.n_points, 0] = np.max(x0)
            
            elif scale_type == 'variance':
                X_scl[i*self.n_points:(i+1)*self.n_points, 0] = np.var(x0)
            
            elif scale_type == 'median':
                X_scl[i*self.n_points:(i+1)*self.n_points, 0] = np.median(x0)
            
            elif scale_type == 'poisson':
                scl_factor = np.sqrt(np.average(x0))
                X_scl[i*self.n_points:(i+1)*self.n_points, 0] = scl_factor
            
            elif scale_type == 'vast_2':
                scl_factor = (np.std(x0)**2 * kurtosis(x0)**2)/np.average(x0)
                X_scl[i*self.n_points:(i+1)*self.n_points, 0] = scl_factor
            
            elif scale_type == 'vast_3':
                scl_factor = (np.std(x0)**2 * kurtosis(x0)**2)/np.max(x0)
                X_scl[i*self.n_points:(i+1)*self.n_points, 0] = scl_factor
            
            elif scale_type == 'vast_4':
                scl_factor = (np.std(x0)**2 * kurtosis(x0)**2)/(np.max(x0)-np.min(x0))
                X_scl[i*self.n_points:(i+1)*self.n_points, 0] = scl_factor
            
            elif scale_type == 'l2-norm':
                scl_factor = np.linalg.norm(x0)
                X_scl[i*self.n_points:(i+1)*self.n_points, 0] = scl_factor
            
            else:
                raise NotImplementedError('The scaling method selected has not been '\
                                      'implemented yet')
        self.X_cnt = X_cnt
        self.X_scl = X_scl
        
        X_norm = Xc / X_scl
        
        # for chimney species data
        if axis_cnt == 0:
            X_cnt = np.zeros((self.n_features_species, self.X_species.shape[1]))
            Xc = np.zeros_like(self.X_species)
            for i in range(self.n_features_species):
                for j in range(self.X_species.shape[1]):
                    x0 = self.X_species[i*self.n_points_species:(i+1)*self.n_points_species, j]
                    X_cnt[i, j] = np.average(x0, axis=axis_cnt)
                    Xc[i*self.n_points_species:(i+1)*self.n_points_species, j] = x0 - X_cnt[i, j]
        else:
            X_cnt = np.average(self.X_species, axis=axis_cnt, keepdims=True)
            Xc = self.X_species - X_cnt    
        
        X_scl = np.zeros((self.X_species.shape[0], 1))
        
        for i in range(self.n_features_species):
            x0 = Xc[i*self.n_points_species:(i+1)*self.n_points_species, :]
            
            if scale_type == 'std':
                X_scl[i*self.n_points_species:(i+1)*self.n_points_species, 0] = np.std(x0)
            
            elif scale_type == 'none':
                X_scl[i*self.n_points_species:(i+1)*self.n_points_species, 0] = 1.
            
            elif scale_type == 'pareto':
                X_scl[i*self.n_points_species:(i+1)*self.n_points_species, 0] = np.sqrt(np.std(x0))
            
            elif scale_type == 'vast':
                scl_factor = np.std(x0)**2/np.average(x0)
                X_scl[i*self.n_points_species:(i+1)*self.n_points_species, 0] = scl_factor
            
            elif scale_type == 'range':
                scl_factor = np.max(x0) - np.min(x0)
                X_scl[i*self.n_points_species:(i+1)*self.n_points_species, 0] = scl_factor
                
            elif scale_type == 'level':
                X_scl[i*self.n_points_species:(i+1)*self.n_points_species, 0] = np.average(x0)
                
            elif scale_type == 'max':
                X_scl[i*self.n_points_species:(i+1)*self.n_points_species, 0] = np.max(x0)
            
            elif scale_type == 'variance':
                X_scl[i*self.n_points_species:(i+1)*self.n_points_species, 0] = np.var(x0)
            
            elif scale_type == 'median':
                X_scl[i*self.n_points_species:(i+1)*self.n_points_species, 0] = np.median(x0)
            
            elif scale_type == 'poisson':
                scl_factor = np.sqrt(np.average(x0))
                X_scl[i*self.n_points_species:(i+1)*self.n_points_species, 0] = scl_factor
            
            elif scale_type == 'vast_2':
                scl_factor = (np.std(x0)**2 * kurtosis(x0)**2)/np.average(x0)
                X_scl[i*self.n_points_species:(i+1)*self.n_points_species, 0] = scl_factor
            
            elif scale_type == 'vast_3':
                scl_factor = (np.std(x0)**2 * kurtosis(x0)**2)/np.max(x0)
                X_scl[i*self.n_points_species:(i+1)*self.n_points_species, 0] = scl_factor
            
            elif scale_type == 'vast_4':
                scl_factor = (np.std(x0)**2 * kurtosis(x0)**2)/(np.max(x0)-np.min(x0))
                X_scl[i*self.n_points_species:(i+1)*self.n_points_species, 0] = scl_factor
            
            elif scale_type == 'l2-norm':
                scl_factor = np.linalg.norm(x0)
                X_scl[i*self.n_points_species:(i+1)*self.n_points_species, 0] = scl_factor
            
            else:
                raise NotImplementedError('The scaling method selected has not been '\
                                      'implemented yet')
        self.X_cnt_species = X_cnt
        self.X_scl_species = X_scl
        
        X_norm_species = Xc / X_scl


        # for flame species data # 20260612 D
        if axis_cnt == 0:
            X_cnt = np.zeros((self.n_features_flame, self.X_flame.shape[1]))
            Xc = np.zeros_like(self.X_flame)
            for i in range(self.n_features_flame):
                for j in range(self.X_flame.shape[1]):
                    x0 = self.X_flame[i*self.n_points_flame:(i+1)*self.n_points_flame, j]
                    X_cnt[i, j] = np.average(x0, axis=axis_cnt)
                    Xc[i*self.n_points_flame:(i+1)*self.n_points_flame, j] = x0 - X_cnt[i, j]
        else:
            X_cnt = np.average(self.X_flame, axis=axis_cnt, keepdims=True)
            Xc = self.X_flame - X_cnt  
        
        X_scl = np.zeros((self.X_flame.shape[0], 1))
        
        for i in range(self.n_features_flame):
            x0 = Xc[i*self.n_points_flame:(i+1)*self.n_points_flame, :]
            
            if scale_type == 'std':
                X_scl[i*self.n_points_flame:(i+1)*self.n_points_flame, 0] = np.std(x0)
            
            elif scale_type == 'none':
                X_scl[i*self.n_points_flame:(i+1)*self.n_points_flame, 0] = 1.
            
            elif scale_type == 'pareto':
                X_scl[i*self.n_points_flame:(i+1)*self.n_points_flame, 0] = np.sqrt(np.std(x0))
            
            elif scale_type == 'vast':
                scl_factor = np.std(x0)**2/np.average(x0)
                X_scl[i*self.n_points_flame:(i+1)*self.n_points_flame, 0] = scl_factor
            
            elif scale_type == 'range':
                scl_factor = np.max(x0) - np.min(x0)
                X_scl[i*self.n_points_flame:(i+1)*self.n_points_flame, 0] = scl_factor
                
            elif scale_type == 'level':
                X_scl[i*self.n_points_flame:(i+1)*self.n_points_flame, 0] = np.average(x0)
                
            elif scale_type == 'max':
                X_scl[i*self.n_points_flame:(i+1)*self.n_points_flame, 0] = np.max(x0)
            
            elif scale_type == 'variance':
                X_scl[i*self.n_points_flame:(i+1)*self.n_points_flame, 0] = np.var(x0)
            
            elif scale_type == 'median':
                X_scl[i*self.n_points_flame:(i+1)*self.n_points_flame, 0] = np.median(x0)
            
            elif scale_type == 'poisson':
                scl_factor = np.sqrt(np.average(x0))
                X_scl[i*self.n_points_flame:(i+1)*self.n_points_flame, 0] = scl_factor
            
            elif scale_type == 'vast_2':
                scl_factor = (np.std(x0)**2 * kurtosis(x0)**2)/np.average(x0)
                X_scl[i*self.n_points_flame:(i+1)*self.n_points_flame, 0] = scl_factor
            
            elif scale_type == 'vast_3':
                scl_factor = (np.std(x0)**2 * kurtosis(x0)**2)/np.max(x0)
                X_scl[i*self.n_points_flame:(i+1)*self.n_points_flame, 0] = scl_factor
            
            elif scale_type == 'vast_4':
                scl_factor = (np.std(x0)**2 * kurtosis(x0)**2)/(np.max(x0)-np.min(x0))
                X_scl[i*self.n_points_flame:(i+1)*self.n_points_flame, 0] = scl_factor
            
            elif scale_type == 'l2-norm':
                scl_factor = np.linalg.norm(x0)
                X_scl[i*self.n_points_flame:(i+1)*self.n_points_flame, 0] = scl_factor
            
            else:
                raise NotImplementedError('The scaling method selected has not been '\
                                      'implemented yet')
        self.X_cnt_flame = X_cnt
        self.X_scl_flame = X_scl
        
        X_norm_flame = Xc / X_scl

        return X_norm, X_norm_species, X_norm_flame
    
    def scale_GPR_data(self, P: float, scale_type: str):
        '''
        Return the scaled input and target for the GPR model.

        Args:
            P (np.ndarray): Data matrix to scale of size (p, d)
            scale_type (str): Type of scaling.
                              The list of scaling methods includes
                              ['std', 'none', 'pareto', 'vast', 'range', 'level', 'max', 'variance',
                              'median', 'poisson', 'vast_2', 'vast_3', 'vast_4', 'l2-norm']
        '''
        
        P_cnt = np.zeros_like(P)
        P_scl = np.zeros_like(P)
        
        for i in range(P.shape[1]):
            x = P[:,i]
            P_cnt[:,i] = np.mean(x)
            
            if scale_type == 'std':
                P_scl[:,i] = np.std(x)
            
            elif scale_type == 'none':
                P_scl[:,i] = 1.
            
            elif scale_type == 'pareto':
                P_scl[:,i] = np.sqrt(np.std(x))
            
            elif scale_type == 'vast':
                scl_factor = np.std(x)**2/np.average(x)
                P_scl[:,i] = scl_factor
            
            elif scale_type == 'range':
                scl_factor = np.max(x) - np.min(x)
                P_scl[:,i] = scl_factor
                
            elif scale_type == 'level':
                P_scl[:,i] = np.average(x)
                
            elif scale_type == 'max':
                P_scl[:,i] = np.max(x)
            
            elif scale_type == 'variance':
                P_scl[:,i] = np.var(x)
            
            elif scale_type == 'median':
                P_scl[:,i] = np.median(x)
            
            elif scale_type == 'poisson':
                scl_factor = np.sqrt(np.average(x))
                P_scl[:,i] = scl_factor
            
            elif scale_type == 'vast_2':
                scl_factor = (np.std(x)**2 * kurtosis(x, None)**2)/np.average(x)
                P_scl[:,i] = scl_factor
            
            elif scale_type == 'vast_3':
                scl_factor = (np.std(x)**2 * kurtosis(x, None)**2)/np.max(x)
                P_scl[:,i] = scl_factor
            
            elif scale_type == 'vast_4':
                scl_factor = (np.std(x)**2 * kurtosis(x, None)**2)/(np.max(x)-np.min(x))
                P_scl[:,i] = scl_factor
            
            elif scale_type == 'l2-norm':
                scl_factor = np.linalg.norm(x.flatten())
                P_scl[:,i] = scl_factor
            
            else:
                raise NotImplementedError('The scaling method selected has not been '\
                                          'implemented yet')
                    
        self.P_cnt = P_cnt
        self.P_scl = P_scl
        P_norm = (P - P_cnt)/P_scl

        return P_norm    

    def fit(self, scaleX_type='std', scaleP_type='std', axis_cnt=1, 
            select_modes='variance', n_modes=99, verbose=False, basis=None):
        '''
        Fit the GPR model to find the POD coefficients.
    
        Args:
        scaleX_type (str): Type of scaling method for the data matrix. Default: ```std```
        scaleP_type (str): Type of scaling method for the parameters space. Default: ```std```
        axis_cnt (int): Axis used to compute the centering coefficient. If ``` None```, 
                        the centering coefficient is a scalar. Default: ```1```
        select_modes (string): methods of modes selection
        n_modes (int or float): parameter that controls the number of modes to be retained. If 
                                ```variance```, n_modes can be a float between 0 and 100
                                If ```number```, n_modes can be an integer between 1 and m
        verbose (bool): If True, it will print informations on the training of the hyperparameters.
                        Default: ```False```
        
        basis (tuple): If not ```None```, the tuple contains the matrices Ur and Ar 
                       without the need for performing the decomposition. Default: ```None```
            
        '''
        
        self.scaleX_type = scaleX_type
        self.scaleP_type = scaleP_type
        self.select_modes = select_modes
        self.n_modes = n_modes
        self.verbose = verbose

        # perform reduction
        self.X0, self.X0_species, self.X0_flame = self.scale_data(scaleX_type, axis_cnt) # 20260612 D

        if basis is None:
            Ur, Ar, _ = self.decomposition(self.X0, select_modes, n_modes)
            Ur_species, Ar_species, _ = self.decomposition(self.X0_species, select_modes, n_modes)
            Ur_flame, Ar_flame, _ = self.decomposition(self.X0_flame, select_modes, n_modes)         # 20260612 D

        else:
            Ur = basis[0]
            Ar = basis[1]

        self.Ur = Ur
        self.Ar = Ar
        self.r = Ar.shape[1]
        self.d = self.P.shape[1]
        
        self.Ur_species = Ur_species
        self.Ar_species = Ar_species
        self.r_species = Ar_species.shape[1]

        self.Ur_flame = Ur_flame            # 20260612 D
        self.Ar_flame = Ar_flame
        self.r_flame = Ar_flame.shape[1]

        Ar_t = np.concat([self.Ar, self.Ar_species, self.Ar_flame], axis=1)  # 20260612 D
        self.r_t = self.r + self.r_species + self.r_flame

        # Get the singular values and the orthonormal basis
        # (comes from the mathematical formulation of SVD: X = U sigma V^t -> sigma * V^t = A)
        Vr = np.zeros_like(Ar_t)
        Sigma_r = np.zeros((self.r_t,))
        for i in range(self.r_t):
            Sigma_r[i] = np.linalg.norm(Ar_t[:,i])
            Vr[:,i] = Ar_t[:,i]/Sigma_r[i]   # normalise the coefficients to have unit norm
        
        self.Sigma_r = Sigma_r

        # scale input data to the GPR
        P0 = GPR.scale_GPR_data(self, self.P, scaleP_type)
        
        self.P0 = P0
        self.Vr = Vr
    
    def train(self, mean=None, kernel=None, likelihood=None, max_iter=1000, 
              rel_error=1e-5, lr=0.1, verbose=False, log=None):
        '''
        Train the GPR model.
        Return the model and likelihood.

        Args:
            Xspecies (np.ndarray): 2D matrix of size (n_species * n_points, n_cases or n, p), 
                                   with n_cases being the number of operating condition explored
            mean (gpytorch.means): The mean passed to the GPR model. Default: ```means.ConstantMean```
            kernel (gpytorch.kernels): The kernel used for the computation of the covariance matrix. 
                                       Default: ```kernels.MaternKernel```
            likelihood (gpytorch.likelihoods): The likelihood passed to the GPR model. 
                                               If ```SingleTask``` mode -> Default: ```GaussianLikelihood()``` 
            max_iter (int): Maximum number of iterations to train the hyperparameters. Default: ```1000```
            rel_error (float): Minimum relative error below which the training of hyperparameters is
                               stopped. Default: ```1e-5```
            lr (float): Learning rate of the Adam optimizer used for minimizing the negative log 
                        likelihood (being the loss function to optimize). Default: ```0.1```
            verbose (bool): If ```True```, it will print informations on the training of the hyperparameters.
                            Default: ```False```
            log (log file): Log variable where all information are printed. Default: ```None```
        '''
        
        self.max_iter = max_iter
        self.rel_error = rel_error 
        self.lr = lr
        self.verbose = verbose
        self.log = log
        
        print('- Number of modes retained for strip temperature: {}'.format(self.r), file=self.log, flush=True)
        print('- Number of modes retained for chimney temperature & species: {}'.format(self.r_species), file=self.log, flush=True)
        print('- Number of modes retained for flame temperature & species: {}'.format(self.r_flame), file=self.log, flush=True) # 20260612 D
        print('- Number of modes retained for GPR: {}'.format(self.r_t), file=self.log, flush=True)
        print('- Data shape: Ur{} Ar{} self.P{} Ur_species{} Ar_species{} Ur_flame{} Ar_flame{} \n'
              .format(self.Ur.shape, self.Ar.shape, self.P.shape, self.Ur_species.shape, self.Ar_species.shape, self.Ur_flame.shape, self.Ar_flame.shape), file=self.log, flush=True)

        # add the species vectors
        # Vr = self.Vr.copy()
        # Vr0, self.Vr_cnt, self.Vr_scl = scale_data(Vr, 1, Vr.shape[0], axis_cnt=0)   # normalize scores

        # self.Vr = np.concat([self.Vr, Xspecies], axis=1)
        # self.rp = self.Vr.shape[1]

        # convert np.ndarray to torch.tensor
        P0_torch = torch.from_numpy(self.P0).contiguous().double()   # train input
        Vr_torch = torch.from_numpy(self.Vr).contiguous().double()   # train label
        
        models = []
        likelihoods = []

        self.mean = mean
        self.kernel = kernel
        self.likelihood = likelihood

        if self.gpr_type == 'MultiTask':
            raise NotImplementedError('GPR.train: ```SingleTask``` is the only mode implemented...')
        else:
            if mean is None:
                self.mean = gpytorch.means.ConstantMean()
            
            if kernel is None:
                self.kernel = gpytorch.kernels.MaternKernel(2.5)
            
            if likelihood is None:
                self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

            Vr_sigma = np.zeros_like(self.Vr)

            # start training per each mode
            for i in range(self.r_t):
                if self.log is not None: print(f'  * Starting training loop mode(variable) -> {i+1:d}/{self.r_t:d}:', file=self.log, flush=True)

                # copy to avoid corruption
                likelihood = copy.deepcopy(self.likelihood)
                mean = copy.deepcopy(self.mean)
                kernel = copy.deepcopy(self.kernel)

                # kernel = copy.deepcopy(self.kernel) if i == 0 else gpytorch.kernels.MaternKernel(0.5, ard_num_dims=self.P.shape[1])
                kernel = copy.deepcopy(self.kernel) if i == 0 else gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel(ard_num_dims=self.P.shape[1]) + 
                                                                                                gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=self.P.shape[1]))
                # kernel = copy.deepcopy(self.kernel) if i == 0 else gpytorch.kernels.ScaleKernel(gpytorch.kernels.ConstantKernel() + 
                                                                                                # gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=self.P.shape[1]))

                model = ExactGPModel(P0_torch, Vr_torch[:,i], likelihood, mean, kernel)   # extract multivariate distribution
                
                model.double()   # cast to double
                likelihood.double()   # cast to double

                model, likelihood, Vr_sigma[:, i] = self._train_loop(model, likelihood, P0_torch, Vr_torch[:,i])

                models.append(model)
                likelihoods.append(likelihood)

        self.Vr_sigma = Vr_sigma   # prediction uncertainty
        self.models = models
        self.likelihoods = likelihoods

        # print('-> models{} likelihoods{} '.format(self.models, self.likelihoods), file=self.log, flush=True)

        if self.log is not None: print(f' !!! DONE !!!', file=self.log, flush=True)

        return models, likelihoods
    
    def predict(self, P_star: float, problem_dict=None, **kwargs):
        '''
        Return the prediction vector. 
        This method has to be used after fit.

        Args:
            P_star (np.ndarray): Set of design features to evaluate the prediction of size (n_p, d), 
                                 where n_p is the number of design conditions and d is he number of 
                                 design parameters
            problem_dict (dict): Dictonary used to solve the constrained optimization problem. 
                                 If not ```None``` it contains:
                                 - 'problem' (cvxpy optimisation problem), 
                                 - 'mean' (cvxpy mean)
                                 - 'cov' (cvxpy covariance)  
                                 - 'v' (variable of the optimisation problem)
        '''
        
        if not hasattr(self, 'models'):
            raise AttributeError('The function fit has to be called '\
                                  'before calling predict.')
        
        # fix dimensions
        if P_star.ndim < 2:
            P_star = P_star[np.newaxis, :]

        n_p = P_star.shape[0]
        P0_star = np.zeros_like(P_star)

        # scale input data to the model
        for i in range(P_star.shape[1]):
            P0_star[:,i] = (P_star[:,i] - self.P_cnt[0,i]) / self.P_scl[0,i]
        
        # move to torch.tensor
        P0_star_torch = torch.from_numpy(P0_star).contiguous().double()
        
        if self.gpr_type == 'MultiTask':    
            raise NotImplementedError('GPR.predict: ```SingleTask``` is the only mode implemented...')
        else:

            V_pred = np.zeros((n_p, self.r_t))   # actual prediction
            V_sigma = np.zeros((n_p, self.r_t))   # prediction uncertainty

            # for each mode retained
            for i in range(self.r_t):
                
                self.models[i].eval()
                self.likelihoods[i].eval()

                observed_pred = self.likelihoods[i](self.models[i](P0_star_torch))
                V_pred[:,i] = observed_pred.mean.detach().numpy()
                V_sigma[:,i] = observed_pred.stddev.detach().numpy()

        # in SVD X = U*A where A = sigma * Vt^t              
        A_pred = np.zeros_like(V_pred)    # POD coefficient to multlpy the POD mode to get back to physical space
        A_sigma = np.zeros_like(V_sigma)  # uncertainty (standard deviation) over POD coefficient

        # # de-normalization step
        # V_pred[:, :self.r] = (V_pred[:, :self.r] * self.Vr_scl) + self.Vr_cnt
        # V_sigma[:, :self.r] = (V_sigma[:, :self.r] * self.Vr_scl) + self.Vr_cnt

        for i in range(self.r_t):
            A_pred[:,i] = self.Sigma_r[i] * V_pred[:,i]
            A_sigma[:,i] = self.Sigma_r[i] * V_sigma[:,i]
        
        # A_pred[:,self.r:] = V_pred[:,self.r:]
        # A_sigma[:,self.r:] = V_sigma[:,self.r:]
        
        return A_pred, A_sigma
    
    def update(self, P_new: float, A_new: float, A_sigma_new=None, retrain=False, verbose=False):     
        '''
        Updates the model with new data.

        Args:
            P_new (np.ndarray): Set of design features of the new data, size (n_p_new, d), 
                                where n_p_new is the number of new condition to include in the 
                                training set
            A_new (np.ndarray): Set of new data in the low dimensional space, size (n_p_new, r),
                                where r is the reduced dimension (modes retained)
            A_sigma_new (np.ndarray): Uncertainty of the new data. Default: ```None```
            retrain (bool): If ```True``` the hyperparameters are retrained. Default: ```False```
            verbose (bool): If True, it will print informations on the training of the hyperparameters.
                            Default: ```False```
        '''
        
        self.verbose = verbose
        
        # create new set of parameters (scale first)
        P0_new = np.zeros_like(P_new)
        for i in range(P_new.shape[1]):
            P0_new[:,i] = (P_new[:,i] - self.P_cnt[0,i]) / self.P_scl[0,i]
            
        # chain with the old data and move to torch.tensor
        P0_tot = np.concatenate([self.P0, P0_new], axis=0)
        P0_tot_torch = torch.from_numpy(P0_tot).contiguous().double()
        
        # create new set of observations
        Vr_new = np.zeros_like(A_new)
        for i in range(self.r):
            # V_sigma_train[:,i] = A_sigma_train[:,i]/self.Sigma_r[i]
            Vr_new[:,i] = A_new[:,i]/self.Sigma_r[i]
        
        # chain with the old data and move to torch.tensor
        Vr_tot = np.concatenate([self.Vr, Vr_new], axis=0)
        Vr_tot_torch = torch.from_numpy(Vr_tot).contiguous().double()
        
        # if the uncertainty is passed, create new set of uncertainties
        if A_sigma_new is not None:
            Vr_sigma_new = np.zeros_like(A_sigma_new)
            for i in range(self.r):
                Vr_sigma_new[:,i] = A_sigma_new[:,i]/self.Sigma_r[i]

            Vr_sigma_tot = np.concatenate([self.Vr_sigma, Vr_sigma_new], axis=0)
            Vr_sigma_tot_torch = torch.from_numpy(Vr_sigma_tot).contiguous().double()
            self.Vr_sigma = np.zeros_like(Vr_sigma_tot)
    
        if self.gpr_type == 'MultiTask':
            raise NotImplementedError('GPR.update: ```SingleTask``` is the only mode implemented...')
        else:
            # for each mode
            for i in range(self.r):

                self.models[i].set_train_data(P0_tot_torch, Vr_tot_torch[:,i], strict=False)
                
                if retrain:
                    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(Vr_sigma_tot_torch[:, i]**2)
                    self.models[i].likelihood = likelihood
                    
                    temp = self._train_loop(self.models[i], self.likelihoods[i], P0_tot_torch, Vr_tot_torch[:,i], i)
                    self.models[i], self.likelihoods[i], self.Vr_sigma[:,i] = temp
    
    def save(self, path: str):
        '''
        Saves the trained model for future uses.
        
        Args:
            path (str): Path pointing to the file where the file needs to to be saved
        '''

        checkpoint = {
            # model hyperparameters
            'model_state_dicts': [m.state_dict() for m in self.models],
            'likelihood_state_dicts': [l.state_dict() for l in self.likelihoods],
            # training data needed to reconstruct the models
            'P0': self.P0,
            'Vr': self.Vr,
            # POD basis
            'Ur': self.Ur,
            'Ar': self.Ar,
            'Sigma_r': self.Sigma_r,
            # scaling parameters needed for predict()
            'P_cnt': self.P_cnt,
            'P_scl': self.P_scl,
            'X_cnt': self.X_cnt,
            'X_scl': self.X_scl,
            # metadata
            'r': self.r,
            'd': self.d,
            'gpr_type': self.gpr_type,
            'mean': self.mean,
            'kernel': self.kernel,
            'likelihood': self.likelihood,
        }

        torch.save(checkpoint, path)

    def load(self, path: str):
        '''
        Loads the models and likelihoods of trained/frozen GPRs.

        Args:
            path (str): Path pointing to the file where the file has been saved saved
        '''

        checkpoint = torch.load(path, weights_only=False)

        self.P0 = checkpoint['P0']
        self.Vr = checkpoint['Vr']
        self.Ur = checkpoint['Ur']
        self.Ar = checkpoint['Ar']
        self.Sigma_r = checkpoint['Sigma_r']
        self.P_cnt = checkpoint['P_cnt']
        self.P_scl = checkpoint['P_scl']
        self.X_cnt = checkpoint['X_cnt']
        self.X_scl = checkpoint['X_scl']
        self.r = checkpoint['r']
        self.d = checkpoint['d']
        self.gpr_type = checkpoint['gpr_type']
        self.mean = checkpoint['mean']
        self.kernel = checkpoint['kernel']
        self.likelihood = checkpoint['likelihood']

        # reconstruct each model and load its weights
        P0_torch = torch.from_numpy(self.P0).double()
        Vr_torch = torch.from_numpy(self.Vr).double()

        self.models = []
        self.likelihoods = []

        # for each mode
        for i in range(self.r):
            likelihood = copy.deepcopy(self.likelihood)
            mean = copy.deepcopy(self.mean)
            kernel = copy.deepcopy(self.kernel)

            model = ExactGPModel(P0_torch, Vr_torch[:, i], likelihood, mean, kernel)
            
            model.double()
            likelihood.double()

            model.load_state_dict(checkpoint['model_state_dicts'][i])
            likelihood.load_state_dict(checkpoint['likelihood_state_dicts'][i])

            self.models.append(model)
            self.likelihoods.append(likelihood)

    @classmethod
    def from_checkpoint(cls, path: str):
        '''
        Loads the models and likelihoods of trained/frozen GPRs from a checkpoint file without initiating the class.

        Args:
            path (str): Path pointing to the file where the file has been saved saved
        '''

        # create the object without calling __init__
        gpr = cls.__new__(cls)
        gpr.load(path)
        return gpr