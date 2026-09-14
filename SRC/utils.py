import torch
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import json
from scipy.stats import kurtosis

def read_param(dictionary: dict, file_name: str=None):
        """
        Read config file

        Args:
            dictionary (dict): contains configuration variables
            file_name (string): path to the json file containing the values to read
        """

        try:
            with open(file_name, "r") as file:
                file_json = json.load(file)

                # Update dictionary
                for key, value in file_json.items():
                    if key in dictionary:
                        dictionary[key] = value

        except FileNotFoundError:
            print(f"!!! {file_name} not found !!!")
        except json.JSONDecodeError as e:
            print(f"Error in file format {e} !")

def read_simulation(file_path: str, data_path: str, log=None, verbose=True, species=False):
    """
        Function to read the data from the simulations.
        
        Args:
            file_path (str): Path to the folder containing the simulations to process
            data_path (str): Folder where the simulation data are stored
            log (str): Path to the file containing the log of the model training.
            verbose (bool): If ```False``` no printout is added to the log file. Default: ```True```
    """
    
    path = os.path.join(file_path, data_path)
    files = [f for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) and '.csv' in f)]
    files.sort()

    if log is None: 
        name_logfile = os.path.join(os.getcwd(), 'log-gpr.txt')
        open(name_logfile, "w").close()   # in case there is already one
        log = open(name_logfile, "a")

    columns = []
    for i, file in enumerate(files):
        if verbose: print('  -> {}'.format(file), file=log, flush=True)
        sim = pd.read_csv(os.path.join(path, file), sep=r"\s+", skipinitialspace=True)
        if i == 0: 
            varsName = sim.columns.to_list()
            # vars = [var for var in varsName if 'cell' not in var and 'coordinate' not in var] if not species else ['temperature', 'o2', 'co2', 'h2o']
            vars = [var for var in varsName if 'temperature' in var] if not species else ['temperature', 'o2', 'co2', 'h2o'] # 20260612 D

        n_3D = sim.shape[0] * len(vars)
        n_cells = sim.shape[0]

        column = np.empty((n_3D,))
        for j, var in enumerate(vars):
            column[j*n_cells:(j+1)*n_cells] = sim.loc[:, var]

        columns.append(column)

    return np.vstack(columns), vars, n_cells

def scale_data(X: float, n_features: int, n_points: int, scale_type='std', axis_cnt=1):
        '''
        Return the scaled data matrix.
        
        Args:
            X (np.ndarray): 2D matrix of size (n_feature * n_points, n_cases or n, p), 
                            with n_cases being the number of operating condition explored
            n_features (int): number of features in the dataset (temperature, velocity, etc.)
            n_points (int): number of points per feature
            scale_type (str): Type of scaling. Default: ```std```.
                              The list of scaling methods includes
                              ['std', 'none', 'pareto', 'vast', 'range', 'level', 'max', 'variance',
                              'median', 'poisson', 'vast_2', 'vast_3', 'vast_4', 'l2-norm']
            axis_cnt (int): Axis used to compute the centering coefficient. If ``` None```, 
                            the centering coefficient is a scalar. Default: ```1```
        '''

        if axis_cnt == 0:
            X_cnt = np.zeros((n_features, X.shape[1]))
            Xc = np.zeros_like(X)
            for i in range(n_features):
                for j in range(X.shape[1]):
                    x0 = X[i*n_points:(i+1)*n_points, j]
                    X_cnt[i, j] = np.average(x0, axis=axis_cnt)
                    Xc[i*n_points:(i+1)*n_points, j] = x0 - X_cnt[i, j]
        else:
            X_cnt = np.average(X, axis=axis_cnt, keepdims=True)
            Xc = X - X_cnt
        
        # X_cnt = np.average(X, axis=axis_cnt, keepdims=True)
        # Xc = X - X_cnt        
        
        X_scl = np.zeros((X.shape[0], 1))
        
        for i in range(n_features):
            x0 = Xc[i*n_points:(i+1)*n_points, :]
            
            if scale_type == 'std':
                X_scl[i*n_points:(i+1)*n_points, 0] = np.std(x0)
            
            elif scale_type == 'none':
                X_scl[i*n_points:(i+1)*n_points, 0] = 1.
            
            elif scale_type == 'pareto':
                X_scl[i*n_points:(i+1)*n_points, 0] = np.sqrt(np.std(x0))
            
            elif scale_type == 'vast':
                scl_factor = np.std(x0)**2/np.average(x0)
                X_scl[i*n_points:(i+1)*n_points, 0] = scl_factor
            
            elif scale_type == 'range':
                scl_factor = np.max(x0) - np.min(x0)
                X_scl[i*n_points:(i+1)*n_points, 0] = scl_factor
                
            elif scale_type == 'level':
                X_scl[i*n_points:(i+1)*n_points, 0] = np.average(x0)
                
            elif scale_type == 'max':
                X_scl[i*n_points:(i+1)*n_points, 0] = np.max(x0)
            
            elif scale_type == 'variance':
                X_scl[i*n_points:(i+1)*n_points, 0] = np.var(x0)
            
            elif scale_type == 'median':
                X_scl[i*n_points:(i+1)*n_points, 0] = np.median(x0)
            
            elif scale_type == 'poisson':
                scl_factor = np.sqrt(np.average(x0))
                X_scl[i*n_points:(i+1)*n_points, 0] = scl_factor
            
            elif scale_type == 'vast_2':
                scl_factor = (np.std(x0)**2 * kurtosis(x0)**2)/np.average(x0)
                X_scl[i*n_points:(i+1)*n_points, 0] = scl_factor
            
            elif scale_type == 'vast_3':
                scl_factor = (np.std(x0)**2 * kurtosis(x0)**2)/np.max(x0)
                X_scl[i*n_points:(i+1)*n_points, 0] = scl_factor
            
            elif scale_type == 'vast_4':
                scl_factor = (np.std(x0)**2 * kurtosis(x0)**2)/(np.max(x0)-np.min(x0))
                X_scl[i*n_points:(i+1)*n_points, 0] = scl_factor
            
            elif scale_type == 'l2-norm':
                scl_factor = np.linalg.norm(x0)
                X_scl[i*n_points:(i+1)*n_points, 0] = scl_factor
            
            else:
                raise NotImplementedError('The scaling method selected has not been '\
                                      'implemented yet')
                    
        X_norm = Xc / X_scl
        
        return X_norm, X_cnt, X_scl

def error_evaluation(X, Xp, mean ,idx):
    """
        Function to evaluate error on prediction.
        
        Args:
            X (Tensor[double]): Tensor containing the reference data
            Xp (Tensor[double]): Tensor containing the prediction to the GPR model in the physical space
            mean (np.ndarray): Array containing mean values per feature
            idx (int): Index of the variable you're interested in evaluating the errors
    """

    diff = X - Xp
    data_l2_norm = torch.norm(X, p=2)

    # normalized squared error
    diff_norm = torch.norm(diff, p=2)
    norm_error =  torch.square(diff_norm)/torch.square(data_l2_norm)

    # reconstruction error
    diff_abs = torch.abs(X - Xp)
    recon_error = torch.mul(diff_abs, (data_l2_norm)**-1)

    # R2 (determination coefficient)
    if mean is not None:
        # diff_mean = X - torch.from_numpy(mean[:, idx]).mean()
        diff_mean = X - mean[idx]
        r_square = 1 - (torch.sum(torch.square(diff))/torch.sum(torch.square(diff_mean)))

    return [norm_error, recon_error, r_square] if mean is not None else [norm_error, recon_error]
    
def plot_comparison(zs, feature, im_shape, idx, cmap='viridis', order='C', show_fig=False, path=None, train=True):
    
    if len(im_shape) == 3:
        im_shape = [im_shape[0], im_shape[2]]

    n_cases = len(zs)
    cases = [r'$\mathbf{Obs.}$', r'$\mathbf{Pred.}$']  
    
    vmin = np.min([zs[0].min(), zs[1].min()])
    vmax = np.max([zs[0].max(), zs[1].max()])

    fig, axs = plt.subplots(ncols=(n_cases), figsize=(5.4, 6))

    axis_labels = ['x', 'y', 'z']
    fig.subplots_adjust(bottom=0., top=1., left=0, right=.925, wspace=0.0, hspace=0.05)

    for i, ax in enumerate(axs):    
        cs = ax.imshow(zs[i].reshape(im_shape, order=order)[::-1], origin='lower', interpolation='gaussian',
                       extent=[0, 0.35, 0, 0.7], cmap=cmap, vmin=vmin, vmax=vmax)
        
        # ax.set_aspect('equal')
        ax.set_xlabel(rf'${axis_labels[0]}-coordinate$', fontsize=15)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(cases[i], fontsize=17)
        if i > 0:
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        else:
            ax.set_ylabel(rf'${axis_labels[2]}-coordinate$', fontsize=15)
            ax.invert_xaxis()
    
    ax_bounds = axs[1].get_position().bounds
    cb_ax = fig.add_axes([0.95, ax_bounds[1], 0.025, ax_bounds[3]])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), 
                        cax=cb_ax, orientation='vertical')
    cbar.set_label(rf'$\boldsymbol{{{feature}}}$', fontsize=15)

    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([r'$\mathbf{min}$', r'$\mathbf{max}$'], fontsize=15)

    # fmt = mpl.ticker.ScalarFormatter(useMathText=True)
    # cbar.formatter.set_powerlimits((0, 4))

    cb_ax.yaxis.set_offset_position('left')
    
    if show_fig: plt.show()
    
    # saving image
    if path is not None:
        pathImg = os.path.join(path, 'img')
        os.makedirs(pathImg, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(pathImg, '{}-train#{}.png'.format(feature, idx)) if train else os.path.join(pathImg, '{}-test#{}.png'.format(feature, idx)))

def extract_outlet_temperature(X):

    region = {
        'temperature': X[0][:601],
        'uncertainty': X[1][:601]
    }

    return region['temperature'].mean(), region['uncertainty'].mean()

########################### REDUCED ORDER MODELING #############################
class ROM:
    def __init__(self, data: float, select_modes='variance', n_modes=99, eigendec=False):
        """
        PCA class for reducing dimensionality.
        
        Args:
            data (Tensor[float]): 2D array of (n_samples, n_features/n_cases), where every row is a sample
            select_modes (string): methods of modes selection
            n_modes (int or float): parameter that controls the number of modes to be retained. 
                                    If ```variance```, n_modes can be a float between 0 and 100
                                    If ```number```, n_modes can be an integer between 1 and m
            eigendec (bool): choice of reduction method. If ```True``` standard PCA is performed, 
                             if ```False```spectral decomposition (SVD) is performed instead
        """

        self.data = np.array(data)
        self.select_modes = select_modes
        self.n_modes = n_modes
        self.eigendec = eigendec

    def reduction(self, U: float, A: float, exp_variance: float):
        """
        Return the reduced taylored basis.

        Args:
            U (numpy.ndarray): the taylored basis to be reduced, size (n,p)
            A (numpy.ndarray): the coefficient matrix to be reduced, size (p,p)
            exp_variance (numpy.ndarray): the array containing the explained variance of the modes, size (p,)
        """

        if self.select_modes == 'variance':
            if not 0 <= self.n_modes <= 100: 
                raise ValueError('The parameter n_modes is outside the[0-100] range.')
                
            # The r-order truncation is selected based on the amount of variance recovered
            if self.n_modes == 100:
                r = A.shape[1]
            else:
                r = 1
                while exp_variance[r-1] < self.n_modes:
                    r += 1
    
        elif self.select_modes == 'number':
            if not type(self.n_modes) is int:
                raise TypeError('The parameter n_modes is not an integer.')
            if not 1 <= self.n_modes <= U.shape[1]: 
                raise ValueError('The parameter n_modes is outside the [1-m] range.')
            r = self.n_modes
        else:
            raise ValueError('The select_mode value is wrong.')

        # Reduce the dimensionality
        Ur = U[:, :r]
        Ar = A[:, :r]

        return Ur, Ar

    def reconstruct(self, Ar:float, modes=None):
        """
        Reconstruct the X matrix from the low-dimensional representation.

        Args:
        Ar (numpy.ndarray): the matrix containing the low-dimensional coefficients, size (n_p,r)
        sampling (numpy.ndarray): matrix used to sample part of the reconstruction, size (s, n)
                                  Default:```None```, meaning that the entire field is reconstructed
        """
        
        if modes is None:
            modes = self.modes
        
        if Ar.ndim < 2:
                Ar = Ar[np.newaxis, :]
        
        if self.eigendec:
            X_rec = Ar @ modes.T
        else:
            X_rec = modes @ Ar.T

        return X_rec
    
    def decomposition(self, data=None, select_modes=None, n_modes=None):
        """
        Return the taylored basis, scores of the decomposition (time dependent coefficient of the modes for SVD
        or data matrix projection onto the new basis for PCA), and the amount of variance of the modes.
        """

        if data is None:
            data = self.data
        if select_modes is None:
            select_modes = self.select_modes
        if n_modes is None:
            n_modes = self.n_modes

        # update if changes occur compared to initialization
        if select_modes and n_modes is not None:
            self.select_modes = select_modes
            self.n_modes = n_modes

        if self.eigendec:
            C = np.cov(data, rowvar=False)   #rowvar=False because the data matrix is (samples x variables)

            ALLevals, ALLevecs = np.linalg.eig(C)
            
            # Order eigenvalue in decreasing order of magnitude
            mask = np.argsort(ALLevals)[::-1]
            evecs = ALLevecs[:,mask]
            evals = ALLevals[mask]
            scores = self.data @ evecs # Get scores
            exp_variance = 100*np.cumsum(evals)/np.sum(evals)
            evecsr, scores = self.reduction(evecs, scores, exp_variance)

            self.modes = evecsr
            self.evals = evals
            self.exp_variance = exp_variance
            r = scores.shape[1]

        else:
            # Compute the SVD of the scaled dataset
            U, S, Vt = np.linalg.svd(data, full_matrices=False)
            A = np.matmul(np.diag(S), Vt).T
            L = S**2    # Compute the eigenvalues
            exp_variance = 100*np.cumsum(L)/np.sum(L)
            Ur, Ar = self.reduction(U, A, exp_variance)
            
            self.modes = Ur
            self.evals = L
            self.exp_variance = exp_variance
            r = Ar.shape[1]

        return Ur if not self.eigendec else evecsr, Ar if not self.eigendec else scores, exp_variance[:r] 

    def plot_explained_variance(self, exp_variance=None, workdir=None):
        """
        Show the varaince explained by each principal component.

        exp_variance (numpy.ndarray): the array containing the explained variance of the modes, size (p,). Default:```None```
        workdir (str): path to the folder where the img will be saved. Default:```None```
        """
        
        if exp_variance is None: exp_variance = self.exp_variance

        plt.figure(figsize=(8, 5))
        plt.bar(range(1, len(exp_variance) + 1), exp_variance, width=0.4, alpha=0.5, align='center', label='Individual Variance')
        plt.plot(range(1, len(exp_variance) + 1), exp_variance, marker='.', linestyle='--', label='Cumulative Variance')
        plt.xlabel('Principal Component Index')
        plt.ylabel('Explained Variance Ratio')
        plt.xticks(ticks=range(1, len(exp_variance)+1, 20), labels=[f"{i}" for i in range(1, len(exp_variance)+1, 20)])
        plt.axhline(y=99, color='r', linestyle='-')
        plt.title('Explained Variance by Principal Components')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        if workdir is not None: plt.savefig(os.path.join(workdir, 'exp_var.png'))
        plt.show()

    def plot_loading_scores(self, var_names: list, component_idx=1, workdir=None):
        """
        Plot the loading scores for a given principal component.

        Args:
            var_names (list[string]): list of variable names
            component_idx (int): index of the principal component to visualize
            workdir (str): path to the folder where the img will be saved. Default:```None```
        """

        if self.modes is None: raise ValueError('Decomposition must be performed before plotting loading scores!')
        top_n = self.data.shape[-1]
        var_names = list(var_names)

        # loadings = modes * singularValue
        loading_scores = self.modes[:, component_idx - 1] * np.sqrt(self.evals[component_idx - 1])

        plt.figure(figsize=(10, 5))
        plt.bar(range(top_n), loading_scores[range(top_n)], align='center')
        plt.xticks(range(top_n), [f'{i}' for i in var_names], rotation=45)
        plt.title(f'Loading Scores for Principal Component #{component_idx}')
        plt.ylabel('Contribution')
        # plt.grid(True)
        plt.tight_layout()
        if workdir is not None: plt.savefig(os.path.join(workdir, f'loading_scores_PC{component_idx}.png'))
        plt.show()
