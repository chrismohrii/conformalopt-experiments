import numpy as np
from tqdm import tqdm
import json

def sim_bm(paths, points, mu=0.0, sigma=1.0, start=0.0, end=1.0, seed=None):
    """
    Simulates Brownian motion.

    Arguments:
        paths (int): Number of paths to generate.
        points (int): Number of points per path.
        mu (float, optional): Mean of Brownian motion process.
        sigma (float, optional): Standard deviation of Brownian motion process.
        start (float, optional): Initial time of simulation.
        end (float, optional): Final time of simulation.
        seed (int, optional): Random seed argument for reproducibility of random generation.

    Returns:
        W (np.array, (paths, points)): Matrix of simulated Brownian motion paths.
        t_axis (np.array, (points,)): Array of time indices.
    """

    # Set seed if available
    if seed:
        rng = np.random.default_rng(seed)
    else:
         rng = np.random.default_rng()

    # Generate random normal variables
    Z = rng.normal(mu, sigma, (paths, points))
    # Compute time indices
    t_axis = np.linspace(start, end, points)
    # Normalize and obtain cumulative sum
    dt = t_axis[1]-t_axis[0]
    dZ = np.sqrt(dt) * Z
    W = np.cumsum(dZ,axis=1)
    return W, t_axis

def integrate_bm(W,t_axis):
    '''
    Approximates the integral in the denominator of U in Lobato (2001)

    Arguments:
        W (np.array, (paths, points)): Matrix of simulated Brownian motion paths.
        t_axis (np.array, (points,)): Array of time indices.

    Returns:
        F_int (np.array, (paths,)): Array of integrated denominator terms in Lobato (2001).
    '''
    # Compute integrand
    F = np.square(W - t_axis.reshape(1,-1)*W[:,-1].reshape(-1,1))
    # Perform numerical integration
    F_int = np.trapz(F,t_axis,axis=1)
    return F_int

def compute_u(W,t_axis):
    '''
    Computes simulated value of U as described in Lobato (2001)

    Arguments:
        W (np.array, (paths, points)): Matrix of simulated Brownian motion paths.
        t_axis (np.array, (points,)): Array of time indices.

    Returns:
        U (np.array, (paths,)): Array of simulated U values.
    '''
    F_int = integrate_bm(W,t_axis)
    U = np.square(W[:,-1])/F_int
    return U

def compute_u_half(W,t_axis):
    '''
    Computes simulated value of U_half as described in Lobato (2001).

    Arguments:
        W (np.array, (paths, points)): Matrix of simulated Brownian motion paths.
        t_axis (np.array, (points,)): Array of time indices.

    Returns:
        U_half (np.array, (paths,)): Array of simulated U_half values.
    '''
    F_int = np.sqrt(integrate_bm(W,t_axis))
    U_half = W[:,-1]/F_int
    return U_half

def generate_u_samples(filename, paths=1000, points=10000, chunks=10000):
    '''
    Generates samples of U as described in Lobato (2001) for estimation of its distribution.

    Arguments:
        filename (str): Path to save simulated U values.
        paths (int): Number of paths to simulate per chunk.
        points (int): Number of points per path.
        chunks (int): Number of total chunks to simulate.

    Returns:
        
    '''
    U = np.zeros((chunks,paths))
    # Obtain simulated U values for each chunk
    for c in tqdm(range(chunks)):
        W, t_axis = sim_bm(paths,points)
        U[c,:] = compute_u(W,t_axis)

    U = U.reshape(-1,1)
    np.savez_compressed(filename, data_array=U)

def generate_u_half_samples(filename, paths=1000, points=10000, chunks=10000):
    '''
    Generates samples of U_half as described in Lobato (2001) for estimation of its distribution.

    Arguments:
        filename (str): Path to save simulated U_half values.
        paths (int): Number of paths to simulate per chunk.
        points (int): Number of points per path.
        chunks (int): Number of total chunks to simulate.

    Returns:
        
    '''
    U_half = np.zeros((chunks,paths))
    # Obtain simulated U_half values for each chunk
    for c in tqdm(range(chunks)):
        W, t_axis = sim_bm(paths,points)
        U_half[c,:] = compute_u_half(W,t_axis)

    U_half = U_half.reshape(-1,1)
    np.savez_compressed(filename, data_array=U_half)

def compute_p_val_two_sided(stat, filename="hypothesis_testing_data/U_data.npz"):
    '''
    Computes two-sided p-val described in Lobato (2001) using data from a precomputed file for CDF estimation.

    Arguments:
        stat (float): Value of T-stat in Lobato (2001).
        filename (str): Path to simulated U values.

    Returns:
        p (float): Estimated two-sided p-value.
    '''
    npz = np.load(filename)
    U = npz['data_array']
    p = np.mean(U >= stat)
    return p

def compute_p_val_one_sided(stat, filename="hypothesis_testing_data/U_half_data.npz", mode="greater"):
    '''
    Computes one sided p-val described in Lobato (2001) using data from a precomputed file for CDF estimation.

    Arguments:
        stat (float): Value of T-half-stat in Lobato (2001).
        filename (str): Path to simulated U_half values.

    Returns:
        p (float): Estimated one-sided p-value.
    '''
    npz = np.load(filename)
    U_half = npz['data_array']
    if mode == "greater":
        p = np.mean(U_half >= stat)
    else:
        p = np.mean(U_half <= stat)
    return p

def two_sided_h_test(diff):
    '''
    Performs two-sided hypothesis test described in Lobato (2001) using data from a precomputed file for CDF estimation.

    Arguments:
        diff (np.array): Difference sequence or time-series data.

    Returns:
        stat (float): alue of T-stat in Lobato (2001).
        p (float): Estimated two-sided p-value.
    '''
    # Check if sequence is constant, if so compare its constant value to the appropriate hypothesis
    if np.all(diff == diff[0]):
        return np.nan, float(diff[0]==0)
    N = len(diff)
    m = np.mean(diff)
    s = np.mean(1/N*np.square(np.cumsum(diff-m)))
    stat = N*np.square(m)/s
    return stat, compute_p_val_two_sided(stat)
    
def one_sided_h_test(diff, mode="greater"):
    '''
    Performs one-sided hypothesis test described in Lobato (2001) using data from a precomputed file for CDF estimation.

    Arguments:
        diff (np.array): Difference sequence or time-series data.
        mode (str, optional): Type of one-sided hypothesis to test.

    Returns:
        stat (float): Value of T-half-stat in Lobato (2001).
        p (float): Estimated one-sided p-value.
    '''
    # Check if sequence is constant, if so compare its constant value to the appropriate hypothesis
    if np.all(diff == diff[0]):
        if mode == "greater":
            return np.nan, float(diff[0] <= 0)
        else:
            return np.nan, float(diff[0] >= 0)
    N = len(diff)
    m = np.mean(diff)
    s = np.mean(1/N*np.square(np.cumsum(diff-m)))
    stat = np.sqrt(N)*m/np.sqrt(s)
    p = compute_p_val_one_sided(stat, mode=mode)
    return stat, p

def confidence_interval(diff, alpha=0.1, filename="hypothesis_testing_data/U_data.npz"):
    '''
    Constructs CIs as described in Shao (2010) using data from a precomputed file for CDF estimation.

    Arguments:
        diff (np.array): Difference sequence or time-series data.
        filename (str): Path to simulated U values.

    Returns:
        ci (np.array, (2,)): Confidence intervals for the mean as descirbed in Shao (2010).
    '''
    # Compute stat
    N = len(diff)
    m = np.mean(diff)
    s = np.mean(1/N*np.square(np.cumsum(diff-m)))
    # Load U vals and compute quantile
    npz = np.load(filename)
    U = npz['data_array']
    q = np.quantile(U,1-alpha)
    # Return confidence interval
    dev = np.sqrt(s*q/N)
    lb = m - dev
    ub = m + dev
    ci = np.array([lb, ub])
    return ci

# Test methods
if __name__ == "__main__":
    # Get two sets of predictions
    experiment_name = 'preregistered_w_val'
    START_TIME = '2024-12-1'
    END_TIME = '2024-12-9'

    data_name = f'ercot_{START_TIME}_{END_TIME}'
        
    # Read results from the JSON file
    with open(f'./{experiment_name}/results/{data_name}.json', 'r') as json_file:
        methods = json.load(json_file)

    predictions_ours =  np.array(methods['linear_qt_fixed']['predictions'])
    predictions_baseline = np.array(methods['conformal_pi']['predictions'])

    diff_sequence = predictions_baseline - predictions_ours

    print(two_sided_h_test(diff_sequence))