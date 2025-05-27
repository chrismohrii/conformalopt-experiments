import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker
from tqdm import tqdm
import cvxpy as cp
from statsmodels.tsa.forecasting.theta import ThetaModel
import numpy as np


BIAS_TERM_QT_ON_AR = 1 # doesn't matter.

# From conformal PID paper. Used for ar scorecaster.
def fit_ar_model(Y,p,X=None):
    T = Y.shape[0]
    M = np.zeros((T-p,p))
    Yflip = np.flip(Y)
    for i in range(0,T-p):
        M[i,:] = Yflip[i+1:i+1+p]
    M = np.flip(M, axis=0)
    M = np.flip(M, axis=1)
    if X is None:
        betahat = np.flip(np.linalg.pinv(M)@Y[p:])
        return betahat
    else:
        d = X.shape[1]
        M = np.concatenate([M,X[p:]],axis=1)
        out = np.flip(np.linalg.pinv(M)@Y[p:])
        betahat = out[d:]
        thetahat = out[:d]
        return betahat, thetahat

# From conformal PID paper. Used for ar scorecaster.
def predict_ar_model(Y_test, betahat, thetahat=None, X_test=None):
    T_test = Y_test.shape[0]
    p = betahat.shape[0]
    if (thetahat is None) or (X_test is None):
        thetahat = np.zeros((1,))
        X_test = np.zeros((T_test,1))
    Yhat = np.zeros((T_test,))
    #for i in range(p,T_test):

    return betahat.dot(np.flip(Y_test)) #+ thetahat.dot(X_test[i])
    #return Yhat

def get_scalar_qt_predictions(scores, q_1, etas, alpha):
    """
    Computes scalar quantile tracking predictions.
    :param scores: (np.array) The scores.
    :param q_1: (float) The quantile to compute.
    :param eta: (np.array) The sequence of learcning rates.
    :return: (float) The sequence of online quantiles.
    """
    T = scores.shape[0]
    print(T)
    q = np.zeros(T)
    q[0] = q_1
    for t in range(T):
        err_t = (scores[t] > q[t]).astype(int)
        if t < T - 1:
            q[t + 1] = q[t] - etas[t] * (alpha - err_t)
    return q

# Parameterized quantile tracking.
def get_linear_qt_predictions(scores, q_1, etas, alpha, p_order, bias):
    """
    Computes linear quantile tracking predictions.
    :param scores: (np.array) The scores.
    :param q_1: (float) The quantile to compute.
    :param eta: (np.array) The sequence of learcning rates.
    :return: (float) The sequence of online quantiles.
    """
    T = scores.shape[0]
    q = np.zeros(T)
    # Implement as theta
    theta = np.zeros((T, p_order+1))
    phi = np.zeros((T, p_order+1))
    theta[p_order] = np.ones(p_order+1) * (1/p_order) if p_order != 0 else np.ones(p_order+1) *q_1 # 0.05 heuristically chosen to be on the scale of the scores.
    for t in range(p_order, T):
        phi[t] = np.concatenate((scores[t-p_order:t], [bias]))
        q[t] = theta[t].T @ phi[t]
        err_t = (scores[t] > q[t]).astype(int)
        if t < T - 1:
            theta[t + 1] = theta[t] + etas[t] * (err_t - alpha) * phi[t]

    #print(f'Final theta: {theta[-1]}')

    # First p predictions are empty
    return q

# Parameterized quantile tracking.
def get_linear_qt_thetas(scores, q_1, etas, alpha, p_order, bias):
    """
    Computes linear quantile tracking predictions.
    :param scores: (np.array) The scores.
    :param q_1: (float) The quantile to compute.
    :param eta: (np.array) The sequence of learcning rates.
    :return: (float) The sequence of online quantiles.
    """
    T = scores.shape[0]
    q = np.zeros(T)
    # Implement as theta
    theta = np.zeros((T, p_order+1))
    phi = np.zeros((T, p_order+1))
    theta[p_order] = np.ones(p_order+1) * (1/p_order) if p_order != 0 else np.ones(p_order+1) *q_1 # 0.05 heuristically chosen to be on the scale of the scores.
    for t in range(p_order, T):
        phi[t] = np.concatenate((scores[t-p_order:t], [bias]))
        q[t] = theta[t].T @ phi[t]
        err_t = (scores[t] > q[t]).astype(int)
        if t < T - 1:
            theta[t + 1] = theta[t] + etas[t] * (err_t - alpha) * phi[t]

    #print(f'Final theta: {theta[-1]}')
    return theta


# Parameterized quantile tracking. Implementation that involves batching.
def get_linear_batched_qt_predictions(scores, q_1, etas, alpha, p_order, bias, batch_size=1):
  T = scores.shape[0]
  q = np.zeros(T)
  num_steps = int(np.floor((T-p_order) / batch_size))
  dim = p_order + 1 # math.comb(p+2, 2) # old was p+1
  theta = np.zeros((num_steps, dim))
  theta[0] = np.ones(dim) * (1/p_order) if p_order != 0 else np.ones(dim) *q_1 # 0.05 heuristically chosen to be on the scale of the scores.

  for i in range(num_steps):
      t = p_order + i * batch_size

      gradients = []
      for j in range(batch_size):
          # Make predictions in batch
          phi_ij = np.concatenate((scores[t+j-p_order:t+j], [bias]))
          q[t+j] = theta[i].T @ phi_ij
          err_ij = (scores[t+j] > q[t+j]).astype(int)
          gradients.append((err_ij - alpha) * phi_ij)
    
      # This is actually the negative gradient.
      gradient = np.mean(gradients, axis=0)
 
      if i < num_steps - 1:
          theta[i + 1] = theta[i] + (etas[i]) * gradient

  # First p predictions are empty
  return q 

# Parameterized quantile tracking. Implementation that involves batching.
def get_linear_batched_qt_thetas(scores, q_1, etas, alpha, p_order, bias, batch_size=1):
  T = scores.shape[0]
  q = np.zeros(T)
  num_steps = int(np.floor((T-p_order) / batch_size))
  dim = p_order + 1 # math.comb(p+2, 2) # old was p+1
  theta = np.zeros((num_steps, dim))
  theta[0] = np.ones(dim) * (1/p_order) if p_order != 0 else np.ones(dim) *q_1 # 0.05 heuristically chosen to be on the scale of the scores.

  for i in range(num_steps):
      t = p_order + i * batch_size

      gradients = []
      for j in range(batch_size):
          # Make predictions in batch
          phi_ij = np.concatenate((scores[t+j-p_order:t+j], [bias]))
          q[t+j] = theta[i].T @ phi_ij
          err_ij = (scores[t+j] > q[t+j]).astype(int)
          gradients.append((err_ij - alpha) * phi_ij)
    
      # This is actually the negative gradient.
      gradient = np.mean(gradients, axis=0)
 
      if i < num_steps - 1:
          theta[i + 1] = theta[i] + (etas[i]) * gradient

  return theta


# Implements ACI on top of an AR model fit to the past window_size scores.
def qt_on_ar(scores, q_1, etas, alpha, p_order=1, window_size=200):
  T = scores.shape[0]
  correction_term = 0
  q = np.zeros(T)
  correction_term = np.zeros(T)
  X = []
  y = np.empty(0)
  # Implement as theta
  phi = np.zeros((T, p_order+1))
  for t in tqdm(range(p_order, T)):
      phi[t] = np.concatenate((scores[t-p_order:t], [BIAS_TERM_QT_ON_AR]))

      # Calculate \theta by minimizing quantile loss on the past
      theta = None

      X.append(phi[t-1])
      y = np.append(y, scores[t-1])

      if t > window_size + p_order:
          X = X[1:]
          y = y[1:]

      # Add a bias term (column of ones) to X
      X_b = np.array(X)

      # Quantile to be estimated
      quantile = 1-alpha

      # Define the regression coefficients
      beta = cp.Variable(X_b.shape[1])

      # Define the residuals
      residuals = y - X_b @ beta

      # Define the quantile loss
      quantile_loss = cp.sum(cp.maximum(quantile * residuals, (quantile - 1) * residuals))

      # Define and solve the problem
      problem = cp.Problem(cp.Minimize(quantile_loss))
      problem.solve()

      theta = beta.value

      q[t] = theta.T @ phi[t] + correction_term[t]

      err_t = (scores[t] > q[t]).astype(int)
      if t < T - 1:
          correction_term[t + 1] = correction_term[t] + etas[t] * (err_t - alpha)

  # First p_order predictions in q are empty
  return q #, correction_term

# Conformal PI control paper. Copied from https://github.com/aangelopoulos/conformal-time-series/blob/main/core/methods.py
def get_pi_control(
    scores,
    q_1,
    etas,
    alpha,
    Csat=50, 
    KI=10,
    ahead=1, 
    T_burnin=20, # for best comparison with ours (I think)
    proportional_lr=True,
    *args,
    **kwargs
):
    data = kwargs['data'] if 'data' in kwargs.keys() else None
    results = get_pid_control(scores, q_1, etas, alpha, data, T_burnin, Csat, KI, True, ahead, proportional_lr=proportional_lr, scorecast=False)
    return results


"""
    This is the master method for the quantile, integrator, and scorecaster methods.
"""
def get_pid_control(
    scores,
    q_1,
    etas,
    alpha,
    data = None,
    T_burnin=20,
    Csat=50,
    KI=10,
    upper=True,
    ahead=1,
    integrate=True,
    proportional_lr=True,
    scorecast=True,
    scorecaster="theta",
    p_order_for_ar_scorecaster=1,
    window_size=200,#10000000,
#    onesided_integrator=False,
    *args,
    **kwargs
):
    lr = etas[0] # new
    # Initialization
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    qts = np.zeros((T_test,))
    integrators = np.zeros((T_test,))
    scorecasts = np.zeros((T_test,))
    covereds = np.zeros((T_test,))
    seasonal_period = kwargs.get('seasonal_period')
    if seasonal_period is None:
        seasonal_period = 1
    # Load the scorecaster
    try:
        # If the data contains a scorecasts column, then use it!
        if 'scorecasts' in data.columns:
            scorecasts = np.array([s[int(upper)] for s in data['scorecasts'] ])
            train_model = False
        else:
            scorecasts = np.load('./.cache/scorecaster/' + kwargs.get('config_name') + '_' + str(upper) + '.npy')
            train_model = False
    except:
        train_model = True
    # Run the main loop
    # At time t, we observe y_t and make a prediction for y_{t+ahead}
    # We also update the quantile at the next time-step, q[t+1], based on information up to and including t_pred = t - ahead + 1.
    # lr_t = lr * (scores[:T_burnin].max() - scores[:T_burnin].min()) if proportional_lr and T_burnin > 0 else lr
    for t in tqdm(range(T_test)):
        t_lr = t
        t_lr_min = max(t_lr - T_burnin, 0)
        lr_t = lr * (scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min()) if proportional_lr and t_lr > 0 else etas[t]
        t_pred = t - ahead + 1 
        if t_pred < 0:
            continue # We can't make any predictions yet if our prediction time has not yet arrived
        # First, observe y_t and calculate coverage
        covereds[t] = qs[t] >= scores[t]
        # Next, calculate the quantile update and saturation function
        grad = alpha if covereds[t_pred] else -(1-alpha)
        #integrator = saturation_fn_log((1-covereds)[T_burnin:t_pred].sum() - (t_pred-T_burnin)*alpha, (t_pred-T_burnin), Csat, KI) if t_pred > T_burnin else 0
        integrator_arg = (1-covereds)[:t_pred].sum() - (t_pred)*alpha
        #if onesided_integrator:
        #    integrator_arg = np.clip(integrator_arg, 0, np.infty)
        integrator = saturation_fn_log(integrator_arg, t_pred, Csat, KI)
        # Train and scorecast if necessary
        if scorecast and train_model and t_pred > T_burnin and t+ahead < T_test:
            curr_scores = np.nan_to_num(scores[max(0, t_pred-window_size):t_pred]) # CHANGE
            if scorecaster=="theta":
                model = ThetaModel(
                        curr_scores.astype(float),
                        period=seasonal_period,
                        ).fit()
                scorecasts[t+ahead] = model.forecast(ahead)
            elif scorecaster=="ar":
                betahat = fit_ar_model(curr_scores, p_order_for_ar_scorecaster, X=None)
                scorecasts[t+ahead] = predict_ar_model(curr_scores[-p_order_for_ar_scorecaster:], betahat)
            else:
                raise ValueError(f'scorecaster {scorecaster} not implemented.')
        # Update the next quantile
        if t < T_test - 1:
            qts[t+1] = qts[t] - lr_t * grad
            integrators[t+1] = integrator if integrate else 0
            qs[t+1] = qts[t+1] + integrators[t+1]
            if scorecast:
                qs[t+1] += scorecasts[t+1]
            if not qs[t+1] <  np.infty:
                print('Infinite prediction being corrected')
                if integrator < np.infty:
                    print(integrator)
                    print('Unexplained infinite prediction.')
                qs[t+1] = max(scores) #scores[t+1]
    results = {"method": "Quantile+Integrator (log)+Scorecaster", "q" : qs}
    if train_model and scorecast:
        pass
        #os.makedirs('./.cache/', exist_ok=True)
        #os.makedirs('./.cache/scorecaster/', exist_ok=True)
        #np.save('./.cache/scorecaster/' + kwargs.get('config_name') + '_' + str(upper) + '.npy', scorecasts)
    return results["q"]

# TODO: possibly remove.
def quantile(
    scores,
    alpha,
    lr,
    ahead,
    proportional_lr=True,
    *args,
    **kwargs
):
    T_burnin = kwargs['T_burnin']
    results = get_pi_control(scores, alpha, lr, 1.0, 0, ahead, T_burnin, proportional_lr=proportional_lr)
    results['method'] = 'Quantile'
    return results

def mytan(x):
    if x >= np.pi/2:
        return np.inf
    elif x <= -np.pi/2:
        return -np.inf
    else:
        return np.tan(x)

def saturation_fn_log(x, t, Csat, KI):
    if KI == 0:
        return 0
    tan_out = mytan(x * np.log(t+1)/(Csat * (t+1)))
    out = KI * tan_out
    return  out

def saturation_fn_sqrt(x, t, Csat, KI):
    return KI * mytan((x * np.sqrt(t+1))/((Csat * (t+1))))

# ACI. Copied from https://github.com/aangelopoulos/conformal-time-series/blob/main/core/methods.py
# Modication: learning rate.
def aci(
    scores,
    q_1,
    etas,
    alpha,
    window_length=10000000,
    T_burnin=20,
    ahead=1,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    alphat = alpha
    qs = np.zeros((T_test,))
    alphas = np.ones((T_test,)) * alpha
    covereds = np.zeros((T_test,))
    for t in tqdm(range(T_test)):
        t_pred = t - ahead + 1
        if t_pred > T_burnin:
            # Setup: current gradient
            if alphat <= 1/(t_pred+1):
                #print('Infinite prediction being corrected')
                qs[t] = max(scores) #scores[t] #max(scores) # CHANGE.np.infty
            else:
                qs[t] = np.quantile(scores[max(t_pred-window_length,0):t_pred], 1-np.clip(alphat, 0, 1), method='higher')
            covereds[t] = qs[t] >= scores[t]
            grad = -alpha if covereds[t_pred] else 1-alpha
            alphat = alphat - etas[t]*grad

            if t < T_test - 1:
                alphas[t+1] = alphat
        else:
            if t_pred > np.ceil(1/alpha):
                qs[t] = np.quantile(scores[:t_pred], 1-alpha)
            else:
                print('Infinite prediction being corrected')
                qs[t] = max(scores) #max(scores) # CHANGE.np.infty
    #results = { "method": "ACI", "q" : qs, "alpha" : alphas}
    return qs

# Loss function implementations.
def quantile_loss(y_true, y_pred, quantile):
    """
    Calculate the quantile loss for given true and predicted values.

    Parameters:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted target values.
    quantile (float): Quantile to use for calculating the loss. Must be between 0 and 1.

    Returns:
    float: Quantile loss.
    """
    assert 0 <= quantile <= 1, "Quantile should be between 0 and 1"

    error = y_true - y_pred
    loss = np.where(error >= 0, quantile * error, (quantile - 1) * error)
    return np.mean(loss)

def absolute_loss(y_true, y_pred):
    """
    Calculate the absolute loss for given true and predicted values.

    Parameters:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted target values.

    Returns:
    float: Absolute loss.
    """
    error = np.abs(y_true - y_pred)
    return np.mean(error)


def square_loss(y_true, y_pred):
    """
    Calculate the square loss for given true and predicted values.

    Parameters:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted target values.

    Returns:
    float: Square loss.
    """
    error = y_true - y_pred
    return np.mean(error**2)
