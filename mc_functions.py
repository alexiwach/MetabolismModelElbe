"""
this script contains the functions for the monte-carlo error propagation for the metabolism model 
"""

# imports
import sys
import numpy as np

# * home brew imports
sys.path.append('/')
from model_functions import get_daily_params_mle

def sine(x, a, b, c, tau):
  """
  Calculate the value of a sinusoidal function at a given point.

  Parameters:
  x (float or array-like): Input value(s) for the function.
  a (float): Amplitude of the sinusoidal function.
  b (float): Amplitude of the cosine term.
  c (float): Phase shift of the sinusoidal function.
  tau (float): Period of the sinusoidal function.

  Returns:
  float or array-like: Value(s) of the sinusoidal function at the given point(s).

  """
  omega = 2 * np.pi / tau
  return a + b * np.cos(omega * (x - c))


def get_mc_summary(arr):
  """
  Calculate summary statistics for a given array.

  Parameters:
  arr (array-like): Input array.

  Returns:
  list: List containing mean, standard deviation, 5th percentile, 50th percentile (median), and 95th percentile.

  """
  mu_arr, sigma_arr = np.nanmean(arr), np.nanstd(arr)
  percentiless_arr = [np.nanpercentile(arr, p) for p in [5, 50, 95]]
  return [mu_arr, sigma_arr, *percentiless_arr]


def get_param_sigmas(doy, q):
  """
  Calculate standard deviations of model parameters.

  Parameters:
  doy (float): Day of the year.
  q (float): Parameter 'q'.

  Returns:
  tuple: Standard deviations of k600, twater, and domax parameters.
  """

  k600sigma = .005 * q ** .54
  twatersigma = sine(doy, .35, .2, 180, 365)
  domaxsigma = sine(doy, .065, .025, 140, 365)

  return k600sigma, twatersigma, domaxsigma


def get_daily_param_permutations(daily_input, n=100):
  """
  Generate daily parameter permutations based on input uncertainties.

  Parameters:
  daily_input (tuple): Tuple containing input parameters (z_t, par_t, do_t1, do_max_t1, t_water, k600, q, doy).
  n (int, optional): Number of permutations (default is 100).

  Returns:
  list of arrays: List containing arrays representing daily parameter permutations.
  """

  z_t, par_t, do_t1, do_max_t1, t_water, k600, q, doy = daily_input

  # get sigmas
  k600_sigma, t_water_sigma, do_max_sigma = get_param_sigmas(doy[0], q[0])
  # * convert from mg/l to µmol/l
  do_max_sigma = mg_to_mmol(do_max_sigma) * 1000

  # draw from normal dists w/ N~(0, sigma)
  k600_errors = np.random.normal(loc=0, scale=k600_sigma, size=n)
  t_water_errors = np.random.normal(loc=0, scale=t_water_sigma, size=n)
  do_max_errors = np.random.normal(loc=0, scale=do_max_sigma, size=n)
  daily_input_vars = []

  # create copy of input and add errors
  for i in range(n):
    daily_input_mut = np.asarray(daily_input[:])
    daily_input_mut[5] += k600_errors[i]
    daily_input_mut[4] += t_water_errors[i]
    daily_input_mut[3] += np.asarray(do_max_errors[i])

    daily_input_vars.append(daily_input_mut)

  return daily_input_vars


def process_input_batch(batch, daily_observed_ddo):
  """
  Process a batch of input data (created by "get_daily_param_permutations") for daily parameter estimation.

  Parameters:
  batch (list of tuples): List of tuples containing input parameters for each grid cell.
  daily_observed_ddo (array-like): Observed daily data.

  Returns:
  list: List of results for each input set in the batch.
  """

  # create storage for batch results
  batch_results = []

  for i in range(len(batch)):
    input_set = batch[i]
    res = get_daily_params_mle(daily_input=input_set, daily_observed_ddo=daily_observed_ddo)
    batch_results.append(res)

  return batch_results
