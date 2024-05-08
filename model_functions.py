"""
this script contains the functions implementing the maximum likelihood estimations for metabolism estimation
"""

# imports 
import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy import optimize

def mg_to_mmol(mg, molar_mass=32):
    """
    Convert a quantity from milligrams (mg) to millimoles (mmol)
    given the molar mass of the substance.

    Parameters:
    mg (float): The quantity in milligrams.
    molar_mass (float): The molar mass of the substance in grams per mole.

    Returns:
    float: The equivalent quantity in millimoles.
    """
    mmol = (mg / molar_mass)
    return mmol


def mmol_to_mg(mmol, molar_mass=32):
    """
    Convert a quantity from millimoles (mmol) to milligrams (mg)
    given the molar mass of the substance.

    Parameters:
    mmol (float): The quantity in millimoles.
    molar_mass (float): The molar mass of the substance in grams per mole.

    Returns:
    float: The equivalent quantity in milligrams.
    """
    mg = mmol * molar_mass# * 0.001
    return mg


def schmidt_number_from_t_water(t):
    """
    This method calculates the Schmidt numbers which is used
    to convert k to k600 (the temperature independent gas transfer
    coefficient for O2 in freshwater); see https://doi.org/10.1029/92JC00188
    :param t: water temperature [deg C]
    :return: schmidt number

    """
    A = 1800.6
    B = 120.1
    C = 3.7818
    D = 0.047608
    sc = A - B * t + C * t ** 2 - D * t ** 3
    return sc


def check_input(model_input):
    """

    :param model_input:
    :return: raises an error if input lengths are incorrect
    """
    z_t, I_t, do_t1, do_max_t1, t_water, k600, q, doy = model_input

    # test if all input arrays are of same length (len = 24)
    if not all(
            [
                (len(do_t1) == 24),
                (len(do_max_t1) == 24),
                (len(z_t) == 24),
                (len(I_t) == 24),
                (len(t_water) == 24),
                (len(k600) == 24),
            ]
    ):
        raise ValueError('inputs have wrong length')


def build_model_input(input_df: pd.DataFrame, date: str):
    """
    This function builds the model input for the 'predict_do' function for a
    given 'date'.
    :param input_df: pandas DataFrame
    :param date: in DD-MM-YYYY format
    :return:
    """

    # build 25 hour date time index for 'date'
    # * build date time index for that date
    start_date = (
                pd.to_datetime(
                    date + ' 03:00:00',
                    format='%Y-%m-%d %H:%M:%S'
                )
    )
    end_date = (
            start_date +
            pd.Timedelta(value=1, unit='D') +
            pd.Timedelta(value=0, unit='H')
    )
    datetimes = pd.date_range(start=start_date, end=end_date, freq='H')
    data2model = input_df[input_df.DateTime.isin(datetimes)]

    # prepare model inputs
    # * split 25 hour input arrays
    z = np.asarray(data2model.depth_M)
    I = np.asarray(data2model.PPFD_hourly_sum_solarPy)
    do_obs = np.asarray(data2model.DO_CONC_in)
    do_max = np.asarray(data2model.DO_MAX_CONC_in)
    t_water = np.asarray(data2model.t_water_daily)
    k600 = np.asarray(data2model.k600)
    # * convert DO from mg/l to µMol/l
    do_obs = mg_to_mmol(do_obs) * 1000
    do_max = mg_to_mmol(do_max) * 1000
    # * calculate hourly change in DO concentrations (ddo)
    ddo = np.diff(a=do_obs, n=1)
    # * crop arrays to 24 hours
    z = z[1:]
    I = I[1:]
    t_water = t_water[1:]
    do_obs = do_obs[1:]
    do_max = do_max[1:]
    k600 = k600[1:]

    # * get Q and day of year for error propagation
    q = np.asarray([data2model.Q_CMS.iloc[0]] *  len(z))
    doy = np.asarray([data2model.DateTime.dt.dayofyear.iloc[0]] * len(z))

    # * build model input and check its validity
    model_input = [z, I, do_obs, do_max, t_water, k600, q, doy]
    check_input(model_input)

    return model_input, ddo


def predict_ddo(model_input, gpp24, er24):
  """
  this function calculates the change in dissolved oxygen (ddo) based on
  water depth, solar radiation, GPP, ER and an oxygen saturation deficit.
  for details see Van de Bogert et al., 2007 at
  https://aslopubs.onlinelibrary.wiley.com/doi/epdf/10.4319/lom.2007.5.145

  :param model_input: 5-tuple of arrays [z, I, do, do_max, t_water]
                      z                                   [m]
                      I                                   [µmol m^-2 h^-1]
                      do_obs                              [µmol/l]
                      do_max                              [µmol/l]
                      t_water                             [deg C]
                      ddo                                 [µmol/l]
  :param gpp24: daily rate of gross primary production    [mmol m^-3 d^-1]
  :param er24: daily rate of gross respiration            [mmol m^-3 d^-1]
  :return: simulated DO                                   [µmol/l]
  """

  # define delta_t (1 [hour]) in [days]
  DELTA_T = 1 / 24

  # split model input
  z_t, par_t, do_t1, do_max_t1, t_water, k600, q, doy = model_input

  # compute daily sum of I [µmol m^-2 d^-1]
  par24 = np.sum(par_t)

  # compute Schmidts number [-] for given water temperatures
  sc = schmidt_number_from_t_water(t=t_water)

  # predict oxygen concentrations
  ddo_sim = (
          ((gpp24/par24) * par_t) -
          (er24 * DELTA_T) +
          (
                  (
                          (k600 * (sc/600) ** -.5) * (do_max_t1 - do_t1)
                  ) / z_t
          ) * DELTA_T
  )

  return ddo_sim


def get_gof_metrics(obs, sim):

  def pearsons_r(predictions, targets):
        """
        this function returns the 'pearsons r' metric based on the output
        of the 'linregress' function from scipy.stats
        :param predictions: array-like
        :param targets: array-like
        :return:
        """
        lr = linregress(predictions,targets)
        return lr.rvalue

  def rmse(predictions, targets):
      """
      this function returns the 'rmse' (root mean square error) metric.
      see https://en.wikipedia.org/wiki/Root-mean-square_deviation
      :param predictions: array-like
      :param targets: array-like
      :returns:
      """

      return np.sqrt(((predictions - targets) ** 2).mean())

  def mean_absolute_error(predictions, targets):
      """
      this function calculates the mean absolute error.
      see www.geeksforgeeks.org/how-to-calculate-mean-absolute-error-in-python
      :param predictions: array-like
      :param targets: array-like
      :returns:
      """
      # sum up errors
      sum_term = 0
      for i in range(len(predictions)):
          sum_term += np.abs(targets[i]-predictions[i])
      # normalize on n
      mae = sum_term / len(predictions)

      return mae

  return [pearsons_r(sim, obs), rmse(sim, obs), mean_absolute_error(sim, obs)]


def log_likelihood(params, x, y):
  """
  Compute the log-likelihood of a given model's parameters for a normal distribution.
  
  Parameters:
  params (tuple): Tuple of model parameters (param1, param2, sigma).
  x (array-like): Input data.
  y (array-like): Observed data.
  
  Returns:
  float: Log-likelihood value.
  """
  # Unpack parameters
  param1, param2, sigma = params

  # Get the model simulation
  y_sim = predict_ddo(model_input=x, gpp24=param1, er24=param2)

  # Calculate residuals
  residuals = y - y_sim

  # Calculate the log-likelihood for a normal distribution
  # Negative sign for minimization purposes
  n = len(x)
  nll = -(-n/2 * np.log(2 * np.pi * sigma**2) - np.sum(residuals**2) / (2 * sigma**2))

  return nll


def get_daily_params_mle(
    daily_input,
    daily_observed_ddo,
    initial_guess=[312, 312, 1],
    bounds=[(0, 1500), (0, 1500), (0, np.inf)]
    ):
  """
  Estimate model parameters using Maximum Likelihood Estimation (MLE) for daily data.

  Parameters:
  daily_input (array-like): Daily input data.
  daily_observed_ddo (array-like): Observed daily data.
  initial_guess (list, optional): Initial guess for the parameters (default is [312, 312, 1]).
  bounds (list of tuples, optional): Bounds for the parameters (default is [(0, 1500), (0, 1500), (0, np.inf)]).

  Returns:
  list: Estimated parameters and goodness-of-fit metrics.
  """

  # get mle estimations of params
  result = optimize.minimize(
      log_likelihood,
      initial_guess,
      args=(daily_input, daily_observed_ddo),
      method='L-BFGS-B',
      bounds=bounds
  )

  # compute goodness of fit metrics
  # * extract params
  gpp_val, er_val, sigma_val = result.x
  # * simulate ddo
  daily_simulated_ddo = predict_ddo(daily_input, gpp_val, er_val)
  # * get metrics
  gof = get_gof_metrics(daily_observed_ddo, daily_simulated_ddo)

  return [gpp_val, er_val, sigma_val, *gof]

