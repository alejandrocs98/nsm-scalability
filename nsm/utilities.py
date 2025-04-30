import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


# Function to process dataframes
def process_df(df, species, mediators, inputs):
    # return vector of evaluation times, t = [n]
    # matrix of inputs, U = [n, len(inputs)]
    # matrix of species initial and final conditions, S = [n, 2, len(species)]
    # matrix of mediator initial and final conditions, M = [n, 2, len(mediators)]
    t = []
    U = []
    S = []
    M = []

    # count number of measurements of species and mediators
    N = np.zeros(len(species) + len(mediators))

    # loop over each unique condition
    for treatment, comm_data in df.groupby("Treatments"):
        # make sure comm_data is sorted in chronological order
        comm_data.sort_values(by='Time', ascending=True, inplace=True)

        # pull evaluation times
        t_eval = np.array(comm_data['Time'].values, float)

        # pull species data
        species_data = np.array(comm_data[species].values, float)
        # species_data = np.clip(np.array(comm_data[species].values, float), 0., np.inf)

        # pull mediators data
        mediators_data = np.array(comm_data[mediators].values, float)
        # mediators_data = np.clip(np.array(comm_data[mediators].values, float), 0., np.inf)

        # pull inputs data
        inputs_data = np.array(comm_data[inputs].values, float)

        # append data
        for i, tf in enumerate(t_eval[1:]):
            # append eval time
            t.append(tf)

            # append input values
            U.append(inputs_data[0])

            # append species data
            S.append(np.stack([species_data[0], species_data[i + 1]], 0))

            # append mediator data
            M.append(np.stack([mediators_data[0], mediators_data[i + 1]], 0))

            # count species measurement if species was originally present
            N[:len(species)] += np.array(species_data[0] > 0, int)

            # count metabolite measurement if not nan
            N[len(species):] += np.array(~np.isnan(mediators_data[1]), int)

    # return data
    return np.array(t), np.stack(U), np.stack(S), np.stack(M), N


def lin_fit(x, a, b):
    return a + b * x

def check_convergence(f):

    # convert to numpy array
    f = np.array(f)

    # ignore nans
    f = f[~np.isnan(f)]

    p, cov = curve_fit(lin_fit, xdata=np.arange(len(f)), ydata=f/np.max(np.abs(f)), p0=[1., 0.])
    a, b, = p

    # return value of slope
    return b
