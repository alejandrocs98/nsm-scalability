import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

class Simulator:

    def __init__(self, system, noise, variable_names):
        # system is a system of ODEs
        self.system = system

        # noise is a callable function to simulate measurement noise
        self.noise = noise

        # names of system variables
        self.variable_names = variable_names

    def integrate(self, x0, t_eval):
        # integrate ODE
        soln = solve_ivp(self.system, (0, t_eval[-1]), x0, t_eval=t_eval).y.T
        return soln

    def simulate(self, X, t_eval):
        # X is a matrix of initial conditions
        # t_eval is an array of measurement times

        sim_dfs = []
        for i, x0 in enumerate(X):
            # deterministic simulation
            sim = self.integrate(x0, t_eval)

            # add measurement noise to non-zero values
            sim += self.noise(sim.shape) * np.array(sim>0, int)

            # store data in pandas dataframe
            exp_df = pd.DataFrame()
            exp_name = [f"exp {i + 1}"] * len(t_eval)
            exp_df["Treatments"] = exp_name
            exp_df["Time"] = t_eval
            exp_df[self.variable_names] = sim
            sim_dfs.append(exp_df)

        sim_df = pd.concat(sim_dfs)
        return sim_df
