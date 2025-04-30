import os
import multiprocessing
from scipy.io import savemat, loadmat
from scipy.special import logsumexp
import numpy as np
# from tqdm import tqdm

# import libraries to compute gradients
import jax
from jax import vjp, jacfwd, vmap, pmap, random
from jax.experimental.ode import odeint
from jax.nn import tanh, sigmoid, softmax, relu
from scipy.optimize import minimize, curve_fit

# system of equations
from .nsm_system import *

# curve fit to check convergence
from .utilities import *

class NSM:
    def __init__(self, dataframe, species, mediators=[], observed=[], inputs=[],
                 n_hidden=8, n_layers=1,
                 s_cap=10., m_cap=10., nu2=.001, sigma2=.01,
                 rng_seed=0, verbose=True):

        # n_hidden: number of units in neural network layers
        # n_layers: number of hidden layers

        # s_cap: carrying capacity of species
        self.s_cap = s_cap
        # m_cap: carrying capacity of mediators
        self.m_cap = m_cap

        # set rng for parameter init
        self.rng_seed = int(rng_seed)

        # number of available devices for parallelizing code
        self.n_devices = 1  # jax.device_count()

        # model structure (latent mediators, neural net layers, hidden units in each layer)
        self.n_h = n_hidden
        self.n_layers = n_layers

        # dimensions
        self.n_s = len(species)
        self.n_m = len(mediators)

        # save names of species and mediators
        self.species = list(species)
        self.mediators = list(mediators)

        # number of state variables (inputs to neural network)
        self.n_x = self.n_s + self.n_m + len(inputs)

        # number of observed variables
        if len(observed) == 0:
            observed = list(species) + list(mediators)
        self.n_obs = len(observed)

        # set up data
        self.t, self.U, self.S, self.M, self.N_obs = process_df(dataframe, species, mediators, inputs)

        # scale species and mediators to be ~O(1)
        self.s_scale = 1. / np.nanmax(self.S[:, -1], 0)
        self.m_scale = 1. / np.nanmax(self.M[:, -1], 0)
        self.S *= self.s_scale
        self.M *= self.m_scale

        # hyperparameters
        self.nu2 = nu2 * np.ones(self.n_obs)
        self.sigma2 = sigma2 * np.ones(self.n_obs)

        # init params as None
        self.z = None
        self.lmbda = None

        # for additional output messages
        self.verbose = verbose

    def init_params(self, sample=False):

        # [C]_ij = rate that species j consumes resource i
        C = g_inv(np.ones([self.n_m, self.n_s]) / self.n_s)
        C_std = np.ones_like(C) / self.n_s

        # [P]_ij = rate that species j produces resource i
        P = g_inv(np.ones([self.n_m, self.n_s]) / self.n_s)
        P_std = np.ones_like(P) / self.n_s

        # degradation rate of metabolites
        d_m = g_inv(0.01 * np.ones(self.n_m))
        d_m_std = np.ones_like(d_m)

        # list of parameters
        params = [C, P, d_m]
        params_std = [C_std, P_std, d_m_std]

        # lower limit on log10 of params
        llim = -3.
        prior = [llim * np.ones_like(C),
                 llim * np.ones_like(P),
                 llim * np.ones_like(d_m)]

        # initialize neural network parameters
        p_std = 1. / np.sqrt(self.n_x)
        W1 = np.zeros([self.n_h, self.n_x])
        W1_std = p_std * np.ones_like(W1)
        b1 = np.zeros(self.n_h)
        b1_std = np.ones_like(b1)

        # init params
        params += [W1, b1]
        params_std += [W1_std, b1_std]
        prior += [np.zeros_like(W1), np.zeros_like(b1)]

        # for each layer
        for i in range(self.n_layers - 1):
            # map hidden dimension to next hidden layer
            p_std = 1. / np.sqrt(self.n_h)
            Wh = np.zeros([self.n_h, self.n_h])
            Wh_std = p_std * np.ones_like(Wh)
            bh = np.zeros(self.n_h)
            bh_std = np.ones_like(bh)

            params += [Wh, bh]
            params_std += [Wh_std, bh_std]
            prior += [np.zeros_like(Wh), np.zeros_like(bh)]

        # map hidden layer to growth rate
        p_std = 1. / np.sqrt(self.n_h)
        Wout = np.zeros([self.n_s + self.n_m, self.n_h])
        Wout_std = p_std * np.ones_like(Wout)
        bout = np.zeros(self.n_s + self.n_m)
        bout_std = np.ones_like(bout)

        # list of parameters
        params += [Wout, bout]
        params_std += [Wout_std, bout_std]
        prior += [np.zeros_like(Wout), np.zeros_like(bout)]

        # transform and flatten params
        self.z = jnp.concatenate([p.ravel() for p in params])
        self.prior = jnp.concatenate([p.ravel() for p in prior])
        self.d = len(self.z)

        z_std = jnp.concatenate([p.ravel() for p in params_std])
        self.alpha = jnp.ones_like(self.z)  # 1. / ((self.z - self.prior) ** 2 + z_std ** 2)
        self.lmbda = jnp.append(self.z, np.log(z_std))

        # determine shapes of parameters
        shapes = []
        k_params = []
        self.n_params = 0
        for param in params:
            shapes.append(param.shape)
            k_params.append(self.n_params)
            self.n_params += param.size
        k_params.append(self.n_params)

        # make shapes immutable
        self.shapes = tuple((tuple(k_params), tuple(shapes)))

        # sample parameters
        if sample:
            self.z = T(self.shapes, np.random.randn(self.d), self.lmbda)

    # compute residual between parameter estimate and prior
    def param_res(self, params):
        # residuals
        res = params - self.prior
        return res

    # negative log posterior
    def nlp(self, z):
        # print("computing nlp...")
        # log prior
        self.NLP = log_prior_z(self.prior, self.alpha, z)

        # total number of samples
        N = len(self.t)

        # evaluate samples in batches
        for batch_inds in np.array_split(np.arange(N), np.ceil(N / self.n_devices)):
            # gradient of log likelihood
            self.NLP += np.nansum(batch_log_likelihood_z(self.shapes,
                                                         self.t[batch_inds],
                                                         self.U[batch_inds],
                                                         self.S[batch_inds],
                                                         self.M[batch_inds],
                                                         self.nu2, self.sigma2,
                                                         self.s_cap, self.m_cap,
                                                         z))

        # print("nlp", self.NLP)
        return self.NLP

    # mean squared error
    def rmse(self, z):

        # init mse
        rmse = 0.

        # total number of samples
        N = len(self.t)

        # evaluate samples in batches
        for idx in range(N):
            # gradient of log likelihood
            # root_mean_squared_error(shapes, tf, inputs, s, m, s_cap, m_cap, z)
            rmse += root_mean_squared_error(self.shapes,
                                            self.t[idx],
                                            self.U[idx],
                                            self.S[idx],
                                            self.M[idx],
                                            self.s_cap, self.m_cap,
                                            z) / N

        # print("nlp", self.NLP)
        return rmse

    # gradient of negative log posterior
    def grad_nlp(self, z):
        # print("computing grad nlp...")

        # log prior
        grad_NLP = grad_log_prior_z(self.prior, self.alpha, z)

        # total number of samples
        N = len(self.t)

        # evaluate samples in batches
        for batch_inds in np.array_split(np.arange(N), np.ceil(N / self.n_devices)):
            # gradient of log likelihood
            gradients = batch_grad_log_likelihood_z(self.shapes,
                                                    self.t[batch_inds],
                                                    self.U[batch_inds],
                                                    self.S[batch_inds],
                                                    self.M[batch_inds],
                                                    self.nu2, self.sigma2,
                                                    self.s_cap, self.m_cap,
                                                    z)

            # ignore exploding gradients
            gradients = np.where(np.abs(gradients) < 1000, gradients, 0.)

            # add sum of gradients
            grad_NLP += np.nansum(gradients, 0)

        return grad_NLP

    def fit_rmse(self, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, epochs=10000):

        # init moments of gradient
        m = np.zeros_like(self.z)
        v = np.zeros_like(self.z)
        t = 0
        epoch = 0

        # order of samples
        N = len(self.t)
        order = np.arange(N)

        # initialize function evaluations
        f = []

        while epoch <= epochs:
            if epoch % 10 == 0:
                # check convergence
                f.append(self.rmse(self.z))

                print("Epoch {:.0f}, RMSE: {:.3f}".format(epoch, f[-1]))
            epoch += 1

            # # batch gradient descent
            # gradient = 0.
            #
            # # take step for each sample
            # n_nan = 0
            # for idx in order:
            #
            #     # gradient of log likelihood (shapes, tf, inputs, s, m, nu2, sigma2, s_cap, m_cap, z)
            #     gradient += grad_root_mean_squared_error(self.shapes,
            #                                              self.t[idx],
            #                                              self.U[idx],
            #                                              self.S[idx],
            #                                              self.M[idx],
            #                                              self.s_cap, self.m_cap,
            #                                              self.z) / N
            #
            #     n_nan += np.sum(np.isnan(gradient))
            #
            # print(f"number of nans: {n_nan}")
            #
            # # ADAM: moment estimation
            # m = beta1 * m + (1 - beta1) * gradient
            # v = beta2 * v + (1 - beta2) * (gradient ** 2)
            #
            # # adjust moments based on number of iterations
            # t += 1
            # m_hat = m / (1 - beta1 ** t)
            # v_hat = v / (1 - beta2 ** t)
            #
            # # take step
            # self.z -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

            # stochastic gradient descent
            np.random.shuffle(order)

            # take step for each sample
            for idx in order:
                # gradient of log likelihood (shapes, tf, inputs, s, m, nu2, sigma2, s_cap, m_cap, z)
                # gradient = grad_log_likelihood_z(self.shapes,
                #                                  self.t[idx],
                #                                  self.U[idx],
                #                                  self.S[idx],
                #                                  self.M[idx],
                #                                  self.nu2, self.sigma2,
                #                                  self.s_cap, self.m_cap,
                #                                  self.z)
                gradient = grad_root_mean_squared_error(self.shapes,
                                                        self.t[idx],
                                                        self.U[idx],
                                                        self.S[idx],
                                                        self.M[idx],
                                                        self.s_cap, self.m_cap,
                                                        self.z)

                # ignore exploding gradients
                gradient = np.where(np.abs(gradient) < 1000., gradient, 0.)

                # moment estimation
                m = beta1 * m + (1 - beta1) * gradient
                v = beta2 * v + (1 - beta2) * (gradient ** 2)

                # adjust moments based on number of iterations
                t += 1
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # take step
                self.z -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

        # return list of function evaluations
        return f

    # variational inference to fit diagonal Gaussian posterior
    def fit_posterior(self, n_sample=1, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, max_epochs=100000, tol=1e-3,
                      patience=5):

        # init moments of gradient
        m = np.zeros_like(self.lmbda)
        v = np.zeros_like(self.lmbda)
        t = 0
        epoch = 0
        passes = 0
        fails = 0

        # order of samples
        N = len(self.t)
        order = np.arange(N)

        # save best parameters
        best_params = np.copy(self.lmbda)

        # initialize function evaluations
        f = []
        n_nan = 0

        while epoch <= max_epochs and passes < patience:

            if epoch % 10 == 0:

                # check convergence
                f.append(self.approx_evidence())

                # determine slope of elbo over time
                if len(f) > 2:
                    slope = check_convergence(f[-10:])
                else:
                    slope = 1.

                # check tolerance
                if abs(slope) < tol:
                    passes += 1
                    print(f"pass {passes}")
                else:
                    passes = 0

                # save parameters if improved
                if f[-1] >= np.max(f):
                    best_params = np.copy(self.lmbda)

                # if slope is negative and not improving, add to fail count
                if slope < 0 and f[-1] < f[-2] and epoch > 100:
                    fails += 1
                    print(f"fail {fails}")
                else:
                    fails = 0

                # if fails exceeds patience, return best parameters
                if fails == patience:
                    self.lmbda = jnp.array(best_params)
                    self.z = self.lmbda.at[:self.d].get()
                    return f

                print("Epoch {:.0f}, ELBO: {:.3f}, Slope: {:.3f}".format(epoch, f[-1], slope))
                print(f"encountered {n_nan} nans")
                n_nan = 0
            epoch += 1

            # # gradient of entropy of approximate distribution w.r.t log_s
            # gradient = np.append(np.zeros(self.d), -np.ones(self.d))
            #
            # # sample parameters
            # y = np.random.randn(n_sample, self.d)
            #
            # # gradient of negative log posterior
            # for yi in y:
            #
            #     # prior
            #     gradient += grad_neg_log_prior_lmbda(self.shapes, self.prior, yi, self.alpha, self.lmbda) / n_sample
            #
            #     # evaluate samples in batches
            #     for batch_inds in np.array_split(np.arange(N), np.ceil(N / self.n_devices)):
            #
            #         # gradient of log likelihood
            #         gradients = batch_grad_neg_log_likelihood_lmbda(self.shapes,
            #                                                         self.t[batch_inds],
            #                                                         self.U[batch_inds],
            #                                                         self.S[batch_inds],
            #                                                         self.M[batch_inds],
            #                                                         yi,
            #                                                         self.nu2, self.sigma2,
            #                                                         self.s_cap, self.m_cap,
            #                                                         self.lmbda)
            #
            #         # ignore exploding gradients
            #         # setting the limit too low can cause fitting problems
            #         gradient += np.nansum(np.where(np.abs(gradients) < 1000., gradients, 0.), 0) / n_sample
            #
            # # normalize gradient
            # # gradient /= np.linalg.norm(gradient)
            #
            # # moment estimation
            # m = beta1 * m + (1 - beta1) * gradient
            # v = beta2 * v + (1 - beta2) * (gradient ** 2)
            #
            # # adjust moments based on number of iterations
            # t += 1
            # m_hat = m / (1 - beta1 ** t)
            # v_hat = v / (1 - beta2 ** t)
            #
            # # take step
            # self.lmbda -= lr * m_hat / (np.sqrt(v_hat) + epsilon)  # / np.sqrt(t)
            #
            # # save the best parameter guess
            # z = transform(reshape(self.shapes, self.lmbda.at[:self.d].get()))
            # self.z = jnp.concatenate([zi.ravel() for zi in z])
            #
            # # update alpha
            # var = np.exp(self.lmbda[self.d:]) ** 2
            # self.alpha = 1. / ((self.z - self.prior) ** 2 + var + 1e-4)

            # shuffle order of data
            np.random.shuffle(order)

            # take step for each sample
            for idx in order:

                # gradient of entropy of approximate distribution w.r.t log_s
                gradient = np.append(np.zeros(self.d), -np.ones(self.d)) / N

                # sample parameters
                y = np.random.randn(n_sample, self.d)

                # gradient of negative log posterior
                for yi in y:

                    # prior (consider evaluating analytically)
                    gradient += grad_neg_log_prior_lmbda(self.shapes,
                                                         self.prior,
                                                         yi,
                                                         self.alpha,
                                                         self.lmbda) / N / n_sample

                    # gradient of log likelihood
                    gradients = grad_neg_log_likelihood_lmbda(self.shapes,
                                                              self.t[idx],
                                                              self.U[idx],
                                                              self.S[idx],
                                                              self.M[idx],
                                                              yi,
                                                              self.nu2, self.sigma2,
                                                              self.s_cap, self.m_cap,
                                                              self.lmbda) / n_sample

                    if any(np.isnan(gradients)) or any(np.isinf(gradients)):
                        n_nan += 1

                    # ignore exploding gradients
                    # setting the limit too low can cause fitting problems
                    gradient += np.where(np.abs(gradients) < 1000., gradients, 0.)
                    # gradient += np.clip(np.nan_to_num(gradients), -10, 10)
                    # gradient += np.nan_to_num(gradients)

                # moment estimation
                m = beta1 * m + (1 - beta1) * gradient
                v = beta2 * v + (1 - beta2) * (gradient ** 2)

                # adjust moments based on number of iterations
                t += 1
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # take step
                self.lmbda -= lr * m_hat / (np.sqrt(v_hat) + epsilon)  # / np.sqrt(t)
                self.z = self.lmbda.at[:self.d].get()

                # # update alpha
                # var = np.exp(self.lmbda[self.d:]) ** 2
                # self.alpha = 1. / ((self.z - self.prior) ** 2 + var + 1e-4)

    def fit_posterior_EM(self, lr=1e-3, n_sample=1, max_epochs=100000, n_sample_hypers=100, patience=1,
                         max_iterations=10):

        # optimize parameter posterior
        print("Updating posterior...")
        f = self.fit_posterior(n_sample=n_sample, lr=lr, max_epochs=max_epochs)

        # init evidence, fail count, iteration count
        # self.approx_evidence()
        # print("\nlog evidence: {:.3f}\n".format(self.approx_evidence()))
        # previdence = np.copy(self.log_evidence)
        previdence = -np.inf
        fails = 0
        t = 0
        while fails <= patience and t < max_iterations:

            # update iteration count
            t += 1

            # update prior and measurement precision estimate
            print("Updating hyperparameters...")
            self.update_hypers(n_sample=n_sample_hypers)

            # optimize parameter posterior
            print("Updating posterior...")
            f = self.fit_posterior(n_sample=n_sample, lr=lr, max_epochs=max_epochs)

            # update evidence
            print("Computing model evidence...")
            print("\nlog evidence: {:.3f}\n".format(self.approx_evidence()))

            # check convergence
            if self.log_evidence <= previdence:
                fails += 1
            else:
                fails = 0
            previdence = np.max([previdence, np.copy(self.log_evidence)])

    # evidence lower bound
    def approx_evidence(self):

        # entropy of posterior
        self.log_evidence = log_abs_det(self.lmbda)

        # divergence from prior
        self.log_evidence -= log_prior_z(self.prior, self.alpha, self.z)

        # total number of samples
        N = len(self.t)

        # evaluate samples in batches
        for batch_inds in np.array_split(np.arange(N), np.ceil(N / self.n_devices)):
            # log posterior
            self.log_evidence -= np.sum(batch_log_likelihood_z(self.shapes,
                                                               self.t[batch_inds],
                                                               self.U[batch_inds],
                                                               self.S[batch_inds],
                                                               self.M[batch_inds],
                                                               self.nu2, self.sigma2,
                                                               self.s_cap, self.m_cap,
                                                               self.z))

        return self.log_evidence

    # EM algorithm to update hyperparameters
    def update_hypers(self, n_sample=100):

        # create dictionaries of estimated/empirical moments for each output
        Z = {}
        Y2 = {}
        for j in range(self.n_obs):
            Z[j] = []
            Y2[j] = []

        # sample from posterior
        y = np.random.randn(n_sample, self.d)
        z = batch_T(self.shapes, y, self.lmbda)

        # stochastic estimate of covariance
        for params in z:

            # total number of samples
            N = len(self.t)

            # evaluate samples in batches
            for batch_inds in np.array_split(np.arange(N), np.ceil(N / self.n_devices)):

                # integrate batch
                batch_pred = batchODE(self.shapes,
                                      self.t[batch_inds],
                                      self.U[batch_inds],
                                      self.S[batch_inds],
                                      self.M[batch_inds],
                                      self.s_cap, self.m_cap,
                                      params)

                # loop over the batched outputs
                for pred, s, m in zip(batch_pred, self.S[batch_inds], self.M[batch_inds]):

                    # make sure predictions aren't NaN
                    if not (np.any(np.isnan(pred)) or np.any(np.isinf(pred))):

                        # concatenate species and mediators
                        true = jnp.append(s[-1], m[-1])

                        # clip negative predictions to zero
                        pred = jnp.clip(pred, 0., jnp.inf)

                        # Determine error
                        y_error = np.nan_to_num(true - np.nan_to_num(pred[:self.n_obs]))

                        # estimate noise
                        for j, (y_j, f_j, e_j) in enumerate(zip(true, pred, y_error)):
                            if y_j > 0:
                                Y2[j].append(y_j ** 2)
                                Z[j].append(e_j ** 2)

        # solve for noise parameters
        for j in range(self.n_obs):
            y2 = np.array(Y2[j])
            z = np.array(Z[j])
            B = np.vstack((np.ones_like(y2), y2)).T
            a = (np.linalg.inv(B.T @ B) @ B.T) @ z
            self.nu2[j] = np.max([a[0], 1e-4])
            self.sigma2[j] = np.max([a[1], 1e-4])

        # update alpha
        var = np.exp(self.lmbda[self.d:]) ** 2
        self.alpha = 1. / ((self.z - self.prior) ** 2 + var + 1e-4)

    def predict_point(self, x_test, t_eval, inputs=None):

        # set inputs to empty array if None
        if inputs is None:
            inputs = np.array([])

        # convert to arrays
        t_eval = np.array(t_eval, dtype=np.float32)

        # separate state
        s_test = np.atleast_2d(x_test)[:, :self.n_s]
        m_test = np.atleast_2d(x_test)[:, self.n_s:]

        # make predictions given initial conditions and evaluation times
        s_out, m_out = runODE_teval(self.shapes,
                                    t_eval, inputs,
                                    s_test[0] * self.s_scale,
                                    m_test[0] * self.m_scale,
                                    self.s_cap, self.m_cap,
                                    self.z)

        return s_out / self.s_scale, m_out / self.m_scale

    def predict_sample(self, x_test, t_eval, n_sample=100, inputs=None):

        # set inputs to empty array if None
        if inputs is None:
            inputs = np.array([])

        # convert to arrays
        t_eval = np.array(t_eval, dtype=np.float32)

        # separate state
        s_test = np.atleast_2d(x_test)[:, :self.n_s]
        m_test = np.atleast_2d(x_test)[:, self.n_s:]

        # sample parameters
        y = np.random.randn(n_sample, self.d)
        z = batch_T(self.shapes, y, self.lmbda)

        # make predictions given initial conditions and evaluation times
        s_out_batch, m_out_batch = [], []

        for zi in z:
            s_out, m_out = runODE_teval(self.shapes,
                                        t_eval, inputs,
                                        s_test[0] * self.s_scale,
                                        m_test[0] * self.m_scale,
                                        self.s_cap, self.m_cap,
                                        zi)

            s_out_batch.append(s_out / self.s_scale)
            m_out_batch.append(m_out / self.m_scale)

        return s_out_batch, m_out_batch

    def predict(self, x_test, t_eval, n_sample=100, inputs=None):

        # set inputs to empty array if None
        if inputs is None:
            inputs = np.array([])

        # convert to arrays
        t_eval = np.array(t_eval, dtype=np.float32)

        # separate state
        s_test = np.atleast_2d(x_test)[:, :self.n_s]
        m_test = np.atleast_2d(x_test)[:, self.n_s:]

        # sample parameters
        y = np.random.randn(n_sample, self.d)
        z = batch_T(self.shapes, y, self.lmbda)

        # make predictions given initial conditions and evaluation times
        s_out_batch, m_out_batch = [], []

        for zi in z:
            s_out, m_out = runODE_teval(self.shapes,
                                        t_eval, inputs,
                                        s_test[0] * self.s_scale,
                                        m_test[0] * self.m_scale,
                                        self.s_cap, self.m_cap,
                                        zi)

            s_out_batch.append(s_out / self.s_scale)
            m_out_batch.append(m_out / self.m_scale)

        # stack posterior predictive samples
        s_out_batch = np.stack(s_out_batch)
        m_out_batch = np.stack(m_out_batch)

        # take mean and variance
        s_mean = np.nanmean(s_out_batch, 0)
        m_mean = np.nanmean(m_out_batch, 0)
        s_var = np.nanvar(s_out_batch, 0)
        m_var = np.nanvar(m_out_batch, 0)

        # account for aleatory uncertainty
        s_var += self.nu2[:self.n_s] + self.sigma2[:self.n_s] * np.clip(s_mean, 0., np.inf) ** 2.
        m_var += self.nu2[self.n_s:] + self.sigma2[self.n_s:] * np.clip(m_mean, 0., np.inf) ** 2.

        return s_mean, s_var, m_mean, m_var