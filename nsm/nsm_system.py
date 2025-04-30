import jax.numpy as jnp
from jax.nn import tanh, sigmoid, relu, softplus
from jax.experimental.ode import odeint
from jax import jit, vmap, pmap, jacfwd, jacrev
from jax.lax import dynamic_slice
from functools import partial


# define system of equations
@jit
def system(x, t, params, inputs, s_cap, m_cap):
    # unpack species and mediators
    s, m = x

    # unpack mediator production, consumption, degradation rates
    C, P, d_m = params[0], params[1], params[2]

    # determine number of layers
    n_layers = (len(params) - 3) // 2 - 2

    # input weights / bias
    Wi, bi = params[3], params[4]

    # output weights / bias
    Wo, bo = params[-2], params[-1]

    # compute input state to nn
    h = jnp.concatenate((s, m, inputs))

    # hidden state
    h = tanh(Wi @ h + bi)

    # for each layer
    for i in range(n_layers):
        # weights and bias
        Wh, bh = params[5 + 2 * i], params[6 + 2 * i]

        # hidden state
        h = tanh(Wh @ h + bh) / 10.

    # output state
    o = (Wo @ h + bo) / 10.

    # growth rates
    dsdt = s * o[:len(s)] * (1. - s / s_cap)

    # rate of change of resources
    # dmdt = softplus(o[len(s):]) * ((P @ relu(dsdt)) * (1. - m / m_cap) - m * (C @ relu(dsdt) + d_m))
    # dmdt = softplus(o[len(s):]) * ((P @ relu(dsdt)) * (1. - m / m_cap) - m * (C @ s + d_m))
    dmdt = softplus(o[len(s):]) * ((P @ s) * (1. - m / m_cap) - m * (C @ s + d_m))

    return dsdt, dmdt


# transform to strictly positive value
@jit
def g(a):
    # return jnp.log(1. + jnp.exp(a))
    return jnp.log10(1. + 10. ** a)


@jit
def g_inv(a):
    # return jnp.log(jnp.exp(a) - 1.))
    return jnp.log10(10. ** a - 1.)


# constrain parameters from real numbers to strictly non-negative values
@jit
def transform(params):

    # unpack params
    C, P, d_m = params[0], params[1], params[2]

    # transform to strictly positive values
    C = g(C)
    P = g(P)
    d_m = g(d_m)

    return [C, P, d_m] + params[3:]


@partial(jit, static_argnums=(0,))
def runODE(shapes, tf, inputs, s, m, s_cap, m_cap, z):
    # transform and reshape parameters
    params = transform(reshape(shapes, z))

    # get initial condition of species and mediators
    s_ic = s[0]
    m_ic = m[0]

    # integrate
    t_eval = jnp.linspace(0., tf, 10)
    s_pred, m_pred = odeint(system, [s_ic, m_ic], t_eval, params, inputs, s_cap, m_cap,
                            rtol=1.4e-8, atol=1.4e-8, mxstep=10000, hmax=jnp.inf)

    # concatenate output
    pred = jnp.concatenate((s_pred, m_pred), -1)

    # return final time point
    return pred[-1]


# integrate batch
batchODE = pmap(runODE,
                in_axes=(None, 0, 0, 0, 0, None, None, None),
                static_broadcasted_argnums=(0,))


# gradient of solution of ODE w.r.t. parameters
@partial(jit, static_argnums=(0,))
def jacODE(shapes, tf, inputs, s, m, s_cap, m_cap, z):
    return jacrev(runODE, -1)(shapes, tf, inputs, s, m, s_cap, m_cap, z)


# evaluate Jacobian in batches
batch_jacODE = pmap(jacODE,
                    in_axes=(None, 0, 0, 0, 0, None, None, None),
                    static_broadcasted_argnums=(0,))


# evaluate ODE at specified time points
@partial(jit, static_argnums=(0,))
def runODE_teval(shapes, t_eval, inputs, s_ic, m_ic, s_cap, m_cap, z):
    # transform and reshape parameters
    params = transform(reshape(shapes, z))

    # integrate
    return odeint(system, [s_ic, m_ic], jnp.array(t_eval), params, inputs, s_cap, m_cap,
                  rtol=1.4e-8, atol=1.4e-8, mxstep=10000, hmax=jnp.inf)


# reshape parameter vector into matrix/vectors
@partial(jit, static_argnums=(0,))
def reshape(shapes, params):
    return [jnp.reshape(dynamic_slice(params, (k1,), (k2 - k1,)), shape) for k1, k2, shape in
            zip(shapes[0], shapes[0][1:], shapes[1])]


# invertible, differentiable function to map noise to model parameters
@partial(jit, static_argnums=(0,))
def T(shapes, y, lmbda):
    # variational parameters
    mu, log_s = lmbda.at[:len(lmbda) // 2].get(), lmbda.at[len(lmbda) // 2:].get()

    return mu + jnp.exp(log_s) * y


@partial(jit, static_argnums=(0,))
def batch_T(shapes, y_batch, lmbda):
    return vmap(T, (None, 0, None))(shapes, y_batch, lmbda)


@jit
def log_abs_det(lmbda):
    log_s = lmbda.at[len(lmbda) // 2:].get()
    return jnp.sum(log_s)


# gradient of entropy of approximating distribution w.r.t. lmbda
@partial(jit, static_argnums=(0,))
def grad_log_abs_det(shapes, y, lmbda):
    return jacrev(log_abs_det, -1)(shapes, y, lmbda)


# log likelihood of standard Gaussian
@jit
def log_py(y):
    return -jnp.dot(y, y) / 2. - len(y) / 2 * jnp.log(2. * jnp.pi)


# evaluate log prior
@partial(jit, static_argnums=(0,))
def neg_log_prior_lmbda(shapes, z_prior, y, alpha, lmbda):
    # transform noise to latent variables
    z = T(shapes, y, lmbda)

    # prior (scaled by batch size)
    lp = jnp.sum(alpha * (z - z_prior) ** 2) / 2.

    return lp


# gradient of negative log prior
@partial(jit, static_argnums=(0,))
def grad_neg_log_prior_lmbda(shapes, z_prior, y, alpha, lmbda):
    return jacrev(neg_log_prior_lmbda, -1)(shapes, z_prior, y, alpha, lmbda)


# evaluate log likelihood
@partial(jit, static_argnums=(0,))
def neg_log_likelihood_lmbda(shapes, tf, inputs, s, m, y, nu2, sigma2, s_cap, m_cap, lmbda):
    # transform noise to latent variables
    z = T(shapes, y, lmbda)

    # measured values
    true = jnp.append(s[-1], m[-1])

    # integrate ode
    pred = jnp.nan_to_num(runODE(shapes, tf, inputs, s, m, s_cap, m_cap, z)[:len(true)])

    # error
    error = jnp.nan_to_num(true - pred)

    # predicted variance
    var = nu2 + sigma2 * jnp.clip(pred, 0., jnp.inf) ** 2

    # likelihood
    lp = jnp.sum(error ** 2 / var / 2. + jnp.log(var) / 2.)

    return lp


# gradient of log posterior w.r.t. variational parameters lmbda
@partial(jit, static_argnums=(0,))
def grad_neg_log_likelihood_lmbda(shapes, tf, inputs, s, m, y, nu2, sigma2, s_cap, m_cap, lmbda):
    return jacrev(neg_log_likelihood_lmbda, -1)(shapes, tf, inputs, s, m, y, nu2, sigma2, s_cap, m_cap, lmbda)


# evaluate gradient of log likelihood in batches
batch_grad_neg_log_likelihood_lmbda = pmap(grad_neg_log_likelihood_lmbda,
                                           in_axes=(None, 0, 0, 0, 0, None, None, None, None, None, None),
                                           static_broadcasted_argnums=(0,))


# evaluate log prior
@jit
def log_prior_z(z_prior, alpha, z):
    # prior
    lp = jnp.sum(alpha * (z - z_prior) ** 2) / 2.

    return lp


# gradient of log prior w.r.t. variational parameters
grad_log_prior_z = jit(jacfwd(log_prior_z, -1))


# evaluate log posterior
@partial(jit, static_argnums=(0,))
def log_likelihood_z(shapes, tf, inputs, s, m, nu2, sigma2, s_cap, m_cap, z):
    # measured values
    true = jnp.append(s[-1], m[-1])

    # integrate ode
    pred = jnp.nan_to_num(runODE(shapes, tf, inputs, s, m, s_cap, m_cap, z)[:len(true)])

    # error
    error = jnp.nan_to_num(true - pred)

    # predicted variance
    var = nu2 + sigma2 * jnp.clip(pred, 0., jnp.inf) ** 2

    # likelihood
    lp = jnp.sum(error ** 2 / var / 2. + jnp.log(var) / 2.)

    return lp


# evaluate log likelihood in batches
batch_log_likelihood_z = pmap(log_likelihood_z,
                              in_axes=(None, 0, 0, 0, 0, None, None, None, None, None),
                              static_broadcasted_argnums=(0,))


# gradient of log posterior w.r.t. variational parameters lmbda
@partial(jit, static_argnums=(0,))
def grad_log_likelihood_z(shapes, tf, inputs, s, m, nu2, sigma2, s_cap, m_cap, z):
    return jacrev(log_likelihood_z, -1)(shapes, tf, inputs, s, m, nu2, sigma2, s_cap, m_cap, z)


# evaluate gradient of log likelihood in batches
batch_grad_log_likelihood_z = pmap(grad_log_likelihood_z,
                                   in_axes=(None, 0, 0, 0, 0, None, None, None, None, None),
                                   static_broadcasted_argnums=(0,))


# evaluate rmse
@partial(jit, static_argnums=(0,))
def root_mean_squared_error(shapes, tf, inputs, s, m, s_cap, m_cap, z):
    # measured values
    true = jnp.append(s[-1], m[-1])

    # integrate ode
    pred = jnp.nan_to_num(runODE(shapes, tf, inputs, s, m, s_cap, m_cap, z))

    # error
    error = jnp.nan_to_num(true - pred)

    # sum of squares error
    # return jnp.sum(error ** 2) / 2.

    # root mean square error
    return jnp.sqrt(jnp.mean(error ** 2))


# gradient of rmse
@partial(jit, static_argnums=(0,))
def grad_root_mean_squared_error(shapes, tf, inputs, s, m, s_cap, m_cap, z):
    return jacrev(root_mean_squared_error, -1)(shapes, tf, inputs, s, m, s_cap, m_cap, z)
