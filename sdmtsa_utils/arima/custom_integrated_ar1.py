import numpy as np
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.arima.model import ARIMA


class CustomIntegratedAR1(MLEModel):
    def __init__(self, endog):
        self.lags = [1, 2, 24, 25, 26, 24*7, 24*7+1, 24*7+2, 24*8, 24*8+1,
                     24*8+2]    # the selected lags
        self.p = self.lags[-1]  # AR order
        self.q = 0              # MA order
        self.r = np.max((self.p, self.q + 1))

        ss_spec = self.init_ss_matrices()
        MLEModel.__init__(self, endog,  k_states=self.r, k_posdef=1, **ss_spec,
                          initialization='approximate_diffuse')
        # used to get initial params for the optimizer
        self.ar1_mod = ARIMA(endog.diff(1).diff(24).diff(24*7),
                             order=(1, 0, 0))

    @property
    def param_names(self):
        return ['const', 'phi', 'sigma2']

    @property
    def start_params(self):
        return self.ar1_mod.start_params

    def transform_params(self, unconstrained):
        """
        We constraint the last parameter ('sigma2') to be positive,
        because it's a variance.
        """
        constrained = unconstrained.copy()
        constrained[-1] = constrained[-1]**2
        return constrained

    def untransform_params(self, constrained):
        """
        Invert transformation for the variance
        """
        unconstrained = constrained.copy()
        unconstrained[-1] = unconstrained[-1]**0.5
        return unconstrained

    def update(self, params, **kwargs):
        params = MLEModel.update(self, params, **kwargs)

        # update state space representation
        self['obs_intercept'][0] = params[0]
        self['transition'] = self.update_transition_matrix(params[1])
        self['state_cov'][0, 0] = params[2]

    def update_transition_matrix(self, phi):
        coeff = [1+phi, -phi, 1, -(1+phi), phi, 1, -(1+phi), phi, -1,
                 1+phi, -phi]
        first_row = np.zeros(self.r).reshape(1, -1)
        for i, lag in enumerate(self.lags):
            first_row[0, lag-1] = coeff[i]
        identiry_mat = np.identity(self.r - 1)
        zero_col = np.zeros(self.r - 1).reshape(-1, 1)
        other_rows = np.concatenate((identiry_mat, zero_col), axis=1)

        transition_matrix = np.concatenate((first_row, other_rows), axis=0)
        return transition_matrix

    def init_ss_matrices(self, init_phi=0):
        transition = self.update_transition_matrix(init_phi)

        design = np.zeros(self.r)
        design[0] = 1
        design = design.reshape(1, -1)

        obs_intercept = np.ones(1)

        selection = np.zeros(self.r)
        selection[0] = 1
        selection = selection.reshape(-1, 1)

        state_intercept = np.zeros(self.r)
        state_intercept.reshape(-1, 1)

        obs_cov = np.zeros((1, 1))
        state_cov = np.ones((1, 1))
        # build state space specification dictionary
        # will be used to pass keyword args to the state stpace form
        # constructor
        ss_spec = dict()
        ss_spec['transition'] = transition
        ss_spec['design'] = design
        ss_spec['obs_intercept'] = obs_intercept
        ss_spec['selection'] = selection
        ss_spec['state_intercept'] = state_intercept
        ss_spec['obs_cov'] = obs_cov
        ss_spec['state_cov'] = state_cov
        return ss_spec
