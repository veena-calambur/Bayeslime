"""
A class that contains the bayesian linear regression used in the model.
"""
from scipy.stats import invgamma 
from scipy.stats import multivariate_normal
import numpy as np

# consider adding lazy evaluation for credible intervals 

class BayesLR:
	def __init__(self, percent=95, prior=None):	
		if prior is not None:
			raise NameError("Currently only support uninformative prior, set to None plz.")
	
		self.percent = percent

	def fit(self, xtrain, ytrain, sample_weight=weights):
		"""
		
		Fit the bayesian linear regression.

		args:
			TODO

		"""

		# TODO check types of input

		# Equation 5 in paper
		diag_pi_z = np.zeros((len(weights), len(weights)))
		np.fill_diagonal(diag_pi_z, weights)

		# probably could make this faster
		V_Phi = np.linalg.inv(xtrain.transpose().dot(diag_pi_z).dot(xtrain) \
						+ np.eye(xtrain.shape[1]))

		Phi_hat = V_Phi.dot(xtrain.transpose()).dot(diag_pi_z).dot(ytrain)

		N = xtrain.shape[0]
		Y_m_Phi_hat = ytrain - Z.dot(Phi_hat)

		s_2 = (1.0 / N) * (Y_m_Phi_hat.dot(diag_pi_z).dot(Y_m_Phi_hat) \
					 + Phi_hat.transpose().dot(Phi_hat))

		# set score to s_2
		self.score = s_2

		self.s_2 = s_2
		self.N = N
		self.V_Phi = V_Phi
		self.Phi_hat = Phi_hat

		self.creds = get_creds(self, percent=self.percent)

		self.crit_params = {
			"s_2": self.s_2,
			"N": self.N,
			"V_Phi": self.V_Phi,
			"Phi_hat": self.Phi_hat,
			"creds": self.creds
		}

	def get_creds(self, percent=95, n_samples=10_000):
		"""
		Get the credible intervals.

		args:
			percent: the percent cutoff for the credible interval, i.e., 95 is 95% credible interval

		"""

		samples = self.draw_posterior_samples(n_samples)
		creds = np.percentile(np.abs(samples - self.Phi_hat),
							  percent,
							  axis=0)

		return creds

	def draw_posterior_samples(self, num_samples):
		"""
		Sample from the posterior.
		
		args:
			num_samples: number of samples to draw from the posterior

		"""

		sigma_2 = invgamma.rvs(N / 2, scale=(N * s_2) / 2, size=num_samples)

		phi_samples = []

		for sig in sigma_2:
			sample = multivariate_normal.rvs(mean=self.Phi_hat, 
											 cov=self.V_Phi * sig, 
											 size=1)
			phi_samples.append(sample)

		phi_samples = np.vstack(phi_samples)

		return phi_samples

