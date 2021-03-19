"""
A class that contains the bayesian linear regression used in the model.
"""
from scipy.stats import invgamma 
from scipy.stats import multivariate_normal
import numpy as np
import collections

# consider adding lazy evaluation for credible intervals 

class BayesLR:
	def __init__(self, percent=95, prior=None):	
		if prior is not None:
			raise NameError("Currently only support uninformative prior, set to None plz.")
	
		self.percent = percent

	def fit(self, xtrain, ytrain, sample_weight):
		"""
		
		Fit the bayesian linear regression.

		args:
			xtrain: the training data
			ytrain: the training labels
			sample_weight: the weights for fitting the regression

		"""

		# store weights
		weights = sample_weight

		# add intercept
		xtrain = np.concatenate((np.ones(xtrain.shape[0])[:,None], xtrain), axis=1)

		# Equation 5 in paper
		diag_pi_z = np.zeros((len(weights), len(weights)))
		np.fill_diagonal(diag_pi_z, weights)

		V_Phi = np.linalg.inv(xtrain.transpose().dot(diag_pi_z).dot(xtrain) \
						+ np.eye(xtrain.shape[1]))

		Phi_hat = V_Phi.dot(xtrain.transpose()).dot(diag_pi_z).dot(ytrain)

		N = xtrain.shape[0]
		Y_m_Phi_hat = ytrain - xtrain.dot(Phi_hat)

		s_2 = (1.0 / N) * (Y_m_Phi_hat.dot(diag_pi_z).dot(Y_m_Phi_hat) \
					 + Phi_hat.transpose().dot(Phi_hat))

		# set score to s_2
		self.score = s_2

		self.s_2 = s_2
		self.N = N
		self.V_Phi = V_Phi
		self.Phi_hat = Phi_hat
		self.coef_ = Phi_hat[1:]
		self.intercept_ = Phi_hat[0]

		self.creds, self.confident = self.get_creds(percent=self.percent)
		self.creds = self.creds[1:]

		self.crit_params = {
			"s_2": self.s_2,
			"N": self.N,
			"V_Phi": self.V_Phi,
			"Phi_hat": self.Phi_hat,
			"creds": self.creds
		}

	def get_ranking_frequency(self, rankings):

		tup_rankings = [tuple(r) for r in rankings]
		return collections.Counter(tup_rankings)


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

		samples = samples[:,1:]
		rankings = np.argsort(np.abs(samples),axis=1)
		
		ranking_frequency = self.get_ranking_frequency(rankings)
		
		if ranking_frequency.most_common()[0][1] > .95 * samples.shape[0]:
			confident=True
		else:
			confident=False

		return creds, confident

	def draw_posterior_samples(self, num_samples):
		"""
		Sample from the posterior.
		
		args:
			num_samples: number of samples to draw from the posterior

		"""

		sigma_2 = invgamma.rvs(self.N / 2, scale=(self.N * self.s_2) / 2, size=num_samples)

		phi_samples = []

		for sig in sigma_2:
			sample = multivariate_normal.rvs(mean=self.Phi_hat, 
											 cov=self.V_Phi * sig, 
											 size=1)
			phi_samples.append(sample)

		phi_samples = np.vstack(phi_samples)

		return phi_samples

