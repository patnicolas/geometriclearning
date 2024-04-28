import unittest

from util.plottermixin import PlotterParameters, PlotterMixin
from ml.proposal_distribution import ProposalBeta, ProposalDistribution
import numpy as np
from scipy import stats

class MyTestCase(unittest.TestCase):
    def test_beta_distribution(self):
        num_data_points = 1000
        alpha = 12
        beta = 10
        num_trails = 96
        h = 10

        plotting_params_1 = PlotterParameters(num_data_points, "x",  "Prior", "Beta distribution")
        x = np.linspace(0, 1, num_data_points)
        posterior = stats.beta(num_trails + alpha, num_trails - h + beta)
        y = posterior.pdf(x)
        PlotterMixin.single_plot_np_array(x, y, plotting_params_1)

    def test_beta_proposal(self):
        num_data_points = 200
        alpha = 12
        beta = 10
        num_trials = 96
        h = 10
        beta_proposal = ProposalBeta(alpha, beta, num_trials, h)
        x = np.linspace(0, 1, num_data_points)
        beta_values = [beta_proposal.log_prior(x) for x in  np.linspace(0, 1, num_data_points)]
        plotting_params = PlotterParameters(num_data_points, "x",  "Probability", "Beta Proposal Prior")
        PlotterMixin.single_plot_np_array(x, beta_values, plotting_params)




if __name__ == '__main__':
    unittest.main()
