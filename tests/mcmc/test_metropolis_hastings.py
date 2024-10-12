import unittest
from unittest import TestCase

from python.ml.metropolis_hastings import MetropolisHastings
from python.ml.proposal_distribution import ProposalBeta, ProposalDistribution
from typing import AnyStr


class TestMetropolisHastings(TestCase):

    @unittest.skip
    def test_mh_sample_high_burn_in(self):
        num_iterations = 12000
        burn_in_ratio = 0.2
        sigma_delta = 0.5
        theta0 = 0.8

        alpha = 4
        beta = 6
        num_trails = 96
        successes = 24
        normal_McMc = ProposalBeta(alpha, beta, num_trails, successes)
        TestMetropolisHastings.execute_metropolis_hastings(
            normal_McMc,
            num_iterations,
            burn_in_ratio,
            sigma_delta,
            theta0,
            f"Beta({alpha}, {beta}) burn_in ratio:{burn_in_ratio}, Initial theta: {theta0} Delta {sigma_delta}"
        )


    def test_mh_sample_no_burn_in(self):
        num_iterations = 12000
        burn_in_ratio = 0.2
        sigma_delta = 0.4
        theta0 = 0.8

        alpha = 4
        beta = 6
        num_trails = 100
        h = 10
        proposal = ProposalBeta(alpha, beta, num_trails, h)

        TestMetropolisHastings.execute_metropolis_hastings(
            proposal,
            num_iterations,
            burn_in_ratio,
            sigma_delta,
            theta0,
            f"Beta({alpha}, {beta}) burn_in ratio:{burn_in_ratio}, Initial theta: {theta0}"
        )


    @unittest.skip
    def test_mh_sample_burn_in_beta(self):
        num_iterations = 12000
        burn_in_ratio = 0.1
        sigma_delta = 0.4
        theta0 = 0.95

        alpha = 4
        beta = 2
        num_trails = 96
        h = 10
        normal_McMc = ProposalBeta(alpha, beta, num_trails, h)

        TestMetropolisHastings.execute_metropolis_hastings(
            normal_McMc,
            num_iterations,
            burn_in_ratio,
            sigma_delta,
            theta0,
            f"Beta({alpha}, {beta}) burn_in ratio:{burn_in_ratio}, Initial theta: {theta0}"
        )

    @staticmethod
    def execute_metropolis_hastings(
            proposed_distribution: ProposalDistribution,
            num_iterations: int,
            burn_in_ratio: float,
            sigma_delta: float,
            theta0: float,
            description: AnyStr):
        metropolis_hastings = MetropolisHastings(proposed_distribution, num_iterations, burn_in_ratio, sigma_delta)

        theta_history, success_rate = metropolis_hastings.sample(theta0)
        theta_history_str = str(theta_history)
        print(f'{description} Success rate {success_rate}')


if __name__ == '__main__':
    unittest.main()



