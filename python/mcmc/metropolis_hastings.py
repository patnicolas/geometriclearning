__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard Library imports
import logging
# 3rd Party imports
import numpy as np
# Library imports
from mcmc.mcmc import MCMC
from mcmc import MCMCException
import python
__all__ = ['MetropolisHastings']


class MetropolisHastings(MCMC):
    """
        Implementation of Metropolis-Hastings Markov Chain Monte Carlo method
    """
    from proposal_distribution import ProposalDistribution

    default_sigma_delta = 0.2

    def __init__(self,
                 proposal: ProposalDistribution,
                 num_iterations: int,
                 burn_in_ratio: float,
                 sigma_delta: float = default_sigma_delta):
        """
        Constructor for the Metropolis-Hastings algorithm

        @param proposal: Proposal distribution
        @type proposal: ProposalDistribution
        @param num_iterations: Number of iterations for the random walk
        @type num_iterations: int
        @param burn_in_ratio: Percentage of number of iterations dedicated to burn-in steps
        @type burn_in_ratio: float
        @param sigma_delta: Covariance or standard deviation used for each step theta -> theta_star
        @type sigma_delta: float
        """
        if num_iterations < 2 or num_iterations > 100000:
            raise ValueError(f'Number of iterations {num_iterations} is out of bounds [2, 10000]')
        if sigma_delta <= 0.0 or sigma_delta >= 1.0:
            raise ValueError(f'Number of iterations {num_iterations} is out of bounds [2, 10000]')
        if burn_in_ratio < 0.0 or burn_in_ratio > 0.5:
            raise ValueError(f'Burn-in ratio {burn_in_ratio} is out of bounds [0.0, 0.5]')
        burn_ins = int(num_iterations*burn_in_ratio)
        assert num_iterations > burn_ins, \
            f'Number of iterations {num_iterations} should be > number of burn-ins {burn_ins}'

        super(MetropolisHastings, self).__init__()
        self.proposal = proposal
        self.num_iterations = num_iterations
        self.sigma_delta = sigma_delta
        self.burn_ins = burn_ins

    def sample(self, theta_0: float) -> (np.array, float):
        """
        Generate a new sample for the Metropolis-Hastings search
        @param theta_0:  Initial value for the parameters in Markov chain
        @type theta_0: float
        @return: Tuple of history of theta values after burn-in and ratio of number of accepted
                 new theta values over total number of iterations after burn-ins
        @rtype: tuple[Numpy array float]
        """
        num_valid_thetas = self.num_iterations - self.burn_ins
        theta_walk = np.zeros(num_valid_thetas)
        accepted_count = 0
        theta = theta_0    # 0.25
        theta_walk[0] = theta

        j = 0
        for i in range(self.num_iterations):
            theta_star = self.proposal.step(theta, self. sigma_delta)

            try:
                # Computes the prior for the current and next sample
                cur_prior = self.proposal.log_prior(theta)
                new_prior = self.proposal.log_prior(theta_star)

                # We only consider positive and non-null prior probabilities
                if cur_prior > 0.0 and new_prior > 0.0:
                    # We use the logarithm value for the comparison to avoid underflow
                    cur_log_posterior = self.proposal.log_posterior(theta, cur_prior)
                    new_log_posterior = self.proposal.log_posterior(theta_star, new_prior)

                    # Apply the selection criteria
                    if MetropolisHastings.__acceptance_rule(cur_log_posterior, new_log_posterior):
                        theta = theta_star
                        if i > self.burn_ins:
                            accepted_count += 1
                            theta_walk[j + 1] = theta_star
                            j += 1
                    else:
                        if i > self.burn_ins:
                            theta_walk[j + 1] = theta_walk[j]
                            j += 1
            except (ArithmeticError, ValueError, IndexError) as e:
                logging.error(e)
                raise MCMCException(e)

        return theta_walk, float(accepted_count) / num_valid_thetas

        # --------------  Supporting methods -----------------------
    @staticmethod
    def __acceptance_rule(currentValue: float, newValue: float) -> bool:
        """
        Implement the acceptance/rejection criteria for a new sample
        @param currentValue: Value of the last sample
        @type currentValue: float
        @param newValue: Value of the new sample
        @type newValue: float
        @return: True if new sample is accepted, False otherwise
        @rtype: float
        """
        if newValue < currentValue:
            raise ValueError(f'New value {newValue} should be >  current value {currentValue}')

        residual = newValue - currentValue
        return True if newValue > currentValue else np.random.uniform(0, 1) < np.exp(residual)