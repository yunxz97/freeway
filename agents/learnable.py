import factors.learnable as freeway_factors

from agents.base import FreewayBaseAgent, variable_range

class FreewayLearnableAgent(FreewayBaseAgent):
    def __init__(self, **kwargs):
        super().__init__(variable_range=variable_range, factors=freeway_factors, **kwargs)