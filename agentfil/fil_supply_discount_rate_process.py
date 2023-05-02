import numpy as np

class FILSupplyDiscountRateProcess:
    """
    This class generates a stochastic process for the supply discount rate. This is used
    by agents to borrow FIL for onboarding or renewing power onto the network.

    In general, from a central bank perspective, discount rates are used to control the 
    money supply. When a central bank wants to reduce inflation, it will raise the discount rate.
    Conversely, when a central bank wants to stimulate growth, it will lower the discount rate.

    In this module, we can simulate a variety of discount rate behaviors. The default behavior
    is a constant discount rate. The other behaviors are:
    - random_walk: the discount rate will randomly walk up or down by a small amount each day
    - adaptive: the discount rate will be adjusted based on some sort of econometric model

    """
    def __init__(self, filecoin_model,
                 min_discount_rate_pct=25, max_discount_rate_pct=150, start_discount_rate_pct=25, 
                 behavior='constant', behavior_kwargs=None, seed=1234):
        self.model = filecoin_model

        self.min_discount_rate_pct = min_discount_rate_pct
        self.max_discount_rate_pct = max_discount_rate_pct
        self.behavior = behavior
        self.behavior_kwargs = {} if behavior_kwargs is None else behavior_kwargs
        
        self.discount_rate_pct = start_discount_rate_pct
        self.rng = np.random.default_rng(seed)

        self.validate_behavior()

    def validate_behavior(self):
        if self.behavior not in ['constant', 'random_walk', 'linear-adaptive', 'sigmoid-adaptive']:
            raise ValueError(f'Invalid behavior: {self.behavior}')
        
    def alter_behavior(self, new_behavior):
        self.behavior = new_behavior
        self.validate_behavior()

    def step(self, circ_supply=None, market_cap=None):
        if self.behavior == 'constant':
            pass
        elif self.behavior == 'random_walk':
            # TODO: random walk variance should be configurable
            if 'random_walk_variance' in self.behavior_kwargs:
                random_walk_variance = self.behavior_kwargs['random_walk_variance']
            else:
                # TODO: raise warning
                random_walk_variance = (self.max_discount_rate_pct - self.min_discount_rate_pct) / 10.
            self.discount_rate_pct = np.clip(self.rng.normal(self.discount_rate_pct, random_walk_variance), self.min_discount_rate_pct, self.max_discount_rate_pct)
        elif self.behavior == 'linear-adaptive':
            # TODO: define some sort of behavior based on network econometrics
            if circ_supply is None:
                raise ValueError('circ_supply must be provided to adaptive discount rate')
            # create a linear mapping from circulating supply to discount rate
            min_circ_supply = 0.
            max_circ_supply = 1.1e9
            m = (self.min_discount_rate_pct - self.max_discount_rate_pct) / (max_circ_supply - min_circ_supply)
            self.discount_rate_pct = m * circ_supply + self.max_discount_rate_pct
        elif self.behavior == 'sigmoid-adaptive':
            if circ_supply is None:
                raise ValueError('circ_supply must be provided to adaptive discount rate')
            raise NotImplementedError('sigmoid-adaptive discount rate not implemented')

        # add into the filecoin_df
        self.model.filecoin_df.loc[self.model.current_day, 'discount_rate_pct'] = self.discount_rate_pct