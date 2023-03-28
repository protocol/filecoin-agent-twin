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
                 min_discount_rate_pct=0, max_discount_rate_pct=5, start_discount_rate_pct=2.5, 
                 behavior='constant', seed=1234):
        self.model = filecoin_model

        self.min_discount_rate_pct = min_discount_rate_pct
        self.max_discount_rate_pct = max_discount_rate_pct
        self.behavior = behavior
        
        self.discount_rate = start_discount_rate_pct
        self.rng = np.random.default_rng(seed)

        self.validate_behavior()

    def validate_behavior(self):
        if self.behavior not in ['constant', 'random_walk', 'adaptive']:
            raise ValueError(f'Invalid behavior: {self.behavior}')

    def step(self, circ_supply=None, market_cap=None):
        if self.behavior == 'constant':
            pass
        elif self.behavior == 'random_walk':
            self.discount_rate = np.clip(self.rng.normal(self.discount_rate, 0.1), self.min_discount_rate, self.max_discount_rate)
        elif self.behavior == 'adaptive':
            # TODO: define some sort of behavior based on network econometrics
            pass

        # add into the filecoin_df
        self.model.filecoin_df.loc[self.model.current_day, 'discount_rate_pct'] = self.discount_rate