class Power:
    def __init__(self, pib, power_type, duration_days):
        self.pib = pib
        self.power_type = power_type
        self.duration = duration_days

    def __add__(self, other):
        assert self.power_type == other.power_type, "Cannot add power of different types"
        return Power(self.pib + other.pib, self.power_type, self.duration)

    def __repr__(self) -> str:
        return 'Power(%d, %s, %d)' % (self.pib, self.power_type, self.duration)

def cc_power(pib, duration_days=360):
    return Power(pib, 'cc', duration_days)

def deal_power(pib, duration_days=360):
    return Power(pib, 'deal', duration_days)
