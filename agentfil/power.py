class Power:
    def __init__(self, amount_bytes, power_type, duration_days):
        self.amount_bytes = amount_bytes
        self.power_type = power_type
        self.duration = duration_days

    def __add__(self, other):
        assert self.power_type == other.power_type, "Cannot add power of different types"
        return Power(self.amount_bytes + other.amount_bytes, self.power_type, self.duration)

def cc_power(amount_bytes, duration_days=360):
    return Power(amount_bytes, 'cc', duration_days)

def deal_power(amount_bytes, duration_days=360):
    return Power(amount_bytes, 'deal', duration_days)
