class Power:
    def __init__(self, amount_bytes, power_type, duration):
        self.amount_bytes = amount_bytes
        self.power_type = power_type
        self.duration = duration

def cc_power(amount_bytes, duration=None):
    return Power(amount_bytes, 'cc', duration)

def deal_power(amount_bytes, duration=None):
    return Power(amount_bytes, 'deal', duration)
