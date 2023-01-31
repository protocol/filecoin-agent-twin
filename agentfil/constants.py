import datetime

NETWORK_START = datetime.date(2020, 10, 15)
NETWORK_DATA_START = datetime.date(2021, 3, 15)

EIB = 2**60
PIB = 2**50
TIB = 2 ** 40
GIB = 2 ** 30
SECTOR_SIZE = 32 * GIB

MIN_VALUE=1e-6

# TODO: is this reasonable? There should be some limitation based on the blockchain,
# but I'm not sure. If all of this is FIL+, this would mean that the maximum onboardable
# power per day is 25 * 10 = 250 PiB QAP, which seems super high.
MAX_DAY_ONBOARD_RBP_PIB=25

FIL_PLUS_MULTIPLER=10

MAX_SECTOR_DURATION_DAYS=1080  # equivalent to 3 years

MC_QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]