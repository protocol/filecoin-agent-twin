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
# power per day is 75 * 10 = 750 PiB QAP, which seems super high.
DEFAULT_MAX_DAY_ONBOARD_RBP_PIB=75  # this is close to the historical max

# From Filecoin Spec:
# However, sector sealing time is estimated to take ~1.5 hours for a 32 GB sector on a machine 
# that meets minimum hardware requirements for storage providers.
# 1 GB = .000931 PiB.  32 GB = .029 PiB
# .029 * 24/1.5 ~ .5 PiB/day
DEFAULT_MIN_SEALING_THROUGHPUT_PIB=0.5
DEFAULT_MAX_SEALING_THROUGHPUT_PIB=25
MIN_SECTORS_ONBOARD=1

FIL_PLUS_MULTIPLER=10

MAX_SECTOR_DURATION_DAYS=1080  # equivalent to 3 years

MC_QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]