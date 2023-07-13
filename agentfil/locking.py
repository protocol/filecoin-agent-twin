from .constants import GIB
from datetime import date

# Updates to support ABM
def noop(*args, **kwargs):
    return {}

def spec_onboard_ratio(day_added_qa_power, total_qa_power, baseline_power):
    # print('spec onboard ratio')
    return day_added_qa_power / max(total_qa_power, baseline_power)

def no_baseline_onboard_ratio_kwargs_extract(current_date, filecoin_df_day, lock_target):
    return {
        'current_date': current_date,
        'activation_date': date(2023, 5, 20),
    }

def no_baseline_onboard_ratio(day_added_qa_power, total_qa_power, baseline_power,
                              current_date = None, activation_date = None):
    if current_date >= activation_date:
        return day_added_qa_power / total_qa_power
    else:
        return spec_onboard_ratio(day_added_qa_power, total_qa_power, baseline_power)

def fofr_cap_onboard_ratio_kwargs_extract(current_date, filecoin_df_day, lock_target):
    return {
        'current_date': current_date,
        'activation_date': date(2023, 5, 20),  # defaults to simulation start
        'day_simple_reward': filecoin_df_day['day_simple_reward'],
        'target_lock': lock_target,
        'circ_supply': filecoin_df_day['circ_supply'],
    }

def fofr_cap_onboard_ratio(day_added_qa_power, total_qa_power, baseline_power,
                           current_date = None, activation_date = None,
                           day_simple_reward=None, target_lock=None, circ_supply=None):
    if current_date >= activation_date:
        # 6 * (1 - 1/(2**⅙)) / log(2) = 0.944 
        simple_decay_factor_1y = 0.944  # TODO: generalize
        num = day_added_qa_power
        
        expected_simplemint_rewards_for_power_1y = simple_decay_factor_1y * 365 * (day_simple_reward*(day_added_qa_power/total_qa_power))
        c = (target_lock * circ_supply * day_added_qa_power)/expected_simplemint_rewards_for_power_1y
        # NOTE: numerator for C gets canceled out and what remains is expected_simplemint_rewards_for_power_1y for pledge, which caps FoFR=100%
        denom = min(max(total_qa_power, baseline_power), c)
        
        # return num/denom
        return (num/denom)
    else:
        return spec_onboard_ratio(day_added_qa_power, total_qa_power, baseline_power)

def min_pledge_onboard_ratio_kwargs_extract(current_date, filecoin_df_day, lock_target):
    return {
        'current_date': current_date,
        'activation_date': date(2023, 5, 20),  # defaults to simulation start
        'target_lock': lock_target,
        'circ_supply': filecoin_df_day['circ_supply'],
        'min_pledge_per_32QAP_sector': 0.1,
    }

def min_pledge_onboard_ratio(day_added_qa_power, total_qa_power, baseline_power,
                             current_date = None, activation_date = None,
                             target_lock=None, circ_supply=None, min_pledge_per_32QAP_sector=0.1):
    """
    Returns normalized qap growth value such that the consensus pledge will be equal to min_pledge
    min_pledge: minimum pledge value in FIL per sector
    # TODO: how to pass in min_pledge properly?
    """
    if current_date >= activation_date:
        # print('calling min_pledge_onboard_ratio')
        num_sectors_added = day_added_qa_power / (32 * GIB)
        ngq =  1./(target_lock * circ_supply)*min_pledge_per_32QAP_sector*num_sectors_added
    else:
        # print('calling spec onboard ratio')
        ngq = spec_onboard_ratio(day_added_qa_power, total_qa_power, baseline_power)
    # print(day_added_qa_power, num_sectors_added, ngq)
    return ngq

def compute_new_pledge_for_added_power(
    day_network_reward: float,
    prev_circ_supply: float,
    day_added_qa_power: float,
    total_qa_power: float,
    baseline_power: float,
    lock_target: float,
    onboard_ratio_callable: callable = spec_onboard_ratio,
    onboard_ratio_callable_kwargs: dict = {},
) -> float:
    # storage collateral
    storage_pledge = 20.0 * day_network_reward * (day_added_qa_power / total_qa_power)
    # consensus collateral
    # normalized_qap_growth = day_added_qa_power / max(total_qa_power, baseline_power)
    normalized_qap_growth = onboard_ratio_callable(day_added_qa_power, total_qa_power, baseline_power, **onboard_ratio_callable_kwargs)
    consensus_pledge = max(lock_target * prev_circ_supply * normalized_qap_growth, 0)
    # total added pledge
    added_pledge = storage_pledge + consensus_pledge

    pledge_cap = day_added_qa_power * 1.0 / GIB  # The # of bytes in a GiB (Gibibyte)
    return min(pledge_cap, added_pledge)

# Initial pledge collateral
def compute_day_delta_pledge(
    day_network_reward: float,
    prev_circ_supply: float,
    day_onboarded_qa_power: float,
    day_renewed_qa_power: float,
    total_qa_power: float,
    baseline_power: float,
    renewal_rate: float,
    scheduled_pledge_release: float,
    lock_target: float = 0.3,
    onboard_ratio_callable: callable = spec_onboard_ratio,
    onboard_ratio_callable_kwargs: dict = {},
) -> float:
    onboards_delta = compute_new_pledge_for_added_power(
        day_network_reward,
        prev_circ_supply,
        day_onboarded_qa_power,
        total_qa_power,
        baseline_power,
        lock_target,
        onboard_ratio_callable=onboard_ratio_callable,
        onboard_ratio_callable_kwargs=onboard_ratio_callable_kwargs
    )
    renews_delta = compute_renewals_delta_pledge(
        day_network_reward,
        prev_circ_supply,
        day_renewed_qa_power,
        total_qa_power,
        baseline_power,
        renewal_rate,
        scheduled_pledge_release,
        lock_target,
        onboard_ratio_callable=onboard_ratio_callable,
        onboard_ratio_callable_kwargs=onboard_ratio_callable_kwargs
    )
    return onboards_delta + renews_delta

def compute_renewals_delta_pledge(
    day_network_reward: float,
    prev_circ_supply: float,
    day_renewed_qa_power: float,
    total_qa_power: float,
    baseline_power: float,
    renewal_rate: float,
    scheduled_pledge_release: float,
    lock_target: float,
    onboard_ratio_callable: callable = spec_onboard_ratio,
    onboard_ratio_callable_kwargs: dict = {},
) -> float:
    # Delta from sectors expiring
    expire_delta = -(1 - renewal_rate) * scheduled_pledge_release
    # Delta from sector renewing
    original_pledge = renewal_rate * scheduled_pledge_release
    new_pledge = compute_new_pledge_for_added_power(
        day_network_reward,
        prev_circ_supply,
        day_renewed_qa_power,
        total_qa_power,
        baseline_power,
        lock_target,
        onboard_ratio_callable=onboard_ratio_callable,
        onboard_ratio_callable_kwargs=onboard_ratio_callable_kwargs
    )
    renew_delta = max(0.0, new_pledge - original_pledge)
    # Delta for all scheduled sectors
    delta = expire_delta + renew_delta
    return delta

####################################################################################
# From MechaFIL

GIB = 2**30

# Block reward collateral
def compute_day_locked_rewards(day_network_reward: float) -> float:
    return 0.75 * day_network_reward


def compute_day_reward_release(prev_network_locked_reward: float) -> float:
    return prev_network_locked_reward / 180.0

####################################################################################

