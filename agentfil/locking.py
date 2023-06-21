from .constants import GIB

def noop(*args, **kwargs):
    return {}

def spec_onboard_ratio(day_added_qa_power, total_qa_power, baseline_power):
    return day_added_qa_power / max(total_qa_power, baseline_power)

def no_baseline_onboard_ratio(day_added_qa_power, total_qa_power, baseline_power):
    return day_added_qa_power / total_qa_power

def fofr_cap_onboard_ratio_kwargs_extract(filecoin_df_day, lock_target):
    return {
        'day_simple_reward': filecoin_df_day['day_simple_reward'],
        'target_lock': lock_target,
        'circ_supply': filecoin_df_day['circ_supply'],
    }

def fofr_cap_onboard_ratio(day_added_qa_power, total_qa_power, baseline_power,
                           day_simple_reward=None, target_lock=None, circ_supply=None):
    # 6 * (1 - 1/(2**⅙)) / log(2) = 0.944 
    simple_decay_factor_1y = 0.944  # TODO: generalize
    num = day_added_qa_power
    
    expected_simplemint_rewards_for_power_1y = simple_decay_factor_1y * 365 * (day_simple_reward*(day_added_qa_power/total_qa_power))
    c = (target_lock * circ_supply * day_added_qa_power)/expected_simplemint_rewards_for_power_1y
    # NOTE: numerator for C gets canceled out and what remains is expected_simplemint_rewards_for_power_1y for pledge, which caps FoFR=100%
    denom = min(max(total_qa_power, baseline_power), c)
    
    # return num/denom
    return (num/denom)

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