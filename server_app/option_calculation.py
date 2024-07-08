import logging
from typing import List, Dict, Union


def _calculate_iron_condor(call_spread: Dict, put_spread: Dict) -> Dict:
    # ensure both spread have the same symbol
    if call_spread['Symbol'] != put_spread['Symbol']:
        raise ValueError('Symbol mismatch')

    # prefix for call and put spreads to distinct both in the merged data
    call_spread_prefixed = {f"Call Spread {key}": value for key, value in call_spread.items()}
    put_spread_prefixed = {f"Put Spread {key}": value for key, value in put_spread.items()}

    # Merge the dictionaries
    data = {**call_spread_prefixed, **put_spread_prefixed}

    # add Symbol
    data["Symbol"] = call_spread['Symbol']

    # add None for each new entry to the dict when any data is None
    if any(value is None for value in data.values()):
        logging.warning(f"Data {data} has None Values. All calculations will be set to None!")
        data["Max Profit"] = None
        data["BPR"] = None
        data["Profit/Risk"] = None
        data["Spread Theta"] = None
        data["JMS"] = None
        data["JMS Kelly"] = None
        return data

    # calculate iron condor values
    data["Max Profit"] = call_spread['Max Profit'] + put_spread['Max Profit']
    data["BPR"] = min(call_spread["Spread Width"], put_spread["Spread Width"]) * 100 - data["Max Profit"]
    data["Profit/Risk"] = round(data["Max Profit"] / data["BPR"], 2)
    data["Theta"] = call_spread["Spread Theta"] + put_spread["Spread Theta"]
    data["JMS"] = _calc_JMS(data, is_iron_condor=True)
    data["JMS Kelly"] = _calc_JMS_kelly_criterion(data, is_iron_condor=True)

    return data

def get_iron_condor_data(call_data: List[Dict], put_data: List[Dict]) -> List[Dict]:
    # ensure both list have the same length
    if len(call_data) != len(put_data):
        raise ValueError("Length of call data and put data must match")

    data = []

    for i in range(len(call_data)):
        call_spread = call_data[i]
        put_spread = put_data[i]
        data.append(_calculate_iron_condor(call_spread, put_spread))

    return data

def put_spread_calculations(data: List[Dict[str, Union[int, float, str, None]]]) -> List[Dict]:
    for spread in data:
        # add None for each new entry to the dict
        if any(value is None for value in spread.values()):
            logging.warning(f"Spread {spread} has None Values. All calculations will be set to None!")
            spread["Max Profit"] = None
            spread["BPR"] = None
            spread["Profit/Risk"] = None
            spread["Spread Theta"] = None
            spread["JMS"] = None
            spread["JMS Kelly"] = None

        # calculate the new entrys for the dict
        else:
            spread["Max Profit"] = round((spread['Short Last Price'] - spread["Long Last Price"]) * 100, 2)
            spread["BPR"] = round(spread["Spread Width"] * 100 - spread["Max Profit"], 2)
            spread["Profit/Risk"] = round(spread["Max Profit"] / spread["BPR"], 2)
            spread["Spread Theta"] = round((abs(spread["Short Theta"]) - abs(spread["Long Theta"])) * -1, 2)
            spread["JMS"] = _calc_JMS(spread)
            spread["JMS Kelly"] = _calc_JMS_kelly_criterion(spread)
    return data

def _calc_JMS_preparatory_values(spread, is_iron_condor=False):
    prep_values = {}

    # the iron condor win probability is 1 - short call delta - short put delta.
    # Where both deltas are absolute.
    if is_iron_condor:
        prep_values["win_prob"] = 1 - abs(spread['Call Spread Short Delta']) - abs(spread['Put Spread Short Delta'])
    else:
        prep_values["win_prob"] = 1 - abs(spread['Short Delta'])

    prep_values["loss_prob"] = 1 - prep_values["win_prob"]
    prep_values["tp_goal"] = 0.6
    prep_values["mental_stop"] = 2
    prep_values["potential_win"] = spread["Max Profit"] * prep_values["tp_goal"]
    prep_values["potential_loss"] = spread["Max Profit"] * prep_values["mental_stop"]
    prep_values["win_fraction"] = prep_values["potential_win"] / spread["BPR"]
    prep_values["loss_fraction"] = prep_values["potential_loss"] / spread["BPR"]
    prep_values["expected_win_value"] = prep_values["win_prob"] * prep_values["potential_win"]
    prep_values["expected_loss_value"] = prep_values["potential_loss"] * prep_values["loss_prob"]

    return prep_values

def _calc_JMS(spread, is_iron_condor=False):
    """
    Calulates Joachims Milchmädchenrechnungs Score.
    Based on the Delta of the short option a mental stop loss and a take profit goal.
    It's a kind of simple expected Value. Further it is normalized with the BPR.
    :param spread:
    :return:
    """
    prep_values = _calc_JMS_preparatory_values(spread, is_iron_condor)
    win_value = prep_values["expected_win_value"]
    loss_value = prep_values["expected_loss_value"]
    jms = (win_value - loss_value) / spread["BPR"] * 100  # *100 for better readability
    jms = round(jms, 2)
    return jms

def _calc_JMS_kelly_criterion(spread, is_iron_condor=False):
    """
    Calculates the kelly criterion based on the values of the JMS calculations.

    For more details, see https://en.wikipedia.org/wiki/JMS_kelly_criterion
    """
    prep_values = _calc_JMS_preparatory_values(spread, is_iron_condor)
    p = prep_values["win_prob"]
    l = prep_values["loss_fraction"]
    q = prep_values["loss_prob"]
    g = prep_values["win_fraction"]
    jms_kelly = (p / l) - (q / g)
    jms_kelly = round(jms_kelly, 2)

    return jms_kelly