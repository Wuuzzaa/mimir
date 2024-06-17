import logging
from typing import List, Dict, Union

def put_spread_calculations(data: List[Dict[str, Union[int, float, str, None]]]) -> List[Dict[str, Union[int, float, str, None]]]:
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

def _calc_JMS_preparatory_values(spread):
    prep_values = {}
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

def _calc_JMS(spread):
    """
    Calulates Joachims Milchmädchenrechnungs Score.
    Based on the Delta of the short option a mental stop loss and a take profit goal.
    It's a kind of simple expected Value. Further it is normalized with the BPR.
    :param spread:
    :return:
    """
    prep_values = _calc_JMS_preparatory_values(spread)
    win_value = prep_values["expected_win_value"]
    loss_value = prep_values["expected_loss_value"]
    jms = (win_value - loss_value) / spread["BPR"] * 100  # *100 for better readability
    jms = round(jms, 2)
    return jms

def _calc_JMS_kelly_criterion(spread):
    """
    Calculates the kelly criterion based on the values of the JMS calculations.

    For more details, see https://en.wikipedia.org/wiki/JMS_kelly_criterion
    """
    prep_values = _calc_JMS_preparatory_values(spread)
    p = prep_values["win_prob"]
    l = prep_values["loss_fraction"]
    q = prep_values["loss_prob"]
    g = prep_values["win_fraction"]
    jms_kelly = (p / l) - (q / g)
    jms_kelly = round(jms_kelly, 2)

    # debug
    # print("#"*80)
    # for key, value in spread.items():
    #     print(f'{key}: {value}')
    #
    # for key, value in prep_values.items():
    #     print(f'{key}: {value}')
    #
    # print(f"jms_kelly: {jms_kelly}")

    return jms_kelly