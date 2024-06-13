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

        # calculate the new entrys for the dict
        else:
            spread["Max Profit"] = round((spread['Short Last Price'] - spread["Long Last Price"]) * 100, 2)
            spread["BPR"] = round(spread["Spread Width"] * 100 - spread["Max Profit"], 2)
            spread["Profit/Risk"] = round(spread["Max Profit"] / spread["BPR"], 2)
            spread["Spread Theta"] = round((abs(spread["Short Theta"]) - abs(spread["Long Theta"])) * -1, 2)
            spread["JMS"] = _calc_JMS(spread)
    return data

def _calc_JMS(spread):
    """
    Calulates Joachims Milchmädchenrechnungs Score.
    Based on the Delta of the short option a mental stop loss and a take profit goal. I
    t's a kind of simple expected Value. Further it is normalized with the BPR.
    :param spread:
    :return:
    """
    win_prob = 1 - abs(spread['Short Delta'])
    loss_prob = 1 - win_prob
    tp_goal = 0.6
    mental_stop = 2
    win_value = win_prob * spread["Max Profit"] * tp_goal
    loss_value = spread["Max Profit"] * mental_stop * loss_prob
    jms = (win_value - loss_value) / spread["BPR"] * 100  # *100 for better readability
    jms = round(jms, 2)
    return jms