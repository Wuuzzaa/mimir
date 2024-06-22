def calculate_yield(profit: float, bpr: float) -> float:
    return profit / bpr

def calculate_annualized_yield(profit: float, days_trade_open: int, bpr: float, fees: float) -> float:
    """

    :param profit:
    :param days_trade_open:
    :param bpr:
    :param fees:
    :return: yield as percent => 0.7 = 70%
    """
    _yield = calculate_yield(profit-fees, bpr)
    annualized_yield = _yield * (365 / days_trade_open)
    annualized_yield = round(annualized_yield, 2)
    return annualized_yield

def calculate_annualized_yield_at_expiration(dte: int,bpr: float, init_option_premium: float, max_loss_via_stoploss: float, delta_short_option: float, fees: float) -> float:
    expected_loss = delta_short_option * max_loss_via_stoploss
    expected_gain = init_option_premium - expected_loss
    expected_gain -= fees
    annualized_yield = calculate_annualized_yield(expected_gain, dte, bpr, fees)
    annualized_yield = round(annualized_yield, 2)
    return annualized_yield

if __name__ == '__main__':
    init_option_premium = 80
    max_loss_via_stoploss = 2 * init_option_premium
    delta_short_option = 0.2
    bpr = 420
    profit = 35
    days_trade_open = 15
    dte = 40
    fees = 7 # opening fees 2 order long and short option for a spread


    annualized_yield_now = calculate_annualized_yield(profit, days_trade_open, bpr, fees*2) # *2 cause we need closing orders
    print(f"{annualized_yield_now} close now")

    annualized_yield_expiration = calculate_annualized_yield_at_expiration(dte, bpr, init_option_premium, max_loss_via_stoploss, delta_short_option, fees)
    print(f"{annualized_yield_expiration} close at expiration")
