import numpy as np
import scipy.stats as si
#import matplotlib.pyplot as plt

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    return call_price, d1, d2

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
    return put_price, d1, d2

def calculate_greeks(S, K, T, r, sigma, d1, d2, option_type='call'):
    if option_type == 'call':
        delta = si.norm.cdf(d1, 0.0, 1.0)
    elif option_type == 'put':
        delta = si.norm.cdf(d1, 0.0, 1.0) - 1

    theta = - (S * si.norm.pdf(d1, 0.0, 1.0) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2 if option_type == 'call' else -d2, 0.0, 1.0)
    vega = S * np.sqrt(T) * si.norm.pdf(d1, 0.0, 1.0)
    return delta, theta, vega


def simulate_stock_prices(current_price, volatility, days, num_simulations=1000):
    dt = 1 / 252  # Zeitintervall in Jahren (1 Handelstag)
    price_paths = np.zeros((num_simulations, days))
    price_paths[:, 0] = current_price

    for t in range(1, days):
        rand = np.random.standard_normal(num_simulations)
        price_paths[:, t] = price_paths[:, t-1] * np.exp((0 - (volatility**2) / 2) * dt + volatility * np.sqrt(dt) * rand)

    return price_paths

def simulate_option_prices(stock_paths, K, T, r, volatility, num_simulations, days, option_type='call'):
    dt = 1 / 252  # Zeitintervall in Jahren (1 Handelstag)
    option_prices = np.zeros(stock_paths.shape)

    if option_type == 'call':
        option_prices[:, 0], d1, d2 = black_scholes_call(stock_paths[:, 0], K, T, r, volatility)
    elif option_type == 'put':
        option_prices[:, 0], d1, d2 = black_scholes_put(stock_paths[:, 0], K, T, r, volatility)

    for t in range(1, days):
        T_t = T - t * dt
        for i in range(num_simulations):
            if option_type == 'call':
                option_prices[i, t], d1, d2 = black_scholes_call(stock_paths[i, t], K, T_t, r, volatility)
            elif option_type == 'put':
                option_prices[i, t], d1, d2 = black_scholes_put(stock_paths[i, t], K, T_t, r, volatility)

    return option_prices

def evaluate_strategy(current_price, volatility, K, T, r, option_price, days, takeprofit, stoploss, early_close, num_simulations=1000, option_type='call'):
    stock_paths = simulate_stock_prices(current_price, volatility, days, num_simulations)
    option_paths = simulate_option_prices(stock_paths, K, T, r, volatility, num_simulations, days, option_type)

    initial_premium = option_price
    tp_dollar = initial_premium * takeprofit
    sl_dollar =tp_dollar * stoploss

    results = {
        "takeprofit_hits": 0,
        "stoploss_hits": 0,
        "early_closes": 0,
        "stock_paths": stock_paths,
        "option_paths": option_paths,
    }

    for i in range(num_simulations):
        for t in range(days):
            current_option_price = option_paths[i, t]
            if current_option_price <= initial_premium - tp_dollar:
                results["takeprofit_hits"] += 1
                break
            elif current_option_price >= initial_premium + sl_dollar:
                results["stoploss_hits"] += 1
                break
            elif t == early_close:
                results["early_closes"] += 1

    return results

if __name__ == '__main__':
    current_price = 100
    volatility = 0.2
    K = 100  # Strike-Preis
    T = 42 / 252  # Restlaufzeit in Jahren
    r = 0.01  # Risikofreier Zinssatz
    option_type = 'call'  # 'put' für Put-Optionen
    option_price = black_scholes_call(current_price, K, T, r, volatility)[0] if option_type == 'call' else black_scholes_put(current_price, K, T, r, volatility)[0]
    days = 42
    takeprofit = 0.6  # 60% Gewinnmitnahme
    stoploss_in_takeprofit = 2  # 2 times take profit value
    early_close = 35  # Vorzeitiges Schließen nach 35 Tagen

    results = evaluate_strategy(current_price, volatility, K, T, r, option_price, days, takeprofit, stoploss_in_takeprofit, early_close, num_simulations=1000, option_type=option_type)
    print(results)
