from werkzeug.exceptions import MethodNotAllowed

from server_app import app
from flask import render_template, Blueprint, request, render_template_string, jsonify

from .option_calculation import get_iron_condor_data
from .querys import *
from .util import get_date_third_friday_of_next_month
from .yield_calculation import *


@app.route('/')
@app.route('/index')
def index():
    default_date = get_date_third_friday_of_next_month()
    return render_template('index.html', default_date=default_date)

@app.route('/results', methods=['POST'])
def results():
    if request.method != 'POST':
        raise MethodNotAllowed(f"Method is not allowed: {request.method}")

    # extract form data
    delta = float(request.form['delta']) / 100
    spread_type = request.form['spread_type']
    expiration_date = request.form['expiration_date'].replace('-', '')
    spread_width = float(request.form['spread_width'])

    if spread_type == "call_credit_spread":
        option_type = "C"
        h1_text = "Call Credit Spread Data"
        table_data = get_put_spread_options(delta, expiration_date, option_type, spread_width)

    elif spread_type == "put_credit_spread":
        option_type = "P"
        h1_text = "Put Credit Spread Data"
        table_data = get_put_spread_options(delta, expiration_date, option_type, spread_width)

    elif spread_type == "iron_condor":
        h1_text = "Iron Condor Data"

        # get call and put spreads
        call_data = get_put_spread_options(delta=delta, expiration_date=expiration_date, option_type="C", spread_width=spread_width)
        put_data = get_put_spread_options(delta=delta, expiration_date=expiration_date, option_type="P",  spread_width=spread_width)

        table_data = get_iron_condor_data(call_data, put_data)
        return render_template('iron_condor_results.html', table_data=table_data, h1_text=h1_text)

    else:
        raise ValueError(f"Unknown spread type: {spread_type}")
    return render_template('spread_results.html', table_data=table_data, h1_text=h1_text)

@app.route('/yield')
def _yield():
    return render_template('yield.html')

@app.route('/calc_yield', methods=['POST'])
def calculate():
    data = request.get_json()
    init_option_premium = float(data['init_option_premium'])
    max_loss_via_stoploss = float(data['max_loss_via_stoploss'])
    delta_short_option = float(data['delta_short_option'])
    bpr = float(data['bpr'])
    profit = float(data['profit'])
    days_trade_open = int(data['days_trade_open'])
    dte = int(data['dte'])
    fees = float(data['fees'])

    annualized_yield_now = calculate_annualized_yield(profit, days_trade_open, bpr, fees * 2) # *2 cause we need closing orders
    annualized_yield_expiration = calculate_annualized_yield_at_expiration(dte, bpr, init_option_premium, max_loss_via_stoploss, delta_short_option, fees)

    return jsonify({
        'annualized_yield_now': annualized_yield_now,
        'annualized_yield_expiration': annualized_yield_expiration
    })