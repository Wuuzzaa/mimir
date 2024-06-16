from werkzeug.exceptions import MethodNotAllowed

from server_app import app
from flask import render_template, Blueprint, request, render_template_string
from .querys import *
from .util import get_date_third_friday_of_next_month


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

    if spread_type == "Call Credit Spread":
        raise Exception("Not implemented yet")

    option_type = "P"
    table_data = get_put_spread_options(delta, expiration_date, option_type, spread_width)
    return render_template('results.html', table_data=table_data)