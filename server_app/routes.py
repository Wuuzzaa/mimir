from server_app import app
from flask import render_template, Blueprint, request, render_template_string
from .querys import *

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    if request.method == 'POST':
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