import sqlite3
import logging
from data.config import *
from .option_calculation import *

DATABASE = f"{DATABASE_FOLDER}/{DATABASE_FILE_NAME}"

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def execute_query(query, params=None):
    logging.debug("\nExecuting query: %s with params: %s", query, params)

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    if params:
        cursor.execute(query, params)
    else:
        cursor.execute(query)
    results = cursor.fetchall()
    conn.close()

    logging.debug("Query results: %s", results)

    return results


def _get_put_spread_short_options(delta, expiration_date, option_type):
    query = """
        SELECT 
            symbols.symbol, 
            options.strike, 
            options.last_price, 
            options.delta, 
            options.vega, 
            options.theta, 
            options.implied_vol_percent,
            options.option_type, 
            symbols.expiration_date
        FROM 
            symbols
        JOIN 
            options ON symbols.id = options.symbol_id
        JOIN 
            (
                SELECT 
                    symbol_id,
                    MIN(ABS(ABS(delta) - :delta_param)) AS min_delta_diff
                FROM 
                    options
                WHERE 
                    option_type = :option_type_param 
                GROUP BY 
                    symbol_id
            ) AS min_delta_table ON options.symbol_id = min_delta_table.symbol_id
                                        AND ABS(ABS(options.delta) - :delta_param) = min_delta_table.min_delta_diff

        WHERE  
        option_type = :option_type_param AND
        expiration_date = :expiration_date_param
                """
    params = {
        'delta_param': delta,
        'expiration_date_param': expiration_date,
        "option_type_param": option_type
    }

    result = execute_query(query, params)
    return result


def _get_put_spread_long_option(symbol, expiration_date, option_type, spread_width, short_strike):
    # Construct the appropriate SQL condition based on option_type
    if option_type == "P":
        sql_where_strike_condition = "Strike <= :short_strike_param - :spread_width_param"
        sql_order_by = "DESC"
    else:  # option_type == "C"
        sql_where_strike_condition = "Strike >= :short_strike_param + :spread_width_param"
        sql_order_by = "ASC"


    query = f"""
    SELECT 
        symbols.symbol, 
        options.strike, 
        options.last_price, 
        options.delta, 
        options.vega, 
        options.theta, 
        options.implied_vol_percent,
        options.option_type, 
        symbols.expiration_date
    FROM 
        symbols
    JOIN 
        options ON symbols.id = options.symbol_id
    WHERE
        symbol = :symbol_param AND
        option_type = :option_type_param AND
        expiration_date = :expiration_date_param AND
        {sql_where_strike_condition}
    ORDER by strike {sql_order_by}
    Limit 1
    """

    params = {
        'symbol_param': symbol,
        'expiration_date_param': expiration_date,
        "option_type_param": option_type,
        "spread_width_param": spread_width,
        "short_strike_param": short_strike
    }

    result = execute_query(query, params)
    return result

def get_put_spread_options(delta, expiration_date, option_type, spread_width):
    short_options = _get_put_spread_short_options(delta, expiration_date, option_type)

    table_data = []

    for row in short_options:
        # add short option data
        spread = {
            'Symbol': row[0],
            'Short Strike': row[1],
            'Short Last Price': row[2],
            'Short Delta': row[3],
            'Short Vega': row[4],
            'Short Theta': row[5],
            'Short Implied Volatility': row[6],
            'Short Option Type': row[7],
            'Short Expiration Date': row[8]

        }

        # query the long option
        long_option = _get_put_spread_long_option(
            symbol=spread["Symbol"],
            expiration_date=spread["Short Expiration Date"],
            option_type=spread["Short Option Type"],
            spread_width=spread_width,
            short_strike=spread["Short Strike"]
        )

        # no long option found make an empty entry
        if long_option == []:
            spread["Long Symbol"] = None
            spread["Long Strike"] = None
            spread["Long Last Price"] = None
            spread["Long Delta"] = None
            spread["Long Vega"] = None
            spread["Long Theta"] = None
            spread["Long Implied Volatility"] = None
            spread["Long Option Type"] = None
            spread["Long Expiration Date"] = None
            spread["Spread Width"] = None

        # add long option data
        else:
            spread["Long Symbol"] = long_option[0][0]
            spread["Long Strike"] = long_option[0][1]
            spread["Long Last Price"] = long_option[0][2]
            spread["Long Delta"] = long_option[0][3]
            spread["Long Vega"] = long_option[0][4]
            spread["Long Theta"] = long_option[0][5]
            spread["Long Implied Volatility"] = long_option[0][6]
            spread["Long Option Type"] = long_option[0][7]
            spread["Long Expiration Date"] = long_option[0][8]
            spread["Spread Width"] = float(abs(spread["Long Strike"] - spread['Short Strike']))

        table_data.append(spread)

    table_data = put_spread_calculations(table_data)
    return table_data