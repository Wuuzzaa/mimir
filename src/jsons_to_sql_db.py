import sqlite3
import json
import os


# json_directory = '../data'
# db_name = 'options_data.db'


def create_database(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Symbol table
    cursor.execute('''CREATE TABLE IF NOT EXISTS symbols (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT,
                    expiration_date TEXT,
                    expiration_month TEXT,
                    underlying_price REAL,
                    scrap_time_seconds REAL
                    )''')

    # Option table
    cursor.execute('''CREATE TABLE IF NOT EXISTS options (
                    id INTEGER PRIMARY KEY,
                    symbol_id INTEGER,
                    option_type TEXT,
                    strike REAL,
                    conid INTEGER,
                    last_price REAL,
                    delta REAL,
                    theta REAL,
                    vega REAL,
                    implied_vol_percent REAL,
                    FOREIGN KEY (symbol_id) REFERENCES symbols(id)
                    )''')
    conn.commit()
    conn.close()


def insert_data(file_path, db_name):
    with open(file_path) as f:
        data = json.load(f)
        symbol = data.get('symbol')
        expiration_date = data.get('expiration_date')
        expiration_month = data.get('expiration_month')
        underlying_price = data.get('underlying_price')
        scrap_time_seconds = data.get('scrap_time_seconds')

        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        cursor.execute('SELECT id FROM symbols WHERE symbol=?', (symbol,))
        symbol_id = cursor.fetchone()
        if symbol_id is None:
            cursor.execute('''INSERT INTO symbols (symbol, expiration_date, expiration_month, 
                            underlying_price, scrap_time_seconds) 
                            VALUES (?, ?, ?, ?, ?)''',
                           (symbol, expiration_date, expiration_month, underlying_price, scrap_time_seconds))
            symbol_id = cursor.lastrowid
        else:
            symbol_id = symbol_id[0]

        option_market_data = data.get('option_market_data')

        for option_type, strikes in option_market_data.items():
            for strike, option_data in strikes.items():
                conid = option_data.get('conid')
                last_price = option_data.get('last_price')
                delta = option_data.get('delta')
                theta = option_data.get('theta')
                vega = option_data.get('vega')
                implied_vol_percent = option_data.get('implied_vol_percent')

                cursor.execute('''INSERT INTO options (symbol_id, option_type, strike, 
                                conid, last_price, delta, theta, vega, implied_vol_percent) 
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                               (symbol_id, option_type, strike, conid, last_price, delta, theta, vega,
                                implied_vol_percent))

        conn.commit()
        conn.close()


def process_json_files(json_directory, db_name):
    for filename in os.listdir(json_directory):
        if filename.endswith('.json'):
            file_path = os.path.join(json_directory, filename)
            insert_data(file_path, db_name)


# if __name__ == "__main__":
#     create_database()
#     process_json_files()
