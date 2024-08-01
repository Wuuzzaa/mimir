import os

from src.api_manager import APIManager
from src.jsons_to_sql_db import create_database, process_json_files
from src.option_scrapper import OptionScrapper
from config import *
import threading
import time


def get_not_scraped_symbols():
    symbols = []
    for symbol in SYMBOLS:
        file_save_path = f"{PATH_JSON_FOLDER}/{symbol}.json"
        if not os.path.exists(file_save_path):
            print(f"{symbol} is not scraped -> Scrape it")
            symbols.append(symbol)
        else:
            print(f"{symbol} already scraped.")

    print(f"{len(symbols)} symbols need to be scraped!")
    return symbols



def json_to_sqlite(json_directory, db_name):
    create_database(db_name)
    process_json_files(json_directory, db_name)


def worker(symbol, api_manager):
    option_scrapper = OptionScrapper(api_manager)

    try:
        option_scrapper.scrape_options(symbol)
    except Exception as e:
        print(f"Error occurred for symbol {symbol}: Exception Type: {type(e).__name__}: {e}")



if __name__ == '__main__':
    symbols = get_not_scraped_symbols()
    num_workers = len(symbols)
    api_manager = APIManager(IB_API_MAX_REQUESTS_PER_SECOND)

    threads = []
    start_time = time.time()

    for symbol in symbols:
        t = threading.Thread(target=worker, args=(symbol, api_manager))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time for {num_workers} worker threads: {execution_time} seconds")

    json_to_sqlite(json_directory=PATH_JSON_FOLDER, db_name=f"{DATABASE_FOLDER}/{DATABASE_FILE_NAME}")
