import json
import time
import os
import requests
import urllib3

from config import *
from src.api_manager import APIManager

# always add verify=False for insecure requests and ignore the warning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class OptionScrapper:
    def __init__(self, api_manager: APIManager):
        self.api_manager = api_manager
        self.file_save_path = None
        self.symbol = None
        self.json = {}
        self.conid_symbol = None
        self.underlying_price = None
        self.strikes = None
        self.options_conids = None
        self.options_marketdata = None
        self.scrap_time_seconds = None

    def _save_json(self):
        print(f"Saving json to {self.file_save_path}")

        with open(self.file_save_path, "w") as json_file:
            json.dump(self.json, json_file, indent=4)
    def _clear(self):
        self.json = {}
        self.strikes = None
        self.conid_symbol = None
        self.underlying_price = None
        self.strikes = None
        self.options_conids = None
        self.options_marketdata = None
        self.file_save_path = None
        self.scrap_time_seconds = None

    def _validation_function_symbol_conid(self, response):
        try:
            conid = response.json()[0]["conid"]
        except Exception:
            return None
        return conid

    def _validation_function_underlying_price(self, response):
        try:
            # cast price and remove possible prefix for closing or halted prices
            price_str = response.json()[0]["31"]
            price_str = price_str.replace("C", "").replace("H", "")
            price = float(price_str)
        except Exception:
            return None

        # Debug print because I can not find the bug here
        if price is None:
            print(f"price is None Debug with the response!!! {response.json()}")
        return price

    def _validation_function_option_chain_strikes(self, response):
        option_strikes = response.json()

        # Calculate lower and upper price bounds
        lower_bound = self.underlying_price * (1 - PRICE_FILTER_PERCENT)
        upper_bound = self.underlying_price * (1 + PRICE_FILTER_PERCENT)

        try:
            # Filter strikes too far away from the underlying price use the put strikes.
            # Should be the same as the call strikes
            strikes = [strike for strike in option_strikes["put"] if lower_bound <= strike <= upper_bound]
        except Exception:
            return None

        # Sort strikes based on their distance from the underlying price
        strikes.sort(key=lambda x: abs(x - self.underlying_price))

        # Trim the lists to contain at most MAX_AMOUNT_STRIKES
        strikes = strikes[:MAX_AMOUNT_STRIKES]
        strikes.sort()

        return strikes

    def _validation_function_options_marketdata(self, response):
        #todo rewrite. store all entries which are ok. request the ones again which failed until all are ok
        data = response.json()
        market_data_fields = list(map(str, MARKET_DATA_FIELDS))

        for contract in data:
            are_all_keys_in_contract = all(field in contract for field in market_data_fields)
            if not are_all_keys_in_contract:
                return None

        return data

    def _request_handler(self,request_url, response_validation_function, error_counter=0, **kwargs):
        """
        Makes the request to the API and checks if the response is valid. When not valid it retries until the maximum
        number of retries is reached. Then an Exception is raised.
        :param request_url:
        :param response_validation_function:
        :param error_counter:
        :param **response_validation_function_params: keyword arguments to pass to response_validation_function
        :return: Validated data without errors for further processes
        """
        if IS_DEBUG_MODE:
            print(f"request_url: {request_url} @ error_counter: {error_counter}")

        if error_counter == MAX_REQUEST_ERROR_RETRIES - 1:
            raise Exception(
                f"request limit exceeded for validation function: {response_validation_function.__name__} and request {request_url}")

        # maybe gets better results by waiting after a bad request
        if error_counter > 0:
            print(f"Wait 10 seconds after unsuccessful request...")
            time.sleep(10)

        self.api_manager.wait_for_api_limit_restriction(self.symbol)
        response = requests.get(request_url, verify=False)

        if not response.ok:
            print(f"Failed to fetch data. Status code: {response.status_code}. retry request...")
            return self._request_handler(request_url, response_validation_function, error_counter + 1, **kwargs)

        validated_data = response_validation_function(response, **kwargs)

        if validated_data is not None:
            return validated_data
        else:
            print(f"WARNING: Response not parseable. Response: {response.json()}")
            return self._request_handler(request_url, response_validation_function, error_counter + 1, **kwargs)
    def _request_symbol_conid(self):
        print(f"Request symbol conid: {self.symbol}")
        request_url = f"{BASE_URL}/iserver/secdef/search?symbol={self.symbol}"
        conid = self._request_handler(request_url, self._validation_function_symbol_conid)
        print(f"-> conid: {conid}")
        self.conid_symbol = conid

    def _request_underlying_price(self):
        print(f"Request underlying price for symbol: {self.symbol} with conid: {self.conid_symbol}")
        request_url = f"{BASE_URL}/iserver/marketdata/snapshot?conids={self.conid_symbol}&fields=31"
        price = self._request_handler(request_url, self._validation_function_underlying_price)
        print("-> underlying price: ", price)
        self.underlying_price = price

    def _request_option_chain_strikes(self):
        print(f"Request option chain strikes for symbol: {self.symbol} with conid: {self.conid_symbol}. Only Strikes around {PRICE_FILTER_PERCENT*100}% of the current underlying price. And Max {MAX_AMOUNT_STRIKES} Strikes nearest to current price.")
        request_url = f"{BASE_URL}/iserver/secdef/strikes?conid={self.conid_symbol}&sectype=OPT&month={EXPIRATION_MONTH}"
        strikes = self._request_handler(request_url, self._validation_function_option_chain_strikes)
        self.strikes = strikes

    def _request_option_strike_contract(self, strike, right):
        print(f"Request option contract for symbol: {self.symbol} strike {strike} and right {right} @expiration {EXPIRATION_MONTH}")

        request_url = f"{BASE_URL}/iserver/secdef/info?conid={self.conid_symbol}&secType=OPT&month={EXPIRATION_MONTH}&strike={strike}&right={right}&exchange=SMART"

        self.api_manager.wait_for_api_limit_restriction(self.symbol)
        response = requests.get(url=request_url, verify=False)

        if not response.ok:
            raise Exception(f"Failed to fetch data. Status code: {response.status_code}")

        for result in response.json():
            if result["maturityDate"] == EXPIRATION_DATE:
                option_conid = result["conid"]
                print(f"-> optionconid: {option_conid}")
                return option_conid

        # expiration date not found return None
        print(f"-> {right} Strike {strike} not found for expiration {EXPIRATION_DATE}. optionconid: {None}")
        return None

    def _request_all_option_strikes_contracts_conids(self):
        print(f"Request option contract validations for symbol: {self.symbol} conid: {self.conid_symbol} and month: {EXPIRATION_MONTH}")

        self._build_json_before_option_strikes_contracts()

        for strike in self.strikes:
            for right in ["C", "P"]:
                conid = self._request_option_strike_contract(strike, right)

                # add conid for option contract
                if conid is not None:
                    self.json["option_market_data"][right][strike] = {"conid": conid}

                # remove strike if we have no option contract
                else:
                    del self.json["option_market_data"][right][strike]

        self.options_conids = [subdict[strike]["conid"] for subdict in self.json["option_market_data"].values() for strike in subdict.keys()]
        print(f"-> Amount of option contracts gathered: {len(self.options_conids)}")
        assert len(self.options_conids) > 1, "No option contracts"

    def _request_options_marketdata(self):
        print(f"Request options marketdata for symbol: {self.symbol} @ expiration date: {EXPIRATION_DATE}")
        query_str_conids = ",".join(str(conid) for conid in self.options_conids)
        query_str_fields = ','.join(map(str, MARKET_DATA_FIELDS))
        request_url = f"{BASE_URL}/iserver/marketdata/snapshot?conids={query_str_conids}&fields={query_str_fields}"
        self.options_marketdata = self._request_handler(request_url, self._validation_function_options_marketdata)

    def _build_json_before_option_strikes_contracts(self):
        self.json = {
            "expiration_date": EXPIRATION_DATE,
            "expiration_month": EXPIRATION_MONTH,
            "symbol": self.symbol,
            "underlying_price": self.underlying_price,
            "option_market_data": {
                "C": {strike: None for strike in self.strikes},
                "P": {strike: None for strike in self.strikes}
            }
        }

    def _remove_strikes_without_options_on_expiration_date(self):
        # assumed Call and Put Strikes are the same!
        self.strikes = self.json["option_market_data"]["C"].keys()

    def _clear_options_marketdata(self):
        print("Clearing options marketdata")

        keys_to_keep = ["conid", "last_price", "delta", "theta", "vega", "implied_vol_percent", "_updated"]

        for contract in self.options_marketdata:
            contract["last_price"] = float(contract["31"].replace("C", "").replace("H", ""))
            contract["delta"] = float(contract["7308"])
            contract["theta"] = float(contract["7310"])
            contract["vega"] = float(contract["7311"])
            contract["implied_vol_percent"] = float(contract["7633"].replace("%", ""))

            # Remove keys not in keys_to_keep
            keys_to_remove = [key for key in contract if key not in keys_to_keep]
            for key in keys_to_remove:
                del contract[key]

    def _merge_options_marketdata_with_strikes(self):
        print("Merging options marketdata with strikes")

        for contract in self.options_marketdata:
            conid = contract['conid']

            for option_type in ['C', 'P']:
                for strike_price, option_data in self.json['option_market_data'][option_type].items():
                    if option_data['conid'] == conid:
                        option_data.update(contract)
                        break

    def scrape_options(self, symbol: str):
        print("#"*80)
        print(f"Start scraping symbol {symbol} with expiration date {EXPIRATION_DATE}")
        print("#"*80)

        # First clear then set symbol!
        self._clear()
        self.symbol = symbol
        self.file_save_path = f"{PATH_JSON_FOLDER}/{self.symbol}.json"

        # Check for file -> skip the scrap
        if os.path.exists(self.file_save_path):
            print(f"File exists already at path: {self.file_save_path}. Skip symbol: {self.symbol}...")
            return

        start_time = time.time()
        self._request_symbol_conid()
        self._request_underlying_price()
        self._request_option_chain_strikes()
        self._request_all_option_strikes_contracts_conids()
        self._remove_strikes_without_options_on_expiration_date()
        self._request_options_marketdata()
        self._clear_options_marketdata()
        self._merge_options_marketdata_with_strikes()
        end_time = time.time()

        self.scrap_time_seconds = end_time - start_time
        self.json["scrap_time_seconds"] = self.scrap_time_seconds
        self._save_json()

        print("#"*80)
        print(f"Finished scraping symbol {symbol} with expiration date {EXPIRATION_DATE} in {self.scrap_time_seconds} seconds.")
        print("#"*80)

    def __str__(self):
        return json.dumps(self.json, indent=4)


