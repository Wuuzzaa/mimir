BASE_URL = "https://localhost:5000/v1/api"
PRICE_FILTER_PERCENT = 0.5
MAX_AMOUNT_STRIKES = 100
MAX_REQUEST_ERROR_RETRIES = 10
IS_DEBUG_MODE = True
EXPIRATION_MONTH = "SEP24"
EXPIRATION_DATE = "20240920" #yyyymmdd
PATH_JSON_FOLDER = "data/json"
DATABASE_FILE_NAME = "data/options_data.db"
DATABASE_FOLDER = "data"
IB_API_MAX_REQUESTS_PER_SECOND = 10

# https://ibkrcampus.com/ibkr-api-page/webapi-doc/#market-data-fields
# 31    Last Price
# 7089  Opt. Volume
# 7308  Delta
# 7310  Theta
# 7311  Vega
# 7633  Implied Vol. %
MARKET_DATA_FIELDS = [
        31,
        # 7089, # seems buggy
        7308,
        7310,
        7311,
        7633
]

# out comment all with low price -> low premium
SYMBOLS = [
        # "AA",
        # "AAL",
        "AAPL",
        "AES",
        "AMD",
        "AMZN",
        "AVGO",
        "BA",
        "BABA",
        "BIDU",
        #"BYND",
        "C",
        "CAT",
        "COF",
        "COST",
        "CRM",
        #"CRON",
        "CRWD",
        "CSCO",
        "DIA",
        "DIS",
        "EEM",
        # "EWZ",
        # "FL",
        # "FXI",
        # "GDX",
        # "GDXJ",
        "GE",
        "GILD",
        "GLD",
        "GM",
        "GOOG",
        "GS",
        # "HAL",
        "HD",
        #"HPE",
        "IBM",
        "IWM",
        "JPM",
        "LOW",
        #"M",
        "MCD",
        "META",
        "MMM",
        "MRVL",
        "MSFT",
        "MU",
        "NFLX",
        #"NIO",
        "NVDA",
        "ORCL",
        "PG",
        "QCOM",
        "QQQ",
        "ROKU",
        "SBUX",
        "SHOP",
        #"SLV",
        "SMH",
        #"SNAP",
        "SPY",
        "SQ",
        "TGT",
        "TLT",
        "TSLA",
        "UBER",
        "USO",
        "V",
        #"WBA",
        "WFC",
        "WMT",
        "X",
        "XBI",
        "XLU",
        "XOM",
        #"JD",
        "XOP",
    ]

