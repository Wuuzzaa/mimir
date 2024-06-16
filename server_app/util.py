import datetime


def get_date_third_friday_of_next_month() -> datetime.date:
    today = datetime.date.today()
    year = today.year
    next_month = today.month + 1
    if next_month > 12:
        next_month = 1
        year += 1

    first_day_next_month = datetime.date(year, next_month, 1)
    first_friday = first_day_next_month + datetime.timedelta((4 - first_day_next_month.weekday() + 7) % 7)
    third_friday = first_friday + datetime.timedelta(days=14)

    return third_friday
