import calendar
from datetime import date, timedelta


def _nth_weekday(year, month, weekday, n):
    month_days = calendar.monthcalendar(year, month)
    matches = [week[weekday] for week in month_days if week[weekday] != 0]
    return date(year, month, matches[n - 1])


def _last_weekday(year, month, weekday):
    month_days = calendar.monthcalendar(year, month)
    matches = [week[weekday] for week in month_days if week[weekday] != 0]
    return date(year, month, matches[-1])


def _observed_fixed_holiday(year, month, day):
    holiday = date(year, month, day)
    if holiday.weekday() == 5:
        return holiday - timedelta(days=1)
    if holiday.weekday() == 6:
        return holiday + timedelta(days=1)
    return holiday


def _easter_sunday(year):
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nyse_holidays(year):
    easter = _easter_sunday(year)
    return {
        _observed_fixed_holiday(year, 1, 1),
        _nth_weekday(year, 1, calendar.MONDAY, 3),
        _nth_weekday(year, 2, calendar.MONDAY, 3),
        easter - timedelta(days=2),
        _last_weekday(year, 5, calendar.MONDAY),
        _observed_fixed_holiday(year, 6, 19),
        _observed_fixed_holiday(year, 7, 4),
        _nth_weekday(year, 9, calendar.MONDAY, 1),
        _nth_weekday(year, 11, calendar.THURSDAY, 4),
        _observed_fixed_holiday(year, 12, 25),
    }


def is_us_stock_market_day(day):
    if day.weekday() >= 5:
        return False
    return day not in _nyse_holidays(day.year)
