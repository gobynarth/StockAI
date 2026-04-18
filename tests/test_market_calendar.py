import datetime as dt
import unittest

from market_calendar import is_us_stock_market_day


class MarketCalendarTests(unittest.TestCase):
    def test_returns_false_on_saturday(self):
        self.assertFalse(is_us_stock_market_day(dt.date(2026, 4, 18)))

    def test_returns_true_on_regular_friday(self):
        self.assertTrue(is_us_stock_market_day(dt.date(2026, 4, 17)))

    def test_returns_false_on_good_friday(self):
        self.assertFalse(is_us_stock_market_day(dt.date(2026, 4, 3)))

    def test_returns_false_on_christmas(self):
        self.assertFalse(is_us_stock_market_day(dt.date(2026, 12, 25)))


if __name__ == "__main__":
    unittest.main()
