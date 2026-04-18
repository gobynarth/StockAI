import unittest

from email_formatting import format_entry_with_date


class EmailFormattingTests(unittest.TestCase):
    def test_format_entry_with_date_includes_entry_date_below_price(self):
        html = format_entry_with_date(12.78, "2026-04-17")

        self.assertIn("$12.78", html)
        self.assertIn("Entered 2026-04-17", html)


if __name__ == "__main__":
    unittest.main()
