import unittest

from nimbus.cost_calculator import APICostCalculator


class APICostCalculatorTests(unittest.TestCase):
    def test_default_cost_is_all_uncached_input_plus_output(self):
        calc = APICostCalculator(
            input_price_per_million=1.0,
            output_price_per_million=10.0,
        )

        self.assertEqual(calc.calculate_cost(1_000_000, 100_000), 2.0)

    def test_cached_input_price_breakdown(self):
        calc = APICostCalculator(
            input_price_per_million=1.0,
            cached_input_price_per_million=0.25,
            output_price_per_million=10.0,
        )

        breakdown = calc.calculate_cost_breakdown(
            input_tokens=1_000_000,
            cached_input_tokens=400_000,
            output_tokens=100_000,
        )

        self.assertEqual(breakdown["uncached_input_tokens"], 600_000)
        self.assertEqual(breakdown["cached_input_tokens"], 400_000)
        self.assertEqual(breakdown["output_tokens"], 100_000)
        self.assertEqual(breakdown["total_cost_usd"], 1.7)

    def test_cached_input_tokens_are_clamped_to_input_tokens(self):
        calc = APICostCalculator(
            input_price_per_million=1.0,
            cached_input_price_per_million=0.25,
            output_price_per_million=10.0,
        )

        breakdown = calc.calculate_cost_breakdown(
            input_tokens=100,
            cached_input_tokens=1_000,
            output_tokens=0,
        )

        self.assertEqual(breakdown["uncached_input_tokens"], 0)
        self.assertEqual(breakdown["cached_input_tokens"], 100)


if __name__ == "__main__":
    unittest.main()
