import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from experiments.run_engine_sweep import _planned_runs, _prepare_output_dir


class RunEngineSweepTests(unittest.TestCase):
    def test_planned_runs_skip_redundant_all_cloud_fractions(self):
        args = Namespace(
            policies=["nimbus", "cachedisp_oracle", "all_cloud"],
            fractions=[0.25, 0.5],
            nimbus_weights=["v2"],
        )

        self.assertEqual(
            _planned_runs(args),
            [
                ("nimbus", 0.25, "v2"),
                ("cachedisp_oracle", 0.25, "v2"),
                ("cachedisp_oracle", 0.5, "v2"),
                ("all_cloud", 1.0, "v2"),
            ],
        )

    def test_prepare_output_dir_removes_stale_outputs_unless_append(self):
        runs = [
            ("nimbus", 0.25, "v2"),
            ("cachedisp_oracle", 0.25, "v2"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            stale = [
                out_dir / "engine_summary.csv",
                out_dir / "engine_sweep_config.json",
                out_dir / "requests_nimbus_v2.csv",
                out_dir / "requests_cachedisp_oracle_f025.csv",
            ]
            for path in stale:
                path.write_text("stale")

            _prepare_output_dir(out_dir, runs, append=False)

            self.assertFalse(any(path.exists() for path in stale))

    def test_prepare_output_dir_preserves_stale_outputs_when_appending(self):
        runs = [("nimbus", 0.25, "v2")]
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            summary = out_dir / "engine_summary.csv"
            summary.write_text("stale")

            _prepare_output_dir(out_dir, runs, append=True)

            self.assertEqual(summary.read_text(), "stale")


if __name__ == "__main__":
    unittest.main()
