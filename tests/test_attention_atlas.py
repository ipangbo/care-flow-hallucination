"""CPU-only tests of the standalone figure code; fixtures are not model data."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "figures" / "plot_attention_atlas.py"
spec = importlib.util.spec_from_file_location("core_attention_atlas", SCRIPT)
atlas = importlib.util.module_from_spec(spec)
spec.loader.exec_module(atlas)


class AttentionAtlasTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.linspace(0, 1, 1024).reshape(32, 32)

    def test_npy_and_npz_load_without_changing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.save(path / "test.npy", self.matrix)
            np.savez(path / "test.npz", hca=self.matrix, model_id=atlas.MODEL_ID)
            for name in ("test.npy", "test.npz"):
                np.testing.assert_array_equal(atlas.load_hca(path / name), self.matrix)

    def test_rejects_wrong_shapes_nonfinite_and_out_of_range(self):
        for values in (np.zeros((40, 40)), np.full((32, 32), np.nan),
                       np.full((32, 32), np.inf), self.matrix - 0.01, self.matrix + 0.01):
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                atlas.validate_hca(values)

    def test_rejects_npz_without_hca(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.npz"
            np.savez(path, attention=self.matrix)
            with self.assertRaisesRegex(ValueError, "key 'hca'"):
                atlas.load_hca(path)

    def test_rejects_interleave_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.npz"
            np.savez(path, hca=self.matrix, model_id="llava-hf/llava-interleave-qwen-7b-hf")
            with self.assertRaisesRegex(ValueError, "Expected.*measurements"):
                atlas.load_hca(path)

    def test_rejects_old_metrics_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.npy"
            np.save(path, self.matrix)
            path.with_name("metrics.json").write_text(
                json.dumps({"model_id": "llava-hf/llava-interleave-qwen-7b-hf"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Expected.*measurements"):
                atlas.load_hca(path)

    def test_clustering_is_one_reversible_global_permutation(self):
        original = self.matrix.copy()
        order = atlas.clustered_head_order(self.matrix)
        np.testing.assert_array_equal(np.sort(order), np.arange(32))
        np.testing.assert_array_equal(self.matrix, original)
        np.testing.assert_array_equal(self.matrix[:, order][:, np.argsort(order)], original)
        np.testing.assert_array_equal(order, atlas.clustered_head_order(self.matrix))
        early_changed = self.matrix.copy()
        early_changed[:15] = 0
        np.testing.assert_array_equal(order, atlas.clustered_head_order(early_changed))

    def test_render_both_orders_and_export_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("original", "clustered-symmetric"):
                output = Path(directory) / mode
                order = atlas.plot_atlas(self.matrix, output, mode)
                for extension in ("png", "pdf", "svg"):
                    self.assertGreater(Path(f"{output}.{extension}").stat().st_size, 100)
                svg = Path(f"{output}.svg").read_text(encoding="utf-8")
                self.assertIn("Attention head", svg)
                self.assertNotIn("(reordered)", svg)
                mapping = np.loadtxt(f"{output}_head_order.csv", delimiter=",", skiprows=1, dtype=int)
                np.testing.assert_array_equal(mapping[:, 1], order + 1)
                if mode == "original":
                    np.testing.assert_array_equal(order, np.arange(32))

    def test_cli_does_not_need_core_or_parent_repository_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.save(path / "fixture.npy", self.matrix)
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(path / "fixture.npy"),
                 "--head-order", "original", "--output", str(path / "result")],
                cwd=path, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((path / "result.pdf").is_file())


if __name__ == "__main__":
    unittest.main()
