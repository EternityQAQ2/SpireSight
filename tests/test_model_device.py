from __future__ import annotations

import unittest
from types import SimpleNamespace

from sts2_advisor import model
from sts2_advisor.cli import build_parser


def hardware(*, cuda=False, mps=False, has_mps=True):
    backends = SimpleNamespace()
    if has_mps:
        backends.mps = SimpleNamespace(is_available=lambda: mps)
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda), backends=backends
    )


class DeviceTests(unittest.TestCase):
    def test_auto_uses_apple_gpu_when_cuda_is_absent(self):
        self.assertEqual(model.select_device(hardware(mps=True)), "mps")

    def test_auto_preserves_cuda_priority(self):
        self.assertEqual(model.select_device(hardware(cuda=True, mps=True)), "cuda")

    def test_auto_falls_back_to_cpu_without_accelerators(self):
        self.assertEqual(model.select_device(hardware()), "cpu")
        self.assertEqual(model.select_device(hardware(has_mps=False)), "cpu")

    def test_cpu_can_be_forced_on_apple_silicon(self):
        self.assertEqual(model.select_device(hardware(mps=True), "cpu"), "cpu")

    def test_unavailable_device_has_actionable_error(self):
        for device in ("mps", "cuda"):
            with self.subTest(device=device), self.assertRaisesRegex(RuntimeError, "--device cpu"):
                model.select_device(hardware(), device)

    def test_invalid_device_is_rejected(self):
        with self.assertRaises(ValueError):
            model.select_device(hardware(), "metal")

    def test_cli_accepts_device_override(self):
        self.assertEqual(build_parser().parse_args(["--device", "cpu"]).device, "cpu")


if __name__ == "__main__":
    unittest.main()
