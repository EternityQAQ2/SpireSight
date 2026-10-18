"""Loading-boundary regressions that run without a CUDA device or model weights."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sts2_advisor.model import TransformersAdvisor


class ModelLoadingTests(unittest.TestCase):
    def load(self, *, cuda=False, device="auto", cache=None):
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: cuda),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
            float16="float16", float32="float32",
        )
        tokenizer_loader = Mock(return_value=object())
        loaded_model = Mock()
        model_loader = Mock(return_value=loaded_model)
        transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_loader),
        )
        with patch.dict("sys.modules", {"torch": torch, "transformers": transformers}):
            backend = TransformersAdvisor("local/model", cache, device=device)
        return backend, tokenizer_loader, model_loader, loaded_model

    def test_cuda_keeps_auto_mapping_half_precision_and_offline_loading(self):
        backend, tokenizer, loader, loaded = self.load(cuda=True)
        self.assertEqual(backend.device, "cuda")
        self.assertEqual(loader.call_args.kwargs["device_map"], "auto")
        self.assertEqual(loader.call_args.kwargs["dtype"], "float16")
        self.assertTrue(loader.call_args.kwargs["local_files_only"])
        self.assertTrue(tokenizer.call_args.kwargs["local_files_only"])
        loaded.to.assert_not_called()
        loaded.eval.assert_called_once()

    def test_cpu_override_disables_cuda_mapping(self):
        backend, _, loader, loaded = self.load(cuda=True, device="cpu")
        self.assertEqual(backend.device, "cpu")
        self.assertIsNone(loader.call_args.kwargs["device_map"])
        self.assertEqual(loader.call_args.kwargs["dtype"], "float32")
        loaded.to.assert_called_once_with("cpu")

    def test_original_cache_environment_defaults_remain_available(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            cache = Path(directory)
            _, tokenizer, loader, _ = self.load(cache=cache)
            self.assertEqual(os.environ.get("HF_HOME"), str(cache))
            self.assertEqual(os.environ.get("HF_HUB_CACHE"), str(cache / "hub"))
            self.assertEqual(os.environ.get("HF_XET_CACHE"), str(cache / "xet"))
            self.assertEqual(loader.call_args.kwargs["cache_dir"], str(cache / "hub"))
            self.assertEqual(tokenizer.call_args.kwargs["cache_dir"], str(cache / "hub"))

    def test_explicit_cache_environment_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            existing = {"HF_HOME": str(cache / "home"), "HF_HUB_CACHE": str(cache / "custom-hub"),
                        "HF_XET_CACHE": str(cache / "custom-xet")}
            with patch.dict(os.environ, existing):
                _, tokenizer, loader, _ = self.load(cache=cache / "default")
                self.assertEqual({name: os.environ[name] for name in existing}, existing)
                self.assertEqual(loader.call_args.kwargs["cache_dir"], existing["HF_HUB_CACHE"])
                self.assertEqual(tokenizer.call_args.kwargs["cache_dir"], existing["HF_HUB_CACHE"])


if __name__ == "__main__":
    unittest.main()
