"""Optional real-model smoke tests; no downloaded weights or network needed."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sts2_advisor.model import RuleAdvisor, TransformersAdvisor
from sts2_advisor.recording import SessionRecorder
from sts2_advisor.retrieval import NullRetriever
from sts2_advisor.service import AdvisorService


HAS_MODEL = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers"))


@unittest.skipUnless(HAS_MODEL, "install the model extra to run real-model smoke tests")
class ModelRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

        cls.torch = torch
        cls.directory = tempfile.TemporaryDirectory(prefix="尖塔 local model ")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.path = Path(cls.directory.name)
        tokenizer = Tokenizer(WordLevel(
            {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "hello": 4, "test": 5},
            unk_token="[UNK]",
        ))
        tokenizer.pre_tokenizer = Whitespace()
        fast = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]",
            bos_token="[BOS]", eos_token="[EOS]",
        )
        fast.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
        fast.save_pretrained(cls.path)
        torch.manual_seed(7)
        GPT2LMHeadModel(GPT2Config(
            vocab_size=6, n_positions=2048, n_embd=16, n_layer=1, n_head=2,
            bos_token_id=2, eos_token_id=3, pad_token_id=0,
        )).save_pretrained(cls.path)

    def check_generation(self, requested, expected):
        backend = TransformersAdvisor(str(self.path), max_new_tokens=2, device=requested)
        parameter = next(backend.model.parameters())
        self.assertEqual(parameter.device.type, expected)
        self.assertEqual(parameter.dtype, self.torch.float32 if expected == "cpu" else self.torch.float16)
        inputs = {key: value.to(parameter.device)
                  for key, value in backend.tokenizer("hello test", return_tensors="pt").items()}
        with self.torch.inference_mode():
            logits = backend.model(**inputs).logits
            output = backend.model.generate(**inputs, max_new_tokens=2, do_sample=False)
        self.assertTrue(self.torch.isfinite(logits).all().item())
        self.assertGreater(output.shape[1], inputs["input_ids"].shape[1])
        self.assertEqual(output.device.type, expected)
        return backend

    def test_cpu_loads_local_model_and_generates(self):
        self.check_generation("cpu", "cpu")

    def test_explicit_hub_cache_works_after_transformers_is_imported(self):
        with tempfile.TemporaryDirectory(prefix="尖塔 hub cache ") as directory:
            cache = Path(directory)
            repository = cache / "hub/models--local--tiny-advisor"
            revision = "a" * 40
            shutil.copytree(self.path, repository / "snapshots" / revision)
            (repository / "refs").mkdir()
            (repository / "refs/main").write_text(revision, encoding="utf-8")
            with patch.dict(os.environ, {"HF_HOME": str(cache), "HF_HUB_CACHE": str(cache / "hub"),
                                         "HF_XET_CACHE": str(cache / "xet")}):
                backend = TransformersAdvisor("local/tiny-advisor", cache, max_new_tokens=2)
            self.assertEqual(backend.model.config.vocab_size, 6)

    def test_apple_gpu_loads_local_model_and_generates(self):
        if not self.torch.backends.mps.is_available():
            self.skipTest("MPS not available on this host")
        self.check_generation("auto", "mps")

    def test_real_invalid_model_output_falls_back_to_legal_rule_advice(self):
        raw = json.loads((Path(__file__).parent / "fixtures/combat_state.json").read_text(encoding="utf-8"))

        class Client:
            def get_singleplayer_state(self):
                return raw

        backend = TransformersAdvisor(str(self.path), max_new_tokens=2, device="cpu")
        output = []
        service = AdvisorService(
            Client(), NullRetriever(), backend, SessionRecorder(self.path, enabled=False),
            backend_name="transformers", fallback_backend=RuleAdvisor(), output=output.append,
        )
        self.assertTrue(service.run_once())
        self.assertIn("改用规则后端", output[0])
        self.assertIn("防御", output[1])
        self.assertFalse(service.run_once())


if __name__ == "__main__":
    unittest.main()
