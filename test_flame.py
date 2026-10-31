"""CPU regression tests: formulas, gradients, pruning, data isolation and reload."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
import torch.nn.functional as F

from data_io import dump_jsonl, load_records, split_records, synthetic_records
from flame_core import (ByteTokenizer, Config, ImportanceTracker, QuantizedLoRALinear, TinyClassifier,
                        binary_metrics, decoder_layers, install_loqdc, layer_discrepancies,
                        merge_adapters, pack_signed, qa_loss, quantize_absmax, select_layers, unpack_signed)
from run import load_artifact, save_artifact


class FormulaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_quantization_equations(self):
        w = torch.tensor([[-1., -0.3, 0., 0.3, 1.]])
        q, scale = quantize_absmax(w, 4)
        self.assertEqual(scale.item(), 7.)
        torch.testing.assert_close(q, torch.tensor([[-7, -2, 0, 2, 7]], dtype=torch.int8))
        self.assertLessEqual((w - q / scale).abs().max().item(), 0.5 / scale + 1e-7)
        q0, s0 = quantize_absmax(torch.zeros(2, 3), 4)
        self.assertEqual(s0.item(), 1.)
        self.assertEqual(q0.count_nonzero().item(), 0)

    def test_packing_all_signed_ranges_and_padding(self):
        for bits in (2, 4, 8):
            q = torch.arange(-2 ** (bits - 1), 2 ** (bits - 1), dtype=torch.int8)
            torch.testing.assert_close(unpack_signed(pack_signed(q, bits), bits, q.shape), q)
            odd = q[:3]
            packed = pack_signed(odd, bits)
            torch.testing.assert_close(unpack_signed(packed, bits, odd.shape), odd)
            self.assertEqual(packed.numel(), (len(odd) + 8 // bits - 1) // (8 // bits))

    def test_literal_penalty_has_zero_adapter_gradient(self):
        cfg = Config(loss_mode="literal", dropout=0, qa_alpha=1)
        model = TinyClassifier(layers=2, max_length=16)
        install_loqdc(model, cfg)
        ids = torch.tensor([[2, 5, 6, 1]])
        ce = F.cross_entropy(model(ids, torch.ones_like(ids)), torch.tensor([1]))
        params = [p for p in model.parameters() if p.requires_grad]
        gradient_ce = torch.autograd.grad(ce, params, retain_graph=True)
        loss, penalty = qa_loss(ce, model, torch.ones(2), cfg)
        self.assertFalse(penalty.requires_grad)
        gradient_qa = torch.autograd.grad(loss, params)
        for left, right in zip(gradient_ce, gradient_qa):
            torch.testing.assert_close(left, right)

    def test_residual_is_explicit_differentiable_extension(self):
        cfg = Config(loss_mode="residual", dropout=0)
        model = TinyClassifier(layers=2)
        install_loqdc(model, cfg)
        residual = layer_discrepancies(model, "residual").sum()
        params = [m.B for m in model.modules() if isinstance(m, QuantizedLoRALinear)]
        grads = torch.autograd.grad(residual, params)
        self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)

    def test_default_residual_changes_gradients_and_adamw_update(self):
        import copy
        from run import parser
        cfg = Config(dropout=0)
        self.assertEqual(cfg.loss_mode, "residual")
        self.assertEqual(parser().parse_args(["demo"]).loss_mode, "residual")
        model = TinyClassifier(layers=2, max_length=16)
        install_loqdc(model, cfg)
        for module in model.modules():
            if isinstance(module, QuantizedLoRALinear):
                module.B.data.normal_(std=0.01)
        control = copy.deepcopy(model)
        ids = torch.tensor([[2, 5, 6, 1]])
        mask = torch.ones_like(ids)
        target = torch.tensor([1])
        ce = F.cross_entropy(model(ids, mask), target)
        loss, penalty = qa_loss(ce, model, torch.ones(2), cfg)
        self.assertTrue(penalty.requires_grad)
        adapters = [p for name, p in model.named_parameters() if name.endswith((".A", ".B"))]
        grads = torch.autograd.grad(penalty, adapters, retain_graph=True)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        control_optimizer = torch.optim.AdamW([p for p in control.parameters() if p.requires_grad], lr=1e-4)
        loss.backward()
        F.cross_entropy(control(ids, mask), target).backward()
        delta = sum((p.grad - q.grad).abs().sum().item()
                    for p, q in zip(model.parameters(), control.parameters()) if p.requires_grad)
        self.assertGreater(delta, 0)
        optimizer.step()
        control_optimizer.step()
        update_delta = sum((p - q).abs().sum().item()
                           for p, q in zip(model.parameters(), control.parameters()) if p.requires_grad)
        self.assertGreater(update_delta, 0)

    def test_lora_merge_forward(self):
        cfg = Config(dropout=0)
        module = QuantizedLoRALinear(nn.Linear(7, 5), cfg)
        with torch.no_grad():
            module.B.normal_()
        module.eval()
        x = torch.randn(3, 7)
        torch.testing.assert_close(module(x), module.merged_linear()(x), rtol=1e-5, atol=1e-5)

    def test_importance_formula_5_and_6(self):
        cfg = Config(beta1=0.5, beta2=0.5)
        model = TinyClassifier(layers=1)
        install_loqdc(model, cfg)
        tracker = ImportanceTracker(model, cfg)
        modules = [m for m in model.modules() if isinstance(m, QuantizedLoRALinear)]
        for m in modules:
            for p in (m.A, m.B):
                p.data.fill_(2)
                p.grad = torch.full_like(p, 3)
        tracker.update()
        self.assertAlmostEqual(tracker.scores().item(), 7 * 2 * 4.5)

    def test_freeze_and_projection_coverage(self):
        model = TinyClassifier(layers=3)
        install_loqdc(model, Config())
        self.assertEqual(sum(isinstance(m, QuantizedLoRALinear) for m in model.modules()), 21)
        for name, p in model.named_parameters():
            self.assertEqual(p.requires_grad, name.endswith((".A", ".B")) or name.startswith("score."))

    def test_topk_preserves_execution_order(self):
        model = TinyClassifier(layers=4)
        original = list(model.layers)
        evaluations = []
        def evaluate():
            kept = [original.index(m) for m in model.layers]
            evaluations.append(kept)
            return {"accuracy": 0.9 if len(kept) >= 2 else 0.8}
        result = select_layers(model, [0.1, 0.9, 0.3, 0.8], evaluate)
        self.assertEqual(result["selected"]["kept_original_indices"], [1, 3])
        self.assertEqual(evaluations, [[0, 1, 2, 3], [1], [1, 3]])

    def test_rejected_pruning_restores_model(self):
        model = TinyClassifier(layers=4)
        original = list(model.layers)
        def evaluate():
            return {"accuracy": 1.0 if len(model.layers) == 4 else 0.1}
        with self.assertRaises(RuntimeError):
            select_layers(model, [1, 2, 3, 4], evaluate, remove_layers=3)
        self.assertEqual(list(model.layers), original)

    def test_metrics(self):
        metrics = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)

    def test_stratified_split_disjoint(self):
        rows = synthetic_records()
        splits = split_records(rows)
        self.assertEqual([len(s) for s in splits], [56, 8, 16])
        ids = [set(r["id"] for r in s) for s in splits]
        self.assertFalse(ids[0] & ids[1] or ids[1] & ids[2] or ids[0] & ids[2])
        self.assertEqual(set.union(*ids), set(r["id"] for r in rows))

    def test_reject_duplicate_dataset(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path = Path(temp) / "data.jsonl"
            rows = synthetic_records()
            dump_jsonl(path, rows + [rows[0]])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_records(path)

    def test_offline_artifact_reload_and_full_merge(self):
        cfg = Config(dropout=0)
        model = TinyClassifier(layers=2, max_length=16)
        install_loqdc(model, cfg)
        for m in model.modules():
            if isinstance(m, QuantizedLoRALinear):
                m.B.data.normal_(std=0.01)
        model.eval()
        ids = torch.tensor([[2, 3, 4, 1]])
        mask = torch.ones_like(ids)
        expected = model(ids, mask)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            save_artifact(model, ByteTokenizer(), cfg, Path(temp), {})
            restored, _, _, _ = load_artifact(temp, torch.device("cpu"))
            restored.eval()
            torch.testing.assert_close(expected, restored(ids, mask))
            merge_adapters(restored)
            torch.testing.assert_close(expected, restored(ids, mask), rtol=1e-4, atol=1e-5)

    def test_hf_llama_reload_pruning_and_merged_export(self):
        from transformers import LlamaConfig, LlamaForSequenceClassification
        from data_io import HFTokenizer
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast
        config = LlamaConfig(vocab_size=10, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                             num_labels=2, pad_token_id=0, use_cache=False)
        model = LlamaForSequenceClassification(config)
        cfg = Config(dropout=0)
        install_loqdc(model, cfg)
        select_layers(model, [0.1, 0.9, 0.8], lambda: {"accuracy": 1.0}, remove_layers=1)
        self.assertEqual([m.self_attn.layer_idx for m in decoder_layers(model)], [0, 1])
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1},
                                                                                 unk_token="[UNK]")),
                                           pad_token="[PAD]", unk_token="[UNK]")
        ids = torch.tensor([[2, 3, 4], [3, 4, 0]])
        mask = ids.ne(0).long()
        model.eval()
        expected = model(ids, mask).logits
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path = Path(temp)
            save_artifact(model, HFTokenizer(tokenizer), cfg, path, {})
            restored, _, _, _ = load_artifact(path, torch.device("cpu"))
            restored.eval()
            torch.testing.assert_close(expected, restored(ids, mask).logits)
            merge_adapters(restored)
            restored.save_pretrained(path / "hf")
            dense = LlamaForSequenceClassification.from_pretrained(path / "hf")
            torch.testing.assert_close(expected, dense(ids, mask).logits, rtol=1e-4, atol=1e-5)

    def test_local_hf_training_entrypoint(self):
        from transformers import LlamaConfig, LlamaForSequenceClassification, PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from run import parser, train_pipeline
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path = Path(temp)
            backbone = path / "backbone"
            config = LlamaConfig(vocab_size=10, hidden_size=16, intermediate_size=32,
                                 num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                 num_labels=2, pad_token_id=0, use_cache=False)
            LlamaForSequenceClassification(config).save_pretrained(backbone)
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1}, unk_token="[UNK]")),
                pad_token="[PAD]", unk_token="[UNK]")
            tokenizer.save_pretrained(backbone)
            dump_jsonl(path / "data.jsonl", synthetic_records())
            args = parser().parse_args(["train", "--model", str(backbone), "--data", str(path / "data.jsonl"),
                                        "--device", "cpu", "--epochs", "1", "--max-length", "16",
                                        "--output", str(path / "output"), "--export-hf", "--gradient-checkpointing"])
            output = train_pipeline(args)
            restored, _, _, _ = load_artifact(output / "checkpoint", torch.device("cpu"))
            self.assertGreater(len(decoder_layers(restored)), 0)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["test"]["n"], 16)
            self.assertTrue((output / "merged_hf" / "config.json").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
