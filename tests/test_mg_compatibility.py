import unittest
from unittest.mock import patch

import torch
from torch import nn

from model_guidance import (
    apply_model_guidance_correction,
    prepare_model_guidance_batch,
)
from models import LabelEmbedder, SiT, SiTMGAdapter
from path_energy import teacher_track_energy
from transport import create_transport


class LabelTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels_seen = []

    def forward(self, x, t, y):
        self.labels_seen.append(y.detach().cpu().clone())
        return y.to(dtype=x.dtype).view(-1, 1, 1, 1).expand_as(x)


class ZeroVelocity(nn.Module):
    def forward(self, x, t, y):
        return torch.zeros_like(x)


class ModelGuidanceCompatibilityTests(unittest.TestCase):
    def test_unconditional_slot_is_opt_in_when_dropout_is_zero(self):
        legacy = LabelEmbedder(10, 4, 0.0)
        mg = LabelEmbedder(10, 4, 0.0, always_allocate_uncond_slot=True)
        cfg = LabelEmbedder(10, 4, 0.1)
        self.assertEqual(legacy.embedding_table.num_embeddings, 10)
        self.assertEqual(mg.embedding_table.num_embeddings, 11)
        self.assertEqual(cfg.embedding_table.num_embeddings, 11)

    def test_mg_batch_preparation_matches_reference_warmup_and_active_layout(self):
        labels = torch.tensor([1, 2, 3, 4])
        warmup_labels, warmup_weights, warmup_guided = prepare_model_guidance_batch(
            labels,
            num_classes=10,
            weight_low=1.5,
            weight_high=1.5,
            drop_fraction=0.25,
            guidance_active=False,
            dtype=torch.float32,
        )
        self.assertEqual(warmup_guided, 0)
        torch.testing.assert_close(warmup_labels, torch.tensor([10, 2, 3, 4]))
        torch.testing.assert_close(warmup_weights, torch.tensor([0.0, 1.0, 1.0, 1.0]))

        active_labels, active_weights, active_guided = prepare_model_guidance_batch(
            labels,
            num_classes=10,
            weight_low=1.5,
            weight_high=1.5,
            drop_fraction=0.25,
            guidance_active=True,
            dtype=torch.float32,
        )
        self.assertEqual(active_guided, 3)
        torch.testing.assert_close(active_labels, torch.tensor([1, 2, 3, 10]))
        torch.testing.assert_close(active_weights, torch.tensor([1.5, 1.5, 1.5, 0.0]))

    def test_mg_correction_only_applies_on_data_side(self):
        target = torch.zeros(3, 1, 1, 1)
        conditional = torch.tensor([1.0, 2.0]).view(2, 1, 1, 1)
        reference = torch.tensor([5.0, 6.0]).view(2, 1, 1, 1)
        times = torch.tensor([0.1, 0.9, 0.9])
        weights = torch.tensor([1.5, 1.5, 0.0])
        corrected = apply_model_guidance_correction(
            target,
            conditional,
            reference,
            times,
            weights,
            data_side_threshold=0.75,
        )
        expected = torch.tensor([0.0, -2.0, 0.0]).view(3, 1, 1, 1)
        torch.testing.assert_close(corrected, expected)

    def test_teacher_track_default_is_exact_legacy_formula(self):
        teacher = LabelTeacher()
        gamma_t = torch.zeros(3, 1, 1, 1)
        gamma_dot_t = torch.tensor([2.0, 4.0, 8.0]).view(3, 1, 1, 1)
        labels = torch.tensor([1, 2, 3])
        times = torch.tensor([0.1, 0.5, 0.9])
        actual = teacher_track_energy(teacher, gamma_t, gamma_dot_t, labels, times)
        expected = (gamma_dot_t.flatten(1) - labels.float().view(-1, 1)).pow(2).mean(dim=1)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(len(teacher.labels_seen), 1)
        torch.testing.assert_close(teacher.labels_seen[0], labels)

    def test_teacher_track_mg_uses_guidance_and_unconditional_drop(self):
        teacher = LabelTeacher()
        gamma_t = torch.zeros(4, 1, 1, 1)
        gamma_dot_t = torch.zeros_like(gamma_t)
        labels = torch.tensor([1, 2, 3, 4])
        times = torch.tensor([0.1, 0.3, 0.9, 0.9])
        actual = teacher_track_energy(
            teacher,
            gamma_t,
            gamma_dot_t,
            labels,
            times,
            mg_active=True,
            mg_w_lo=1.5,
            mg_w_hi=1.5,
            mg_drop_frac=0.25,
            mg_data_side_threshold=0.75,
            num_classes=10,
        )
        # Guided teacher velocities are [1, -2, -0.5], and the dropped item uses
        # the unconditional label value 10.
        expected = torch.tensor([1.0, 4.0, 0.25, 100.0])
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(teacher.labels_seen[0], torch.tensor([1, 2, 3, 10]))
        torch.testing.assert_close(teacher.labels_seen[1], torch.tensor([10, 10, 10]))

    def test_native_mg_adapter_flips_time_and_sign_without_state_changes(self):
        adapter = object.__new__(SiTMGAdapter)
        nn.Module.__init__(adapter)
        x = torch.tensor([2.0])
        t = torch.tensor([0.2])
        y = torch.tensor([3])

        def fake_forward(_self, received_x, received_t, received_y):
            torch.testing.assert_close(received_x, x)
            torch.testing.assert_close(received_t, torch.tensor([0.8]))
            torch.testing.assert_close(received_y, y)
            return torch.tensor([7.0])

        with patch.object(SiT, "forward", autospec=True, side_effect=fake_forward):
            output = adapter.forward(x, t, y)
        torch.testing.assert_close(output, torch.tensor([-7.0]))

    def test_native_mg_cfg_inherits_single_adapted_forward_path(self):
        self.assertIs(SiTMGAdapter.forward_with_cfg, SiT.forward_with_cfg)

    def test_native_mg_adapter_matches_real_model_forward_and_cfg(self):
        config = dict(
            input_size=2,
            patch_size=1,
            in_channels=4,
            hidden_size=8,
            depth=1,
            num_heads=1,
            num_classes=5,
            class_dropout_prob=0.1,
            learn_sigma=False,
        )
        base = SiT(**config).eval()
        mg = SiTMGAdapter(**config).eval()
        torch.manual_seed(123)
        with torch.no_grad():
            for parameter in base.parameters():
                parameter.normal_(mean=0.0, std=0.05)
        mg.load_state_dict(base.state_dict(), strict=True)

        x = torch.randn(4, 4, 2, 2)
        t = torch.tensor([0.1, 0.3, 0.7, 0.9])
        y = torch.tensor([1, 2, 5, 5])
        with torch.no_grad():
            expected = -base(x, 1.0 - t, y)
            actual = mg(x, t, y)
            expected_cfg = -base.forward_with_cfg(x, 1.0 - t, y, 1.5)
            actual_cfg = mg.forward_with_cfg(x, t, y, 1.5)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_cfg, expected_cfg)

    def test_transport_standard_loss_is_unchanged_and_exposes_mg_intermediates(self):
        torch.manual_seed(0)
        transport = create_transport("Linear", "velocity", None)
        x1 = torch.randn(3, 2, 2, 2)
        terms = transport.training_losses(
            ZeroVelocity(),
            x1,
            model_kwargs={"y": torch.tensor([1, 2, 3])},
        )
        expected_loss = terms["ut"].flatten(1).pow(2).mean(dim=1)
        torch.testing.assert_close(terms["loss"], expected_loss)
        torch.testing.assert_close(terms["pred"], terms["model_output"])
        self.assertEqual(terms["xt"].shape, x1.shape)
        self.assertEqual(terms["t"].shape, (3,))


if __name__ == "__main__":
    unittest.main()
