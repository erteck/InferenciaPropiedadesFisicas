"""Pruebas del congelado del artículo (Documentación/Propuesta_congelado_Katachi.md).

Usan los pesos reales de ImageNet V1 (torchvision los descarga una vez) para comprobar,
bit a bit, que lo congelado no cambia al entrenar y que lo entrenable sí."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torchvision

from src.katachi_fits.chain import Chain, SplitInputConv, frozen_param_names
from src.katachi_fits.config import TrainConfig
from src.katachi_fits.io_guard import WriteGuard
from src.katachi_fits.train import GPUData, Trainer, freezing_audit
from src.katachi_fits.transforms import Normalizer, band_permutation, network_input

NORM = Normalizer(np.ones(3) * 0.01, np.ones(3) * 5, 3.0)


def _data(n=8, px=32, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=0.02, size=(n, 3, px, px)).astype(np.float32)
    y = np.stack([10 + rng.normal(0, .5, n), rng.normal(0, .5, n), 1.5 + rng.normal(0, .1, n)], 1).astype(np.float32)
    return GPUData(x, y, torch.device("cpu"))


def _trainer(tmp_path, **kw):
    cfg = TrainConfig(seed=0, batch_size=4, max_epochs=2, fused_adam=False, channels_last=False,
                      cudnn_benchmark=False, **kw)
    g = WriteGuard(tmp_path, read_only=[tmp_path / "ro"])
    return Trainer(cfg, NORM, 32, tmp_path / "run", g, torch.device("cpu"), log=lambda *_: None)


def test_frozen_layers_stay_identical_to_imagenet(tmp_path):
    t = _trainer(tmp_path)
    t.fit(_data(), None)
    imagenet = torchvision.models.resnet50(weights="IMAGENET1K_V1").state_dict()
    audit = freezing_audit(t.model, 144, imagenet)
    for _, r in audit.iterrows():
        ok, tot = map(int, r["congelados idénticos a ImageNet"].split("/"))
        cambio, tot_e = map(int, r["entrenables modificados"].split("/"))
        assert ok == tot == 288, r.to_dict()     # 144 parámetros + 144 buffers de BatchNorm (48 capas × 3)
        assert cambio == tot_e, r.to_dict()                    # todo lo entrenable se movió
    assert audit["parámetros entrenables"].tolist() == [7877633, 7880769, 7883905]


def test_first_conv_image_part_frozen_extra_channels_learn(tmp_path):
    t = _trainer(tmp_path)
    w0 = torchvision.models.resnet50(weights="IMAGENET1K_V1").conv1.weight
    assert torch.count_nonzero(t.model.sfr.conv1.conv_extra.weight) == 0      # empieza en cero
    t.fit(_data(), None)
    for net in (t.model.sfr, t.model.d4000):
        assert torch.equal(net.conv1.conv_img.weight, w0)
        assert torch.count_nonzero(net.conv1.conv_extra.weight) > 0
    assert torch.equal(t.model.mass.conv1.weight, w0)


def test_split_conv_equals_single_conv():
    torch.manual_seed(0)
    s = SplitInputConv(nn.Conv2d(3, 8, 7, 2, 3, bias=False), 2)
    nn.init.normal_(s.conv_extra.weight)
    single = nn.Conv2d(5, 8, 7, 2, 3, bias=False)
    single.weight.data = s.as_single_weight().detach().clone()
    x = torch.rand(2, 5, 32, 32)
    torch.testing.assert_close(s(x), single(x))


def test_stopped_network_is_fixed(tmp_path):
    t = _trainer(tmp_path)
    t.net_stopped = [True, False, False]
    antes = copy.deepcopy(t.model.mass.state_dict())
    antes_sfr = copy.deepcopy(t.model.sfr.state_dict())
    t.train_epoch(_data())
    for k, v in t.model.mass.state_dict().items():
        assert torch.equal(v, antes[k]), k                     # pesos y buffers de BatchNorm intactos
    assert any(not torch.equal(v, antes_sfr[k]) for k, v in t.model.sfr.state_dict().items())


def test_frozen_batchnorm_stays_in_eval_mode(tmp_path):
    t = _trainer(tmp_path)
    t._set_train_mode()
    congeladas = [m for m in t.model.modules() if getattr(m, "_frozen_stats", False)]
    assert len(congeladas) == 3 * 48 and all(not m.training for m in congeladas)
    entrenables = [m for m in t.model.modules() if isinstance(m, nn.BatchNorm2d) and not getattr(m, "_frozen_stats", False)]
    assert all(m.training for m in entrenables)


def test_band_order_z_r_g():
    assert band_permutation(("z", "r", "g")) == [2, 1, 0]
    x = torch.stack([torch.full((4, 4), v) for v in (1.0, 2.0, 3.0)])[None] * 0.01   # g=1, r=2, z=3
    out = network_input(x, NORM, 4, ("z", "r", "g"))
    assert out[0, 0, 0, 0] > out[0, 1, 0, 0] > out[0, 2, 0, 0]                        # canal 0 = z


def test_control_run_trains_everything(tmp_path):
    t = _trainer(tmp_path, freeze_pretrained=False, per_network=False)
    assert all(p.requires_grad for p in t.model.parameters())
    assert t.model.sfr.conv1.weight.shape[1] == 4                                     # como los pesos publicados


def test_frozen_names_match_katachi_threshold():
    names = frozen_param_names(144)
    assert len(names) == 144 and "layer4.1.conv1.weight" in names and "layer4.1.conv2.weight" not in names


def test_lr_is_paper_value():
    assert TrainConfig().lr == (1e-3, 1e-3, 1e-3)
