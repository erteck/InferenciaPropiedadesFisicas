"""Cadena de Katachi: tres ResNet50 (M* → SFR → D4000), plan §3.4.

- ImageNet V1; `fc = Linear(2048, 1)`.
- Sin congelado (modelos publicados de Katachi): las redes de SFR y D4000 reemplazan
  `conv1` por una convolución nueva aleatoria de 4 y 5 canales.
- Con congelado (artículo §3.1.3; Documentación/Propuesta_congelado_Katachi.md):
  · se congelan los parámetros preentrenados con índice < 144 en el orden de la
    ResNet50 (D1) y sus BatchNorm quedan en modo evaluación (D2);
  · en SFR y D4000, `conv1` se divide en la parte de imagen (pesos de ImageNet,
    congelada) y la de los canales extra (iniciada en cero, entrenable) (D3).
- La predicción previa se agrega, sin normalizar y desacoplada (`detach`), como un
  plano constante.
- `GhostBatchNorm2d` reproduce `nn.DataParallel` en 2 GPUs: estadísticas por cada
  mitad del lote (troceo de `torch.chunk`) y estadísticas acumuladas solo de la
  primera mitad (la réplica 0 usa los buffers originales).
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class GhostBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm que normaliza cada mitad del lote por separado, como `nn.DataParallel` en 2 GPUs."""
    def __init__(self, *args, splits: int = 2, **kw):
        super().__init__(*args, **kw)
        self.splits = splits

    @classmethod
    def from_bn(cls, bn: nn.BatchNorm2d, splits: int) -> "GhostBatchNorm2d":
        g = cls(bn.num_features, eps=bn.eps, momentum=bn.momentum, affine=bn.affine,
                track_running_stats=bn.track_running_stats, splits=splits)
        g.load_state_dict(bn.state_dict())
        return g.to(bn.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.splits <= 1 or x.shape[0] < 2:
            return super().forward(x)
        parts = torch.chunk(x, min(self.splits, x.shape[0]), dim=0)
        self.num_batches_tracked.add_(1)
        out = [F.batch_norm(parts[0], self.running_mean, self.running_var, self.weight, self.bias,
                            True, self.momentum, self.eps)]
        for p in parts[1:]:
            out.append(F.batch_norm(p, None, None, self.weight, self.bias, True, 0.0, self.eps))
        return torch.cat(out, 0)


def ghostify(module: nn.Module, splits: int) -> nn.Module:
    """Reemplaza todas las BatchNorm2d de un módulo por GhostBatchNorm2d."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d) and not isinstance(child, GhostBatchNorm2d):
            setattr(module, name, GhostBatchNorm2d.from_bn(child, splits))
        else:
            ghostify(child, splits)
    return module


def resnet50(in_ch: int, imagenet: bool = True) -> nn.Module:
    """ResNet50 con salida escalar; si `in_ch` ≠ 3, primera capa nueva aleatoria (como Katachi)."""
    m = models.resnet50(weights="IMAGENET1K_V1" if imagenet else None)
    if in_ch != 3:
        m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(2048, 1)
    return m


class SplitInputConv(nn.Module):
    """conv1 de 4 o 5 canales escrita como suma de dos convoluciones: la de imagen
    (3 canales, pesos preentrenados) y la de los canales extra (masa, SFR), iniciada
    en cero. Es matemáticamente igual a una sola Conv2d con los pesos concatenados,
    pero permite congelar solo la parte preentrenada."""

    def __init__(self, conv_img: nn.Conv2d, n_extra: int):
        super().__init__()
        self.conv_img = conv_img
        self.conv_extra = nn.Conv2d(n_extra, conv_img.out_channels, kernel_size=conv_img.kernel_size,
                                    stride=conv_img.stride, padding=conv_img.padding, bias=False)
        nn.init.zeros_(self.conv_extra.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_img(x[:, :3]) + self.conv_extra(x[:, 3:])

    def as_single_weight(self) -> torch.Tensor:
        return torch.cat([self.conv_img.weight, self.conv_extra.weight], 1)


def frozen_param_names(first_trainable: int) -> set[str]:
    """Nombres de los parámetros de una ResNet50 estándar con índice < first_trainable
    (orden de `model.parameters()`, el mismo del bucle de Katachi)."""
    ref = models.resnet50(weights=None)
    ref.fc = nn.Linear(2048, 1)
    return {n for i, (n, _) in enumerate(ref.named_parameters()) if i < first_trainable}


def freeze_pretrained(net: nn.Module, first_trainable: int, freeze_bn_stats: bool = True) -> dict:
    """Congela en `net` los parámetros preentrenados con índice < first_trainable.
    `conv1.weight` de una SplitInputConv corresponde a su parte de imagen; los canales
    extra quedan entrenables. Marca las BatchNorm congeladas para que siempre corran en
    modo evaluación. Devuelve el conteo de parámetros congelados y entrenables."""
    frozen = frozen_param_names(first_trainable)
    alias = {"conv1.conv_img.weight": "conv1.weight"}
    for name, p in net.named_parameters():
        p.requires_grad = alias.get(name, name) not in frozen
    for mod_name, mod in net.named_modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod._frozen_stats = freeze_bn_stats and not mod.weight.requires_grad
    n_frozen = sum(p.numel() for p in net.parameters() if not p.requires_grad)
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {"congelados": n_frozen, "entrenables": n_train}


def apply_frozen_bn(module: nn.Module) -> None:
    """Pone en modo evaluación las BatchNorm marcadas como congeladas."""
    for m in module.modules():
        if getattr(m, "_frozen_stats", False):
            m.eval()


def add_plane(x: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Agrega `value` (B,1) como plano constante (B,1,H,W)."""
    plane = value.detach().view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3]).to(x.dtype)
    return torch.cat([x, plane], 1)


class Chain(nn.Module):
    """Las tres ResNet50 de Katachi encadenadas: M* → SFR → D4000.

    `freeze_first=None` reproduce la arquitectura de los modelos publicados (todo
    entrenable, conv1 nueva aleatoria). Con `freeze_first=144` se aplica el congelado
    del artículo (ver la docstring del módulo)."""

    def __init__(self, imagenet: bool = True, ghost_splits: int = 2, freeze_first: int | None = None,
                 freeze_bn_stats: bool = True):
        super().__init__()
        self.mass = resnet50(3, imagenet)
        if freeze_first is None:
            self.sfr = resnet50(4, imagenet)
            self.d4000 = resnet50(5, imagenet)
        else:
            self.sfr = resnet50(3, imagenet)
            self.d4000 = resnet50(3, imagenet)
            self.sfr.conv1 = SplitInputConv(self.sfr.conv1, 1)
            self.d4000.conv1 = SplitInputConv(self.d4000.conv1, 2)
        if ghost_splits > 1:
            for m in (self.mass, self.sfr, self.d4000):
                ghostify(m, ghost_splits)
        self.freeze_report = {}
        if freeze_first is not None:
            for name in ("mass", "sfr", "d4000"):
                self.freeze_report[name] = freeze_pretrained(getattr(self, name), freeze_first, freeze_bn_stats)

    def nets(self) -> tuple[nn.Module, nn.Module, nn.Module]:
        return self.mass, self.sfr, self.d4000

    def train(self, mode: bool = True):
        """Como `nn.Module.train`, pero las BatchNorm congeladas siguen en evaluación."""
        super().train(mode)
        apply_frozen_bn(self)
        return self

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p1 = self.mass(x)
        x4 = add_plane(x, p1)
        p2 = self.sfr(x4)
        x5 = add_plane(x4, p2)
        p3 = self.d4000(x5)
        return p1, p2, p3


class ChainTarget(nn.Module):
    """Imagen → una salida de la cadena, sin `detach`, para SHAP (efecto total del
    píxel sobre la predicción, incluido el camino por las redes previas)."""

    def __init__(self, chain: Chain, target: int):
        super().__init__()
        self.chain, self.target = chain, target

    def forward(self, x):
        p1 = self.chain.mass(x)
        if self.target == 0:
            return p1
        plane = lambda v: v.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        x4 = torch.cat([x, plane(p1)], 1)
        p2 = self.chain.sfr(x4)
        if self.target == 1:
            return p2
        return self.chain.d4000(torch.cat([x4, plane(p2)], 1))


def load_katachi_state(path: Path) -> OrderedDict:
    """Pesos de un checkpoint de Katachi sin el prefijo `module.` de DataParallel."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    return OrderedDict((k.replace("module.", ""), v) for k, v in sd.items())


def katachi_chain(mass: Path, sfr: Path, d4000: Path) -> Chain:
    """Cadena publicada de Katachi (B0), sin Ghost BN (solo evaluación)."""
    c = Chain(imagenet=False, ghost_splits=1)
    c.mass.load_state_dict(load_katachi_state(mass), strict=True)
    c.sfr.load_state_dict(load_katachi_state(sfr), strict=True)
    c.d4000.load_state_dict(load_katachi_state(d4000), strict=True)
    return c.eval()


def katachi_solo(path: Path) -> nn.Module:
    """Red solo-imagen publicada por Katachi (las que usó para sus mapas SHAP)."""
    m = resnet50(3, imagenet=False)
    m.load_state_dict(load_katachi_state(path), strict=True)
    return m.eval()


class T50Net(nn.Module):
    """Red D4000→t50 publicada (`d400_to_t50.pytorch`): entradas (D4000, sSFR),
    salida t50/13.6 Gyr. La capa oculta se reutiliza 3 veces, como en Katachi."""

    def __init__(self):
        super().__init__()
        self.linear_in = nn.Linear(2, 100)
        self.linear_out = nn.Linear(100, 1)
        self.linear = nn.Linear(100, 100)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.activation(self.linear_in(x))
        for _ in range(3):
            out = self.activation(self.linear(out))
        return self.linear_out(out)
