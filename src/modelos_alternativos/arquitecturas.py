"""Arquitecturas de los modelos alternativos (Avance 4), independientes de la variante.

Un modelo se describe con una `EspecificacionModelo` y se construye con `construir_modelo`:

    imagen ──► Extractor ──► vector (d)
                                 │
    escalares (k, puede ser 0) ──┴─► concatenación (d + k) ──► CabezaMultisalida ──► salidas

- El extractor es cualquier red de timm sin clasificador. Para los ViT se elige cómo resumir los
  tokens (CLS, promedio de parches o ambos); para las redes convolucionales, timm ya promedia.
- Las variables escalares (por ejemplo, el redshift) son opcionales; el modelo no conoce su
  significado.
- La estrategia de ajuste (`completo`, `congelado`, `ultimos_bloques`, `lora`) se aplica al
  extractor sin importar su familia.

Ninguna clase de este módulo sabe qué propiedad física predice ni qué variable auxiliar recibe.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import timm
import torch
from torch import nn

ESTRATEGIAS_AJUSTE = ("completo", "congelado", "ultimos_bloques", "lora")
AGREGACIONES_TOKENS = ("cls", "promedio", "cls_y_promedio")


# ---------------------------------------------------------------------------
# Especificación declarativa
# ---------------------------------------------------------------------------
@dataclass
class EspecificacionModelo:
    """Todo lo que define la arquitectura de un modelo, sin código propio de una variante."""
    nombre_extractor: str                          # nombre de timm con sus pesos, p. ej. "convnext_tiny.fb_in22k_ft_in1k"
    preentrenado: bool = True
    tasa_drop_path: float = 0.0                    # stochastic depth del extractor
    agregacion_tokens: str = "cls_y_promedio"      # solo para extractores de tokens (ViT)
    estrategia_ajuste: str = "completo"            # ver ESTRATEGIAS_AJUSTE
    bloques_abiertos: int = 0                      # para "ultimos_bloques"
    rango_lora: int = 8                            # para "lora"
    alfa_lora: float = 16.0
    capas_lora: tuple = ("qkv", "proj")
    unidades_ocultas: int = 256                    # cabeza (protocolo común)
    dropout_cabeza: float = 0.1
    lado_entrada: int | None = None                # tamaño de imagen con que se crea el extractor (ViT)

    def __post_init__(self):
        if self.estrategia_ajuste not in ESTRATEGIAS_AJUSTE:
            raise ValueError(f"estrategia_ajuste debe ser una de {ESTRATEGIAS_AJUSTE}")
        if self.agregacion_tokens not in AGREGACIONES_TOKENS:
            raise ValueError(f"agregacion_tokens debe ser una de {AGREGACIONES_TOKENS}")

    def a_dict(self) -> dict:
        d = asdict(self); d["capas_lora"] = list(self.capas_lora)
        return d


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------
class Extractor(nn.Module):
    """Red de timm sin clasificador que devuelve un vector por imagen, con atributo `dimension`."""

    def __init__(self, red: nn.Module, agregacion_tokens: str = "cls_y_promedio"):
        super().__init__()
        self.red = red
        self.es_de_tokens = hasattr(red, "blocks") and hasattr(red, "forward_features") and hasattr(red, "patch_embed")
        self.agregacion_tokens = agregacion_tokens
        self.tokens_prefijo = int(getattr(red, "num_prefix_tokens", 1))
        factor = 2 if (self.es_de_tokens and agregacion_tokens == "cls_y_promedio") else 1
        self.dimension = red.num_features * factor

    def forward(self, imagen: torch.Tensor) -> torch.Tensor:
        if not self.es_de_tokens:
            return self.red(imagen)                         # timm ya aplica el promedio global
        tokens = self.red.forward_features(imagen)
        cls, parches = tokens[:, 0], tokens[:, self.tokens_prefijo:].mean(1)
        if self.agregacion_tokens == "cls":
            return cls
        if self.agregacion_tokens == "promedio":
            return parches
        return torch.cat([cls, parches], 1)

    def bloques(self) -> list[nn.Module]:
        """Unidades en orden de profundidad, para abrir las últimas: bloques de un ViT o de las etapas de una CNN."""
        if hasattr(self.red, "blocks"):
            return list(self.red.blocks)
        if hasattr(self.red, "stages"):
            return [b for etapa in self.red.stages for b in getattr(etapa, "blocks", [etapa])]
        if hasattr(self.red, "layers"):
            return list(self.red.layers)
        raise ValueError("No se reconocen los bloques de este extractor")


def crear_extractor(espec: EspecificacionModelo) -> Extractor:
    argumentos = {"pretrained": espec.preentrenado, "num_classes": 0, "drop_path_rate": espec.tasa_drop_path}
    arquitectura = espec.nombre_extractor.split(".")[0]
    if "vit" in arquitectura or "eva" in arquitectura:       # los ViT aceptan tamaños de imagen variables
        argumentos["dynamic_img_size"] = True
        if espec.lado_entrada:
            argumentos["img_size"] = espec.lado_entrada
    return Extractor(timm.create_model(espec.nombre_extractor, **argumentos), espec.agregacion_tokens)


# ---------------------------------------------------------------------------
# Cabeza y modelo
# ---------------------------------------------------------------------------
class CabezaMultisalida(nn.Module):
    """Una cabeza independiente por salida: Linear(d, u) → ReLU → Dropout → Linear(u, 1)."""

    def __init__(self, dimension_entrada: int, n_salidas: int, unidades_ocultas: int = 256, dropout: float = 0.1):
        super().__init__()
        self.cabezas = nn.ModuleList([nn.Sequential(nn.Linear(dimension_entrada, unidades_ocultas), nn.ReLU(),
                                                    nn.Dropout(dropout), nn.Linear(unidades_ocultas, 1))
                                      for _ in range(n_salidas)])

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        return torch.cat([c(vector) for c in self.cabezas], 1)


class ModeloRegresionImagen(nn.Module):
    """Extractor de imagen + variables escalares opcionales + cabeza multisalida."""

    def __init__(self, extractor: Extractor, n_escalares: int, n_salidas: int, unidades_ocultas: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.extractor, self.n_escalares = extractor, n_escalares
        self.cabeza = CabezaMultisalida(extractor.dimension + n_escalares, n_salidas, unidades_ocultas, dropout)

    def forward(self, imagen: torch.Tensor, escalares: torch.Tensor | None = None) -> torch.Tensor:
        vector = self.extractor(imagen)
        if self.n_escalares:
            vector = torch.cat([vector, escalares.to(vector.dtype)], 1)
        return self.cabeza(vector)


# ---------------------------------------------------------------------------
# Estrategias de ajuste del extractor
# ---------------------------------------------------------------------------
def congelar(modulo: nn.Module) -> None:
    for p in modulo.parameters():
        p.requires_grad = False


class LinealLoRA(nn.Module):
    """Capa lineal congelada más una corrección de bajo rango B·A (Hu et al. 2022). B empieza en cero."""

    def __init__(self, base: nn.Linear, rango: int = 8, alfa: float = 16.0):
        super().__init__()
        self.base, self.escala = base, alfa / rango
        congelar(self.base)
        self.lora_a = nn.Parameter(torch.empty(rango, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rango))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_a.t() @ self.lora_b.t()) * self.escala


def _insertar_lora(modulo: nn.Module, capas: tuple, rango: int, alfa: float) -> int:
    reemplazadas = 0
    for nombre, hijo in list(modulo.named_children()):
        if isinstance(hijo, nn.Linear) and nombre in capas:
            setattr(modulo, nombre, LinealLoRA(hijo, rango, alfa)); reemplazadas += 1
        else:
            reemplazadas += _insertar_lora(hijo, capas, rango, alfa)
    return reemplazadas


def aplicar_estrategia_ajuste(extractor: Extractor, espec: EspecificacionModelo) -> dict:
    """Deja entrenables solo los parámetros que indica la estrategia. Devuelve un resumen."""
    estrategia = espec.estrategia_ajuste
    if estrategia == "completo":
        for p in extractor.parameters():
            p.requires_grad = True
    elif estrategia == "congelado":
        congelar(extractor)
    elif estrategia == "ultimos_bloques":
        congelar(extractor)
        bloques = extractor.bloques()
        for bloque in bloques[len(bloques) - espec.bloques_abiertos:] if espec.bloques_abiertos > 0 else []:
            for p in bloque.parameters():
                p.requires_grad = True
        for nombre in ("norm", "fc_norm", "norm_pre", "head"):
            capa = getattr(extractor.red, nombre, None)
            if isinstance(capa, nn.Module):
                for p in capa.parameters():
                    p.requires_grad = True
    elif estrategia == "lora":
        congelar(extractor)
        n = _insertar_lora(extractor.red, tuple(espec.capas_lora), espec.rango_lora, espec.alfa_lora)
        if n == 0:
            raise ValueError(f"No se encontraron capas {espec.capas_lora} para LoRA en {espec.nombre_extractor}")
    return {"estrategia_ajuste": estrategia,
            "parametros_entrenables_extractor": int(sum(p.numel() for p in extractor.parameters() if p.requires_grad))}


def construir_modelo(espec: EspecificacionModelo, n_escalares: int, n_salidas: int) -> ModeloRegresionImagen:
    extractor = crear_extractor(espec)
    aplicar_estrategia_ajuste(extractor, espec)
    return ModeloRegresionImagen(extractor, n_escalares, n_salidas, espec.unidades_ocultas, espec.dropout_cabeza)


# ---------------------------------------------------------------------------
# Grupos de parámetros para el optimizador
# ---------------------------------------------------------------------------
def grupos_de_parametros(modelo: ModeloRegresionImagen, tasa_extractor: float, tasa_cabeza: float,
                         decaimiento_pesos: float, decaimiento_por_capa: float = 1.0) -> list[dict]:
    """Grupos de AdamW: extractor con decaimiento de la tasa por capa y cabeza con su propia tasa.

    El decaimiento por capa multiplica la tasa de cada capa por `decaimiento_por_capa` elevado a su
    distancia a la salida (timm, `param_groups_layer_decay`), de modo que las capas iniciales, más
    genéricas, cambian menos. Sesgos y normalizaciones no llevan decaimiento de pesos. Solo se
    incluyen parámetros entrenables.
    """
    from timm.optim import param_groups_layer_decay

    red = modelo.extractor.red
    grupos, vistos = [], set()
    if hasattr(red, "group_matcher") and decaimiento_por_capa < 1.0:
        sin_decaimiento = set(getattr(red, "no_weight_decay", lambda: set())())
        for g in param_groups_layer_decay(red, weight_decay=decaimiento_pesos, layer_decay=decaimiento_por_capa,
                                          no_weight_decay_list=sin_decaimiento):
            parametros = [p for p in g["params"] if p.requires_grad]
            if parametros:
                grupos.append({"params": parametros, "lr": tasa_extractor * g["lr_scale"],
                               "weight_decay": g["weight_decay"], "grupo": "extractor"})
                vistos.update(id(p) for p in parametros)
    restantes = [(n, p) for n, p in modelo.extractor.named_parameters() if p.requires_grad and id(p) not in vistos]
    for con_decaimiento in (True, False):
        parametros = [p for n, p in restantes if (p.ndim > 1) == con_decaimiento]
        if parametros:
            grupos.append({"params": parametros, "lr": tasa_extractor,
                           "weight_decay": decaimiento_pesos if con_decaimiento else 0.0, "grupo": "extractor"})
    for con_decaimiento in (True, False):
        parametros = [p for p in modelo.cabeza.parameters() if (p.ndim > 1) == con_decaimiento]
        grupos.append({"params": parametros, "lr": tasa_cabeza,
                       "weight_decay": decaimiento_pesos if con_decaimiento else 0.0, "grupo": "cabeza"})
    return grupos


def contar_parametros(modelo: nn.Module) -> dict:
    return {"parametros_totales": int(sum(p.numel() for p in modelo.parameters())),
            "parametros_entrenables": int(sum(p.numel() for p in modelo.parameters() if p.requires_grad))}
