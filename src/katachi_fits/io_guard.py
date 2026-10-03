"""Escritura segura: nada se escribe dentro de las rutas de solo lectura.

Garantías del plan §6:
- Solo se escribe bajo `results_root`.
- Nunca dentro del catálogo ni de la carpeta de FITS.
- No se sobrescribe un archivo existente, salvo los marcados como "rodantes"
  (el checkpoint `last.pt`), que se reemplazan de forma atómica.
- Este módulo no expone ninguna operación de borrado.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, Iterable


class WriteGuardError(RuntimeError):
    """Intento de escribir donde no está permitido."""
    pass


class WriteGuard:
    """Única puerta de escritura del proyecto: solo bajo la carpeta de resultados, nunca en rutas de solo lectura, sin sobrescribir."""
    def __init__(self, results_root: Path, read_only: Iterable[Path]):
        self.root = Path(results_root).resolve()
        self.read_only = [Path(p).resolve() for p in read_only]
        for ro in self.read_only:
            if _is_within(self.root, ro):
                raise WriteGuardError(f"La carpeta de resultados {self.root} está dentro de una ruta de solo lectura {ro}")

    def check(self, path: Path, *, allow_replace: bool = False) -> Path:
        """Valida una ruta de escritura y la devuelve resuelta; lanza WriteGuardError si no está permitida."""
        p = Path(path).resolve()
        if not _is_within(p, self.root):
            raise WriteGuardError(f"Escritura fuera de la carpeta de resultados: {p}")
        for ro in self.read_only:
            if p == ro or _is_within(p, ro):
                raise WriteGuardError(f"Escritura en ruta de solo lectura: {p}")
        if p.exists() and not allow_replace:
            raise WriteGuardError(f"El archivo ya existe y no se sobrescribe: {p}")
        return p

    def mkdir(self, path: Path) -> Path:
        """Crea una carpeta dentro de la carpeta de resultados."""
        p = Path(path).resolve()
        if not _is_within(p, self.root) and p != self.root:
            raise WriteGuardError(f"Carpeta fuera de resultados: {p}")
        p.mkdir(parents=True, exist_ok=True)
        return p

    def write(self, path: Path, writer: Callable[[Path], None], *, allow_replace: bool = False) -> Path:
        """Escribe con `writer(tmp_path)` y mueve atómicamente a `path`."""
        p = self.check(path, allow_replace=allow_replace)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.", suffix=".tmp")
        os.close(fd)
        try:
            writer(Path(tmp))
            os.replace(tmp, p)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)  # solo el temporal propio
        return p

    def write_text(self, path: Path, text: str, *, allow_replace: bool = False) -> Path:
        return self.write(path, lambda t: Path(t).write_text(text, encoding="utf-8"), allow_replace=allow_replace)

    def write_json(self, path: Path, obj, *, allow_replace: bool = False) -> Path:
        return self.write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, default=str), allow_replace=allow_replace)

    def download(self, url: str, path: Path) -> Path:
        """Descarga si no existe; si existe, la reutiliza."""
        p = Path(path)
        if p.exists() and p.stat().st_size > 0:
            return p

        def fetch(tmp: Path) -> None:
            with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 22):
                    f.write(chunk)

        return self.write(p, fetch)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    """Huella SHA-256 de un archivo."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()
