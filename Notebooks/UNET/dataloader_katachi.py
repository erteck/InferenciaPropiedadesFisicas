import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

# Tu importación experta corregida en 2D
from dataloaders.preprocesamiento_katachi import FITSPipelinePyTorch

class glx_dataset_katachi(Dataset):
    def __init__(self, path_fits):
        '''
        1. Almaceno la ruta del directorio de las imágenes FITS crudas.
        2. Cargo el catálogo CSV maestro oficial del equipo (MaNGA/Pipe3D).
        3. Filtro la lista de archivos para asegurar consistencia geométrica.
        4. Instancio el pipeline del preprocesamiento astronómico.
        '''
        self.pfits = Path(path_fits)
        #self.p = self.pfits / Path('FITS')
        self.fits = [f for f in self.pfits.iterdir() if f.suffix.lower() == ".fits"]
        self.glx_prep = FITSPipelinePyTorch()

    def __len__(self):
        return len(self.fits)

    def __getitem__(self, idx):
        ruta_fits = self.fits[idx]
        
        # Extraemos el identificador exacto de la galaxia (ej: plateifu) usando el nombre del archivo
        nm = ruta_fits.name.split('_')[0]
        
        # Invocación de tu pipeline de Astropy y Photutils
        insumos_pytorch = self.glx_prep.fits_pytorch(
            fits_path_or_data=ruta_fits, 
            sigma_gauss=3.0
        )
        
        # Extraemos los tensores ópticos puros en 2D enmascarados (1, 256, 256)
        tensor_g = torch.tensor(insumos_pytorch['chan_g'], dtype=torch.float32)
        tensor_r = torch.tensor(insumos_pytorch['chan_r'], dtype=torch.float32)
        tensor_z = torch.tensor(insumos_pytorch['chan_z'], dtype=torch.float32)
        tensor_grz = torch.tensor(insumos_pytorch['chan_grz'], dtype=torch.float32)
        
        # El Dataset ahora entrega de forma limpia: (Input de 3 canales, Vector de 3 targets)
        return tensor_g, tensor_r, tensor_z, tensor_grz
