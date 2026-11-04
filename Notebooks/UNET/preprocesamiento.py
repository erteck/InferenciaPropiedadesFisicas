import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import astropy
from astropy.io import fits
from astropy.visualization import ImageNormalize, AsinhStretch, LinearStretch, LogStretch, MinMaxInterval, PercentileInterval, AsymmetricPercentileInterval, make_lupton_rgb
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.convolution import Gaussian2DKernel, convolve

from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources, detect_threshold
from photutils.aperture import CircularAperture

import skimage as ski
from skimage.segmentation import inverse_gaussian_gradient, morphological_geodesic_active_contour
from skimage.measure import regionprops, shannon_entropy

from scipy import ndimage

class FITSPipelinePyTorch:
    def __init__(self, fits_path_or_name=None):
        # Instancio los tipos de normalización solicitados en el init
        self.minmax = MinMaxInterval()
        self.itqr = PercentileInterval(percentile=98)
        self.aprint = AsymmetricPercentileInterval(lower_percentile=1, upper_percentile=98)
        
        # Si se pasa una ruta en el constructor, se puede inicializar la lectura aquí.
        # De igual forma, se permite pasar la ruta o los datos directo a la función pública para compatibilidad con el DataLoader.
        if fits_path_or_name is not None:
            self.cargar_fits(fits_path_or_name)
        else:
            self.headers = None
            self.image_data = None

    def cargar_fits(self, fits_path):
        """
        Función para abrir y cargar los datos y cabeceras del archivo FITS.
        """
        fits_path = Path(fits_path)
        with fits.open(fits_path) as hdulist:
            hdulist.info()
            self.headers = fits.getheader(fits_path)
            self.image_data = fits.getdata(fits_path)

    def _metricas_fondo(self, arreglo):
        '''
        El objetivo de la función es poder detectar el valor de fondo para cada canal de las diferentes imágenes.
        Pipeline:
            1. Obtener el box_size que es dato más importante para calcular el los "bloques" en los que se va a analizar al imagen.
                1. Cálculo d ela mediana con sigma_clipped_stats
                2. Uso regionprops, sirve para hacer varios segmentos de la imagen
                3. Obtengo el valor de axis_major_length, esto lo obtengo del valor que tenga la area más grande.
                4. Calculo el tamaño del box size y filter; dejo un valor fijo de alpha 5, lo anterior basado en que si aplico un alpha de 4 ó 6 tendría box muy grande o muy pequeño, perdiendo propiedades de la galaxia
                NOTA: EL BOX_SIZE y el FILTER requieren ser números impares
            2. Cálculo del valor del fondo (cielo astronómico).
                1. Implemento photutils.background.Background2D para obtener el valor del cielo astronómico.
        Return Empaqueto todos los resultados de los arreglos son ruido, la mediana de sigma_clipped_stats, y los arreglos de los backgrounds
        '''
        for idx, arr in enumerate(arreglo):

            sc = SigmaClip(sigma=3.0)
            back_median = MedianBackground(sigma_clip=sc)

            #Métricas del cielo astronómico por canal
            _, sc_med, sc_std = sigma_clipped_stats(data=arr,sigma=3.5, maxiters=10)
            #obtengo el regionprops
            #mapa temporal
            luz = sc_med + (sc_std*2)
            mapa_temp = arr>=luz
            #obtengo el regionprops
            reg_props = regionprops(np.astype(mapa_temp,int))
            #obtención del tamaño del eje mas grande de una elipse que engloba la galaxia
            max_axis = [i.axis_major_length for i in reg_props if i.area == max([i.area for i in reg_props])][0] #resto 1 para tener el índice correcto, los labels están recorridos
            #tamaño box_size y filtros
            bs = int(np.round(max_axis / 5, 0))
            filters = int(np.round(2 * (bs / 20) + 1, 0))
            if bs %2 ==0: bs+=1
            if filters%2==0: filters +=1

            #cálculo del ruido por canal y recalculo los valores de los arreglos
            bkg = Background2D(arr, box_size=(bs, bs), filter_size=(filters, filters), 
                    sigma_clip=sc, bkg_estimator=back_median, filter_threshold=sc_med)
            arr_cielo_canal = arr - bkg.background

            #guardo el valor por cada canal
            if idx==0:
                arr_cielo_g = arr_cielo_canal
                sc_g = sc_med
                bkg_g = bkg
            elif idx==1:
                arr_cielo_r = arr_cielo_canal
                sc_r = sc_med
                bkg_r = bkg
            else:
                arr_cielo_z = arr_cielo_canal
                sc_z = sc_med
                bkg_z = bkg

        return {
            'g':{'cielo':arr_cielo_g, 'sc':sc_g, 'bkg':bkg_g},
            'r':{'cielo':arr_cielo_r, 'sc':sc_r, 'bkg':bkg_r},
            'z':{'cielo':arr_cielo_z, 'sc':sc_z, 'bkg':bkg_z},
        },{'box_size':bs, 'filters':filters}

    def _matrices_colapso(self, arreglos_limpios, bkg_params, img_data):
        '''
        Descripción del pipeline:

        Son 3 inputs:
            - los arrgelos de background canales
            - el tamaño del box_size y filter_size
            - el arreglo RAW

        1. Calcular la matriz de lupton y su background
        2. Calcular la matríz estándar
        3. Calcular la matriz estadística
        4. Diccionario con los 3 tipos de matrices
        
        Return diccionario de matrices y las métricas del fondo colapsado (rms y mediana)
        
        '''
        glx_lupton= np.stack(
            (
                arreglos_limpios['g']['cielo'],
                arreglos_limpios['r']['cielo'],
                arreglos_limpios['z']['cielo']
            )
            ,axis=0)

        glx_lupton = np.mean(glx_lupton, axis=0)

        _, _, std = sigma_clipped_stats(glx_lupton, sigma=3, maxiters=10)
        sc = SigmaClip(sigma=3.0)

        bkg = Background2D(
            glx_lupton, 
            box_size=(bkg_params['box_size'], 
            bkg_params['box_size']),
            filter_size=(bkg_params['filters'],
            bkg_params['filters']), 
            sigma_clip=sc, 
            bkg_estimator=MedianBackground(sigma_clip=sc), 
            filter_threshold=std*3
        )
        
        #proceso de Estándar
        glx_estandar = np.mean(img_data, axis=0)
        #Cálculo de la galaxia estadística
        for k,v in arreglos_limpios.items():
            if k=='g':
                norm_g = 1/(v['bkg'].background_rms**2 + 1e-10)
                canal_g = v['cielo']
            elif k=='r':
                norm_r = 1/(v['bkg'].background_rms**2+ 1e-10)
                canal_r = v['cielo']
            else:
                norm_z = 1/(v['bkg'].background_rms**2+ 1e-10)
                canal_z = v['cielo']
        #pondero los rms 
        sum_rms = norm_g + norm_r + norm_z
        glx_estadistica = (canal_g * (norm_g/sum_rms)) + (canal_r * (norm_r/sum_rms)) + (canal_z * (norm_z/sum_rms))

        dict_glx = {'Lupton': glx_lupton, 'Estandar': glx_estandar, 'Estadistica':glx_estadistica}
        return dict_glx, bkg.background_rms, bkg.background_median
        
    def _norm_asinh(self, arrs_multibanda):
        '''
        Descripción del pipeline:
            1. Cálculo de las métricas del cielo astronómico
            2. Cálculo de los valores alpha para los tipos de escalamiento (lupton, lupton estándar y estadístico)
            3. Instanciar los tipos de normalización de la imagen
            4. Normalización de la imagen
        RETURN arreglos y normalizaciones por canal

        NOTA: RECORDAR QUE SE SUMARON LSO 3 CANALES, POR LO TANTO AHORA ES SÓLO 1 ARREGLO
        '''
        
        # Se utilizan las instancias compartidas desde self en el constructor
        minmax = self.minmax
        itqr = self.itqr
        aprint = self.aprint

        for k,v in arrs_multibanda.items():
            
            # cálculo de valores alpha para hacer el AsinhStrech
            _, med_cielo, std_cielo = sigma_clipped_stats(v, sigma=3, maxiters=10)
            denominator = np.max(v)-np.min(v)
            alpha_lupton= std_cielo/denominator
            alpha_lest = std_cielo*3
            alpha_est = (med_cielo-np.min(v))/denominator

            asin_lupton = AsinhStretch(a=alpha_lupton)
            asin_lest = AsinhStretch(a=alpha_lest)
            asin_est = AsinhStretch(a=alpha_est)

            #normalización con lupton
            minmax_norm_lupton = ImageNormalize(data=v,interval=minmax, stretch=asin_lupton)
            percent_norm_lupton = ImageNormalize(data=v,interval=itqr, stretch=asin_lupton)
            aprint_norm_lupton = ImageNormalize(data=v,interval=aprint, stretch=asin_lupton)
            #normalización con lupton estándar
            minmax_norm_lest = ImageNormalize(data=v,interval=minmax, stretch=asin_lest)
            percent_norm_lest = ImageNormalize(data=v,interval=itqr, stretch=asin_lest)
            aprint_norm_lest = ImageNormalize(data=v,interval=aprint, stretch=asin_lest)
            #normalización con estadítico
            minmax_norm_est = ImageNormalize(data=v,interval=minmax, stretch=asin_est)
            percent_norm_est = ImageNormalize(data=v,interval=itqr, stretch=asin_est)
            aprint_norm_est = ImageNormalize(data=v,interval=aprint, stretch=asin_est)

            minmax_lupton=minmax_norm_lupton(v).data
            percent_lupton=percent_norm_lupton(v).data
            aprint_lupton=aprint_norm_lupton(v).data
            minmax_lest=minmax_norm_lest(v).data
            percent_lest=percent_norm_lest(v).data
            aprint_lest=aprint_norm_lest(v).data
            minmax_est=minmax_norm_est(v).data
            percent_est=percent_norm_est(v).data
            aprint_est=aprint_norm_est(v).data

            if k=='Lupton':
                dict_lupton = {
                    "MinMax Lupton":minmax_lupton, "Percentile Lupton":percent_lupton, "Asym Per Lupton":aprint_lupton,
                    "MinMax Lupton Est":minmax_lest, "Percentile Lupton Est":percent_lest, "Asym Per Lup Est":aprint_lest,
                    "MinMax Est":minmax_est, "Percentile Est":percent_est, "Asym Per Est":aprint_est
                            }
                dict_norm_lupton = {
                    "MinMax Lupton":minmax_norm_lupton, "Percentile Lupton":percent_norm_lupton, "Asym Per Lupton":aprint_norm_lupton,
                    "MinMax Lupton Est":minmax_norm_lest, "Percentile Lupton Est":percent_norm_lest, "Asym Per Lup Est":aprint_norm_lest,
                    "MinMax Est":minmax_norm_est, "Percentile Est":percent_norm_est, "Asym Per Est":aprint_norm_est
                            }
            elif k=='Estandar':
                dict_estandar = {
                    "MinMax Lupton":minmax_lupton, "Percentile Lupton":percent_lupton, "Asym Per Lupton":aprint_lupton,
                    "MinMax Lupton Est":minmax_lest, "Percentile Lupton Est":percent_lest, "Asym Per Lup Est":aprint_lest,
                    "MinMax Est":minmax_est, "Percentile Est":percent_est, "Asym Per Est":aprint_est
                            }
                dict_norm_estandar = {
                    "MinMax Lupton":minmax_norm_lupton, "Percentile Lupton":percent_norm_lupton, "Asym Per Lupton":aprint_norm_lupton,
                    "MinMax Lupton Est":minmax_norm_lest, "Percentile Lupton Est":percent_norm_lest, "Asym Per Lup Est":aprint_norm_lest,
                    "MinMax Est":minmax_norm_est, "Percentile Est":percent_norm_est, "Asym Per Est":aprint_norm_est
                            }
            else:
                dict_estadistica = {
                    "MinMax Lupton":minmax_lupton, "Percentile Lupton":percent_lupton, "Asym Per Lupton":aprint_lupton,
                    "MinMax Lupton Est":minmax_lest, "Percentile Lupton Est":percent_lest, "Asym Per Lup Est":aprint_lest,
                    "MinMax Est":minmax_est, "Percentile Est":percent_est, "Asym Per Est":aprint_est
                            }
                dict_norm_estadistica = {
                    "MinMax Lupton":minmax_norm_lupton, "Percentile Lupton":percent_norm_lupton, "Asym Per Lupton":aprint_norm_lupton,
                    "MinMax Lupton Est":minmax_norm_lest, "Percentile Lupton Est":percent_norm_lest, "Asym Per Lup Est":aprint_norm_lest,
                    "MinMax Est":minmax_norm_est, "Percentile Est":percent_norm_est, "Asym Per Est":aprint_norm_est
                            }

        dict_arreglos = {'Lupton':dict_lupton, 'Estandar':dict_estandar, 'Estadistica':dict_estadistica}
        dict_norms = {'Lupton':dict_norm_lupton, 'Estandar':dict_norm_estandar, 'Estadistica':dict_norm_estadistica}
        
        return dict_arreglos, dict_norms
                
    def _log_norm(self, galaxias):
        '''
        Descripción del pipeline:

        NOTA:
        1. Asegurar que no tegno valores en 0.0 porque entonces no puedo normalizar el background de cada canal
        '''
        
        #Genero el grid search para logstrech
        minmax = MinMaxInterval()
        itqr = PercentileInterval(percentile=95)
        aprint = self.aprint

        #genero un arreglo que contenga los 3 arreglos
        glx_global = np.stack((galaxias['Lupton'], galaxias['Estandar'], galaxias['Estadistica']), axis=0)
        l_norms= [minmax, itqr, aprint]
        l_alphas = [10, 100, 1000]
        #comprehension list para sacar las combinaciones de los 3
        combinaciones = [(glx, norm, alpha) for glx in glx_global for norm in l_norms for alpha in l_alphas]

        #Normalizo todos los arreglos con log strech
        dict_lupton= {}
        dict_estandar= {}
        dict_estadistica= {}
        dict_lupton_norms = {}
        dict_estandar_norms = {}
        dict_estadistica_norms = {}
        for idx, c in enumerate(combinaciones):
            #obtengo el nombre de cada combinación
            tnorm = str(c[1]).split()[0].split(".")[3]
            t_log = str(c[2])
            f_key = f'{tnorm}_log_{t_log}'
            #normalizaciones
            norm = ImageNormalize(data=c[0],interval=c[1], stretch=LogStretch(c[2]))
            glx_norm = norm(c[0]).data
            if idx>=0 and idx<=8:
                dict_lupton[f_key]=glx_norm
                dict_lupton_norms[f_key]=norm
            elif idx>=9 and idx<=17:
                dict_estandar[f_key]=glx_norm
                dict_estandar_norms[f_key]=norm
            else:
                dict_estadistica[f_key]=glx_norm
                dict_estadistica_norms[f_key]=norm


        dict_arr = {'Lupton':dict_lupton, 'LuptonEstandar':dict_estandar, 'Estadistica':dict_estadistica}
        dict_norms = {'Lupton':dict_lupton_norms, 'LuptonEstandar':dict_estandar_norms, 'Estadistica':dict_estadistica_norms}

        return dict_arr, dict_norms
        
    def _norm_ganadora(self, asinh_dict, log_dict, n_log_dict, n_asinh_dict):
        '''
        Descripción del pipeline:

        1. Junto todos los diccionarios en 1 diccionario
        2. Lista vacía para guardar las métricas
        3. Los loops para calcular las métricas expuestas arriba
            1. Recorro el master_dict
            2. Entro a cada item hasta llegar al arreglo 
            3. Obtengo la Entropía de Shannon y CFD
                1. Shannon tiene base 2, por lo tanto la escala va a de 0-8
                2. Obtener en conteo de los pixeles de la galaxia
                3. Rescalo los pixeles de 0-256 para obtener escala de grises
                4. Obtengo la frecuencia y frecuenta relativa acumulada
                5. Tengo el pixel que es igual o mayor a 0.15 y el de 0.85
                6. Resto ambos pixeles para saber que tan amplia es la delta
            5. Almaceno todo en pd_metricas ordenado por CFD y Shannon de mayor a menor
        
        Return pd_metricas
        '''

        master_dict = {'AsinhStrech': asinh_dict, 'LogStrech':log_dict}
        pd_metricas = pd.DataFrame()
        for k_md, v_md in master_dict.items():
            for k_est, v_est in v_md.items():
                for k_norm, v_norm in v_est.items():
                    glx_norm = np.clip(v_norm, 0,1)
                    glx_norm = (glx_norm * 255).astype(np.uint8)
                    shannon = shannon_entropy(glx_norm, base=10)
                    conteos, _ = np.histogram(a=glx_norm, bins=256, range=(0,256))
                    frecuencias_acumuladas = np.cumsum(conteos)
                    cdf_valores = frecuencias_acumuladas / frecuencias_acumuladas[-1]
                    d15 = np.argmax(cdf_valores>=0.15)
                    d85 = np.argmax(cdf_valores>=0.85)
                    delta = d85-d15
                    saturacion = (np.sum(glx_norm >= 1.0) / len(glx_norm)) * 100

                    dict_metricas = pd.DataFrame([{
                        'estiramiento': k_md,
                        'colapso': k_est,
                        'norm': k_norm,
                        'cfd': delta,
                        'shannon': shannon,
                        'saturacion':saturacion
                    }])

                    pd_metricas = pd.concat([pd_metricas, dict_metricas], ignore_index=True)
        
        pd_metricas.sort_values(by=['cfd', 'shannon', 'saturacion'], ascending=[False, False, True], inplace=True)
        ganadora = pd_metricas.head(1)
        strech = ganadora['estiramiento'].iloc[0]
        colapso = ganadora['colapso'].iloc[0]
        norm = ganadora['norm'].iloc[0]
        glx_ganadora = master_dict[strech][colapso][norm]
        if strech=='AsinhStrech':
            norm_ganador = n_asinh_dict[colapso][norm]
        else:
            norm_ganador = n_log_dict[colapso][norm]
        return glx_ganadora, {'strech':strech, 'colapso': colapso, 'norm':norm}, norm_ganador, pd_metricas
        
    def _mascara_gaussiana(self, arreglo, bkg, sigma, headers):
        '''
        Descripción del pipeline:
        1. Obtener la segmentación gaussina para separar la galaxia y las estrellas
        2. Elaborar la máscara booleana de las estrellas de campo
        3. Generar la máscara final con la galaxia del tamaño total
        Return el boolean mask, y el seg_gaus
        '''
        ## PASO 1:
        #instancio los Kernels
        g2dK=Gaussian2DKernel(x_stddev=3, y_stddev=3)
        #convolusionamos la data
        convolve_Gauss = convolve(arreglo, g2dK)

        umbral_i = detect_threshold(convolve_Gauss, n_sigma=sigma, error=bkg)
        #segmento la imágen
        seg_gauss_i = detect_sources(data=convolve_Gauss, threshold=umbral_i, n_pixels=5, connectivity=8)
        
        ##PASO 2:
        # Quedarnos sólo con la imagen el centro
        # 1. Obetner los descriptores geométricos de cada segmentación de la imagen
        segments = regionprops(label_image=seg_gauss_i.data)
        #2. Extraer la propieadad coords para ubicar cada una de las coordenadas en un acomprehension list
        # Dinamicé el centro de la imagen, así ya sin importar el W/H simepre busca el centro por defecto
        cl_coords = [(s.label, np.sqrt((s.centroid[0]- np.round(headers['IMAGEH']/2,0))**2 +(s.centroid[1]-np.round(headers['IMAGEW']/2,0))**2 )) for s in segments] # corregir el cálculo del centroide
        #3. obtener el valor mínimo de de la comprehensión list, esto porque es el que tiene la menor distancia del centro del imagen
        id_galaxia=min(cl_coords, key= lambda dist:dist[1])
        bm_galaxia = seg_gauss_i.data == id_galaxia[0]
        bm_estrellas = (seg_gauss_i.data > 0) & (seg_gauss_i.data != id_galaxia[0])

        return bm_galaxia, seg_gauss_i, bm_estrellas
        
    def _max_luz_spline(self, glx, mask):
        '''
        Descripción del pipeline:
            1. Se aplica inverse_gaussian_gradient para poder delimitar los bordes de la galaxia sumando todos los canales
            2. Otener el arreglo del spline con morphological_geodesic_active_contour
            3. Se obtiene la máscara binanria
        RETURN spline y máscara binaria, y normalización
        '''
        
        #PASO 1
        arr_invgaus = inverse_gaussian_gradient(image=glx)
        #PASO 2
        snk_MGAC = morphological_geodesic_active_contour(
            gimage=arr_invgaus,
            num_iter=10,
            init_level_set=mask,
            threshold=0.1,
            balloon=0.2,
            smoothing=2,
        )
        #PASO 3
        mask_snk=ndimage.binary_fill_holes(snk_MGAC)
        mask_nn = mask_snk.astype(np.float32) # máscara para la neural network
        mask_cont = mask_snk.astype(np.int8) # spline para graficar

        return {
            'inverse_gausse': arr_invgaus,
            'snk_mgac': snk_MGAC,
            'mask_nn':mask_nn,
            'mask_cont':mask_cont
        }
        
    def _fits_finales(self, arr_limpios, norm, mascara, glx_final,std, glx_ganadora):
        '''
        Descripción del pipeline:
            1. Saco el estiramiento ganador de dict_ganador['strech'] para dirigirme al diccionario ganador
            2. obtengo el rgb, en caso de aplicar, para los grumos, masa y d4000
            3. Normalizo los arreglos de masa, y d4000
            4. Saco el arreglo de grumos y aplico la máscara
            5. Saco el arreglo de masa y aplico la máscara
            6. Saco el arreglo para D4000 y aplico la máscara
        RETURN arreglo_masa, arreglo_grumos, arreglo_d4000
        '''

        #paso 2
        arr_g = arr_limpios['g']['cielo']
        arr_r = arr_limpios['r']['cielo']
        arr_z = arr_limpios['z']['cielo']

        #agrego arreglo de máscara 3D y cálculo de masa
        mask_3d = np.stack((glx_final['mask_cont'], glx_final['mask_cont'], glx_final['mask_cont']),axis=2)
        gal_rgb = make_lupton_rgb(image_r=arr_z, image_g=arr_r, image_b=arr_g, stretch=0.005)
        glx_masa = glx_ganadora* mascara

        #cálculo de grumos
        _,_,gal_std = sigma_clipped_stats(data=gal_rgb, sigma=5, maxiters=10, mask=mask_3d)
        r=11
        ks= 2*r+1
        kernel = Gaussian2DKernel(x_stddev=gal_std, x_size = ks, y_size=ks)
        arr_k= kernel.array
        x,y = np.ogrid[-r:r+1, -r:r+1]
        mask_cir = (x**2 + y**2)<=r**2
        mask_cir.astype(np.int8)
        arr_k=arr_k*mask_cir
        arr_k=np.stack((arr_k,arr_k,arr_k), axis=2)
        glx_gauss = convolve(gal_rgb, arr_k, boundary='extend')
        glx_grumos = norm((gal_rgb-glx_gauss)*mask_3d).data
        

        #paso 6
        # 1. Extraemos los píxeles reales cubiertos por tu máscara 2D
        glx_d4000 = arr_g / (arr_r + 1e-8)
        pixeles_validos = glx_d4000[mascara > 0]

        # 2. ---- EL FILTRO DE OUTLIERS (Percentiles) ----
        # Encontramos los límites ignorando divisiones entre cero o ruido exterior
        limite_inferior = np.percentile(pixeles_validos, 1)
        limite_superior = np.percentile(pixeles_validos, 99)

        # 3. Escalamos proporcionalmente usando este marco robusto
        glx_d4000_normalizado = (glx_d4000 - limite_inferior) / (limite_superior - limite_inferior)

        # 4. El clip defensivo final cortará los excedentes y encenderá la galaxia
        glx_d4000_f = np.clip(glx_d4000_normalizado, 0, 1) * mascara

        
        return glx_grumos, glx_masa, glx_d4000_f

    # =========================================================================
    # --- ÚNICA FUNCIÓN PÚBLICA (ORQUESTADORA) ---
    # =========================================================================
    def fits_pytorch(self, fits_path_or_data=None, sigma_gauss=3.0, csv_output_path="metricas_norm_ganadora.csv"):
        """
        Orquesta secuencialmente todo tu pipeline de procesamiento astronómico.
        Permite recibir una ruta nueva o usar los datos cargados previamente por la clase.
        """
        # Si se pasa una ruta de archivo directamente a la función, la procesamos
        if isinstance(fits_path_or_data, (str, Path)):
            self.cargar_fits(fits_path_or_data)
            img_data = self.image_data
        elif fits_path_or_data is not None:
            # Si pasas el arreglo directamente (útil para transformaciones dinámicas en memoria)
            img_data = fits_path_or_data
        else:
            # Si no se pasó nada, usamos lo cargado en el init
            if self.image_data is None:
                raise ValueError("No se ha proporcionado ninguna ruta FITS ni datos de imagen válidos.")
            img_data = self.image_data

        # 1. Sustracción de luz parásita y métricas de fondo
        arreglos_limpios, bkg_params = self._metricas_fondo(img_data)
        
        # 2. Colapso de matrices
        dict_glx, bkg_rms, bkg_median = self._matrices_colapso(arreglos_limpios, bkg_params, img_data)
        
        # 3. Normalizaciones Asinh
        dict_arr_asinh, dict_norms_asinh = self._norm_asinh(dict_glx)
        
        # 4. Normalizaciones Log
        dict_arr_log, dict_norms_log = self._log_norm(dict_glx)
        
        # 6. Determinación de la normalización ganadora y obtención de pd_metricas
        glx_ganadora, dict_ganador, norm_ganador, pd_metricas = self._norm_ganadora(
            dict_arr_asinh, dict_arr_log, dict_norms_log, dict_norms_asinh
        )
        
        # Exportar las métricas recopiladas a un archivo CSV como se especificó
        pd_metricas.to_csv(csv_output_path, index=False)
        
        # Usamos los headers leídos reales de la galaxia si están disponibles, si no, tomamos las dimensiones del array
        if self.headers and 'IMAGEH' in self.headers and 'IMAGEW' in self.headers:
            headers_dict = self.headers
        else:
            headers_dict = {
                'IMAGEH': img_data.shape[1] if len(img_data.shape) == 3 else img_data.shape[0],
                'IMAGEW': img_data.shape[2] if len(img_data.shape) == 3 else img_data.shape[1]
            }
        
        # 7. Generación de máscara Gaussiana
        bm_galaxia, seg_gauss_i, bm_estrellas = self._mascara_gaussiana(
            glx_ganadora, bkg_rms, sigma_gauss, headers_dict
        )
        
        # 8. Segmentación mediante contornos activos (Spline)
        glx_final = self._max_luz_spline(glx_ganadora, bm_galaxia)
        
        # 9. Obtención de FITS Finales (grumos, masa, d4000)
        grumos, masa, dn4000 = self._fits_finales(
            arreglos_limpios, norm_ganador, bm_galaxia, glx_final, bkg_rms, glx_ganadora
        )
        
        # --- Formateo de los canales para PyTorch ---
        if len(grumos.shape) == 3:
            grumos_torch = np.moveaxis(grumos, source=2, destination=0)
        else:
            grumos_torch = grumos
            
        if len(masa.shape) == 3:
            masa_torch = np.moveaxis(masa, source=2, destination=0)
        else:
            masa_torch = masa
            
        if len(dn4000.shape) == 3:
            dn4000_torch = np.moveaxis(dn4000, source=2, destination=0)
        else:
            dn4000_torch = dn4000
            
        return {
            'masa': masa_torch,
            'grumos': grumos_torch,
            'dn4000': dn4000_torch,
            'mask': glx_final['mask_nn']
        }
