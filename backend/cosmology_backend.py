import numpy as np
from classy import Class
from pyExSHalos import generate_field
from pyExSHalos.power_spectrum import calc_pk
from typing import Dict, Tuple, Optional, Union
import logging
from scipy.interpolate import interp1d

class CosmologyCalculator:
    def __init__(self, default_params: Optional[Dict] = None):
        """
        Inicializa a calculadora cosmológica com parâmetros padrão ou personalizados.
        
        Parâmetros:
        -----------
        default_params : dict, opcional
            Dicionário de parâmetros cosmológicos. Se None, usa valores padrão do Planck.
        """
        self.default_params = default_params or {
            'output': 'mPk',
            'A_s': 2.1e-9,
            'n_s': 0.96,
            'h': 0.67,
            'omega_b': 0.022,
            'omega_cdm': 0.12,
            'tau_reio': 0.06,
            'P_k_max_h/Mpc': 20.0,
            'non linear': 'halofit',
        }
        
        # Configuração de logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('CosmologyCalculator')
        
        # Cache para resultados computados
        self._pk_cache = None
        self._cosmo_instance = None

    def _validate_cosmology_params(self, params: Dict) -> None:
        """Valida os parâmetros cosmológicos de entrada."""
        required_params = ['h', 'omega_b', 'omega_cdm', 'A_s', 'n_s']
        for param in required_params:
            if param not in params:
                raise ValueError(f"Parâmetro cosmológico necessário ausente: {param}")
                
        if params.get('P_k_max_h/Mpc', 0) <= 0:
            raise ValueError("P_k_max_h/Mpc deve ser positivo")

    def compute_power_spectrum(self, 
                             params: Optional[Dict] = None,
                             k_min: float = 1e-4,
                             k_max: float = 10.0,
                             n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computa o espectro de potência linear usando CLASS.
        
        Parâmetros:
        -----------
        params : dict, opcional
            Parâmetros cosmológicos. Se None, usa os parâmetros padrão.
        k_min : float
            Valor mínimo de k (h/Mpc)
        k_max : float
            Valor máximo de k (h/Mpc)
        n_points : int
            Número de pontos no espaço logarítmico de k
            
        Retorna:
        --------
        Tuple[np.ndarray, np.ndarray]
            Arrays de k (h/Mpc) e P(k) (Mpc/h)^3
        """
        try:
            params = params or self.default_params
            self._validate_cosmology_params(params)
            
            # Usa instância existente ou cria nova
            if self._cosmo_instance is None:
                self._cosmo_instance = Class()
                self._cosmo_instance.set(params)
                self._cosmo_instance.compute()
            
            # Gera array logarítmico de k
            k = np.logspace(np.log10(k_min), np.log10(k_max), n_points)
            
            # Calcula P(k) em z=0
            Pk = np.array([self._cosmo_instance.pk(ki, 0.0) for ki in k])
            
            # Cache dos resultados
            self._pk_cache = (k, Pk)
            
            self.logger.info(f"Espectro de potência computado para k=[{k_min:.2e}, {k_max:.2e}] h/Mpc")
            return k, Pk
            
        except Exception as e:
            self.logger.error(f"Erro ao computar espectro de potência: {str(e)}")
            # Limpa a instância em caso de erro
            if self._cosmo_instance is not None:
                self._cosmo_instance.struct_cleanup()
                self._cosmo_instance = None
            raise

    def generate_density_field(self, 
                             boxsize: float = 1000.0,  # Mpc/h
                             ngrid: int = 128,
                             Pk: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                             seed: Optional[int] = None,
                             interp_kwargs: Optional[Dict] = None) -> np.ndarray:
        """
        Gera um campo de densidade gaussiano usando pyExSHalos.
        
        Parâmetros:
        -----------
        boxsize : float
            Tamanho da caixa em Mpc/h
        ngrid : int
            Número de células por dimensão
        Pk : tuple, opcional
            Tupla (k, Pk) para o espectro de potência. Se None, computa automaticamente.
        seed : int, opcional
            Semente para o gerador aleatório
        interp_kwargs : dict, opcional
            Argumentos para a interpolação do espectro de potência
            
        Retorna:
        --------
        np.ndarray
            Campo de densidade 3D (δ = ρ/ρ̄ - 1)
        """
        try:
            # Obtém espectro de potência se não fornecido
            if Pk is None:
                if self._pk_cache is None:
                    self.compute_power_spectrum()
                k, Pk = self._pk_cache
            
            # Configura interpolação
            interp_kwargs = interp_kwargs or {'kind': 'cubic', 'fill_value': 'extrapolate'}
            Pk_interp = interp1d(k, Pk, **interp_kwargs)
            
            # Gera o campo
            delta = generate_field(
                boxsize=boxsize,
                ngrid=ngrid,
                Pk=Pk_interp,
                seed=seed,
                verbose=True
            )
            
            self.logger.info(f"Campo de densidade gerado: boxsize={boxsize} Mpc/h, ngrid={ngrid}^3")
            return delta
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar campo de densidade: {str(e)}")
            raise

    def measure_power_spectrum(self, 
                             delta: np.ndarray, 
                             boxsize: float,
                             k_bins: Optional[Union[int, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mede o espectro de potência de um campo de densidade.
        
        Parâmetros:
        -----------
        delta : np.ndarray
            Campo de densidade 3D
        boxsize : float
            Tamanho da caixa em Mpc/h
        k_bins : int ou np.ndarray, opcional
            Número de bins ou bordas dos bins para o espectro de potência
            
        Retorna:
        --------
        Tuple[np.ndarray, np.ndarray]
            k (h/Mpc) e P(k) (Mpc/h)^3 medidos
        """
        try:
            if not isinstance(delta, np.ndarray) or delta.ndim != 3:
                raise ValueError("delta deve ser um array numpy 3D")
                
            if boxsize <= 0:
                raise ValueError("boxsize deve ser positivo")
                
            # Mede P(k)
            k, Pk = calc_pk(delta, boxsize, bins=k_bins)
            
            self.logger.info(f"Espectro de potência medido para campo {delta.shape}")
            return k, Pk
            
        except Exception as e:
            self.logger.error(f"Erro ao medir espectro de potência: {str(e)}")
            raise

    def __del__(self):
        """Limpeza adequada ao destruir a instância"""
        if self._cosmo_instance is not None:
            self._cosmo_instance.struct_cleanup()
            self._cosmo_instance = None