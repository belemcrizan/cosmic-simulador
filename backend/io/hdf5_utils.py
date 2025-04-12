import h5py
import numpy as np
from typing import Dict, Union, Any
import logging
from pathlib import Path

# Configuração básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_to_hdf5(filename: Union[str, Path], 
                data_dict: Dict[str, Union[np.ndarray, list, int, float]],
                compression: str = 'gzip',
                compression_opts: int = 4,
                overwrite: bool = True) -> None:
    """
    Salva um dicionário de dados em um arquivo HDF5 de forma robusta.
    
    Parâmetros:
    -----------
    filename : str ou Path
        Caminho para o arquivo HDF5 a ser criado
    data_dict : dict
        Dicionário contendo os dados a serem salvos (chaves são strings, valores são arrays/lista/números)
    compression : str, opcional
        Tipo de compressão a ser usado (None, 'gzip', 'lzf', 'szip')
    compression_opts : int, opcional
        Nível de compressão (para 'gzip', 0-9)
    overwrite : bool, opcional
        Se True, sobrescreve o arquivo se ele existir
        
    Retorna:
    --------
    None
    
    Exceções:
    ---------
    TypeError: Se os tipos de entrada forem inválidos
    IOError: Se houver problemas ao escrever o arquivo
    """
    try:
        # Validação de entrada
        if not isinstance(data_dict, dict):
            raise TypeError("data_dict deve ser um dicionário")
            
        filename = Path(filename)
        if not overwrite and filename.exists():
            raise FileExistsError(f"Arquivo {filename} já existe e overwrite=False")
            
        # Verifica se o diretório existe
        filename.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(filename, 'w') as f:
            for key, value in data_dict.items():
                if not isinstance(key, str):
                    raise TypeError(f"Todas as chaves devem ser strings. Recebido: {type(key)}")
                    
                # Converte listas para numpy arrays
                if isinstance(value, list):
                    value = np.array(value)
                    
                # Cria o dataset com opções de compressão
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value, 
                                   compression=compression,
                                   compression_opts=compression_opts if compression == 'gzip' else None)
                elif isinstance(value, (int, float)):
                    f.create_dataset(key, data=value)
                else:
                    raise TypeError(f"Tipo não suportado para a chave {key}: {type(value)}")
                    
            # Adiciona metadados sobre a compressão
            f.attrs['compression'] = compression
            if compression == 'gzip':
                f.attrs['compression_level'] = compression_opts
                
        logger.info(f"Dados salvos com sucesso em {filename}")
        
    except Exception as e:
        logger.error(f"Erro ao salvar em {filename}: {str(e)}")
        raise

def load_from_hdf5(filename: Union[str, Path], 
                  keys: Union[str, list, None] = None,
                  verbose: bool = False) -> Dict[str, Any]:
    """
    Carrega dados de um arquivo HDF5 de forma robusta.
    
    Parâmetros:
    -----------
    filename : str ou Path
        Caminho para o arquivo HDF5
    keys : str, list ou None, opcional
        Chaves específicas para carregar (None carrega todas)
    verbose : bool, opcional
        Se True, imprime informações sobre os datasets carregados
        
    Retorna:
    --------
    dict
        Dicionário contendo os datasets carregados
        
    Exceções:
    ---------
    FileNotFoundError: Se o arquivo não existir
    KeyError: Se alguma chave solicitada não existir no arquivo
    """
    try:
        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(f"Arquivo {filename} não encontrado")
            
        data = {}
        with h5py.File(filename, 'r') as f:
            # Determina quais chaves carregar
            load_keys = keys if keys is not None else list(f.keys())
            if isinstance(load_keys, str):
                load_keys = [load_keys]
                
            # Carrega os datasets
            for key in load_keys:
                if key not in f:
                    raise KeyError(f"Chave '{key}' não encontrada no arquivo")
                    
                data[key] = f[key][()]
                
                if verbose:
                    dataset = f[key]
                    logger.info(f"Dataset carregado: {key}")
                    logger.info(f"  Formato: {dataset.shape}")
                    logger.info(f"  Tipo: {dataset.dtype}")
                    if 'compression' in dataset.compression:
                        logger.info(f"  Compressão: {dataset.compression}")
                        
            # Carrega metadados se existirem
            if f.attrs:
                data['_metadata'] = dict(f.attrs)
                
        return data
        
    except Exception as e:
        logger.error(f"Erro ao carregar {filename}: {str(e)}")
        raise

# Funções adicionais úteis
def inspect_hdf5(filename: Union[str, Path]) -> None:
    """
    Inspeciona a estrutura de um arquivo HDF5 sem carregar todos os dados.
    
    Parâmetros:
    -----------
    filename : str ou Path
        Caminho para o arquivo HDF5
        
    Retorna:
    --------
    None
    """
    try:
        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(f"Arquivo {filename} não encontrado")
            
        with h5py.File(filename, 'r') as f:
            print(f"\nEstrutura do arquivo {filename}:")
            print(f"  Número de datasets: {len(f.keys())}")
            print(f"  Metadados globais: {dict(f.attrs)}")
            
            print("\nDetalhes dos datasets:")
            for key in f.keys():
                dataset = f[key]
                print(f"  {key}:")
                print(f"    Formato: {dataset.shape}")
                print(f"    Tipo: {dataset.dtype}")
                if dataset.compression:
                    print(f"    Compressão: {dataset.compression}")
                    
    except Exception as e:
        logger.error(f"Erro ao inspecionar {filename}: {str(e)}")
        raise