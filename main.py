import pandas as pd
import numpy as np
import mlflow
from loguru import logger
from copy import deepcopy
from config import params
from source.process.process_general import Normalizer, melt_dataframe, preprocess_ids
from experiments.experiment_latent_factor import run_latent_factor_experiment
from experiments.experiment_group_average import run_grouped_avg_experiment
from experiments.experiment_period_filtering import run_period_collaborative_experiment
from experiments.experiment_farm_filtering import run_farm_collaborative_experiment
from experiments.experiment_lag_farm_filtering import run_lag_farm_collaborative_experiment

if __name__ == "__main__":
    mlflow.set_tracking_uri(uri=params['uri'])
    mlflow.set_experiment(params['exp_name'])

    # 1. Carregar a nova base multivariada
    df = pd.read_csv(params['nord_pool'])
    df['periodId'] = pd.to_datetime(df['periodId'])
    df.set_index('periodId', inplace=True)
    
    # =====================================================================
    # LINHA PARA TESTE RÁPIDO
    df = df.loc['2019-01-01':'2019-12-31'] 
    # =====================================================================

    lista_estacoes = ['SALVADOR', 'LEM', 'CARAVELAS', 'ITABERABA']

    for estacao in lista_estacoes:
        logger.info("\n" + "="*60 + f"\nINICIANDO TREINO MULTIVARIADO PARA: {estacao}\n" + "="*60)

        # Filtra as 3 variáveis da cidade atual (ex: SALVADOR, SALVADOR_pressao, SALVADOR_umidade)
        colunas_estacao = [c for c in df.columns if c.startswith(estacao)]
        df_estacao = df[colunas_estacao].copy()

        if df_estacao.empty:
            continue

        # =========================================================================
        # ESTRATÉGIA DE MASCARAMENTO OUT-OF-SAMPLE
        # =========================================================================
        # Descobrimos o ponto exato de corte (últimos 30% do tempo)
        tamanho_teste = int(len(df_estacao) * params['test_size'])
        indice_corte = len(df_estacao) - tamanho_teste
        
        # 1. Salvar o Gabarito (Apenas a coluna do vento real no futuro para validar o RMSE)
        df_teste_gabarito = df_estacao.iloc[indice_corte:][[estacao]].copy()
        
        # 2. Aplicar a Máscara (Apagar as respostas de vento do futuro no DataFrame principal)
        # Assim o modelo verá a Pressão e Umidade do futuro para poder fazer Pearson, mas não o Vento.
        col_idx_vento = df_estacao.columns.get_loc(estacao)
        df_estacao.iloc[indice_corte:, col_idx_vento] = np.nan
        
        # Melt do Contexto (Linha do tempo inteira, mas com vento futuro em NaN)
        training_df_datetime, _ = melt_dataframe(df_estacao, id_vars_='periodId', var_name_='farmId', value_name_='power')
        
        # Melt do Gabarito (Para entregar à função de erro do MLflow)
        validation_df_datetime, _ = melt_dataframe(df_teste_gabarito, id_vars_='periodId', var_name_='farmId', value_name_='power')

        # --- Normalização e Mapeamento Padrão ---
        normalizer = Normalizer()
        norm_training_df_datetime, norm_validation_df_datetime, _, _ = normalizer.normalize_power(training_df_datetime, validation_df_datetime, id_col='farmId', power_col='power')

        training_df = deepcopy(norm_training_df_datetime)
        validation_df = deepcopy(norm_validation_df_datetime)

        training_df, validation_df, id2datetime_mapping, id2farm_mapping = preprocess_ids(training_df, validation_df)

        local_params = deepcopy(params)
        local_params['based_on'] = f"{params['based_on']}_{estacao}_MULTI"

        logger.info('-------------------- Iniciando MLflow --------------------')
        if local_params['model'] == 'period-collaborative':
            run_period_collaborative_experiment(training_df, validation_df, **local_params)
            logger.success(f"Experimento concluído para {estacao}.")
        elif local_params['model'] == 'farm-collaborative':
            run_farm_collaborative_experiment(training_df, validation_df, **local_params)
            logger.success(f"Experimento concluído para {estacao}.")
        elif local_params['model'] == 'lag-farm-collaborative':
            run_lag_farm_collaborative_experiment(training_df, validation_df, **local_params)
            logger.success(f"Experimento concluído para {estacao}.")
        elif local_params['model'] == 'latent-factor':
            run_latent_factor_experiment(training_df, validation_df, **local_params)
            logger.success(f"Experimento concluído para {estacao}.")