import pandas as pd
import mlflow
from loguru import logger
from copy import deepcopy
from config import params
from source.utils.utils_general import read_nordpool_csv
from source.process.process_general import Normalizer
from source.process.process_general import melt_dataframe, split_train_test,  preprocess_ids, filter_data_by_common_periods_farms
from experiments.experiment_latent_factor import run_latent_factor_experiment
from experiments.experiment_group_average import run_grouped_avg_experiment
from experiments.experiment_period_filtering import run_period_collaborative_experiment
from experiments.experiment_farm_filtering import run_farm_collaborative_experiment
from experiments.experiment_lag_farm_filtering import run_lag_farm_collaborative_experiment

if __name__ == "__main__":

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri=params['uri'])
    mlflow.set_experiment(params['exp_name'])

    # uri and experiment name for logging
    uri = str(params['uri'])
    exp_name = params['exp_name']

    logger.info(f'Tracking server {uri}')
    logger.info(f'Experiment name {exp_name}')

    # read nord pool csv file
    df = pd.read_csv(params['nord_pool'])
    df['periodId'] = pd.to_datetime(df['periodId'])
    df.set_index('periodId', inplace=True)
    logger.info(f"NORDPOOL dataframe head:\n{df.head()}")
    logger.success("Data loaded.")


    lista_estacoes = ['SALVADOR', 'LEM', 'CARAVELAS', 'ITABERABA']

    for estacao in lista_estacoes:
        logger.info("\n" + "="*60 + f"\nINICIANDO TREINO E TESTE ISOLADO PARA: {estacao}\n" + "="*60)

        #  melt dataframe for matrix factorization and retain non-missing values
        df_melt_without_nan, _ = melt_dataframe(df, id_vars_='periodId', var_name_='farmId', value_name_='power')
        logger.info(f"Melted dataframe head:\n{df_melt_without_nan.head(3)}")
        logger.success("Melt Dataframe created.")

        df_estacao_isolada = df_melt_without_nan[df_melt_without_nan['farmId'] == estacao].reset_index(drop=True)

        if df_estacao_isolada.empty:
            logger.warning(f"Estação {estacao} não encontrada no banco de dados. Pulando...")
            continue

        # split to training and validation sets (Passando agora apenas o df_estacao_isolada)
        training_df_datetime, validation_df_datetime = split_train_test(df_estacao_isolada,
                                                                        test_size = params['test_size'],
                                                                        block = params['block'],
                                                                        blocksize = params['blocksize'],
                                                                        seed = params['seed'])
        logger.info(f"Training dataframe head:\n{training_df_datetime.head(3)}")
        logger.success("Train/Test data splitted.")

        # match training/validation periods, farms
        # filtered_training_df, filtered_validation_df = filter_data_by_common_periods_farms(training_df_datetime, validation_df_datetime)
        
        filtered_training_df = training_df_datetime
        filtered_validation_df = validation_df_datetime
        
        logger.info(f"Filtered Training dataframe head:\n{filtered_training_df.head(3)}")
        logger.success("Train/Test data keys matched.")

        # normalize data
        normalizer = Normalizer()
        norm_training_df_datetime,  norm_validation_df_datetime, _, _ = normalizer.normalize_power(filtered_training_df, filtered_validation_df, id_col='farmId', power_col='power')
        logger.success("Data normalized by max.")

        # copy datasets for grouped avg model
        training_df = deepcopy(norm_training_df_datetime)
        validation_df = deepcopy(norm_validation_df_datetime)

        # retain and adjust ids for factorization
        training_df, validation_df, id2datetime_mapping, id2farm_mapping = preprocess_ids(training_df, validation_df)
        logger.success("Ids Processed.")

        logger.info('-------------------- MLflow experiment --------------------')

        local_params = deepcopy(params)
        local_params['based_on'] = f"{params['based_on']}_{estacao}"

        if local_params['model'] == 'grouped-average':
            # run grouped-avg model experiment
            logger.info('-------------------- ' + local_params['model'] + ' --------------------')
            run_grouped_avg_experiment(training_df, validation_df, **local_params)
            logger.success(f"Experiment for {estacao} successfully concluded.")

        elif local_params['model'] == 'period-collaborative':
            # run period collaborative model experiment
            logger.info('-------------------- ' + local_params['model'] + ' --------------------')
            run_period_collaborative_experiment(training_df, validation_df, **local_params)
            logger.success(f"Experiment for {estacao} successfully concluded.")

        elif local_params['model'] == 'farm-collaborative':
            # run period collaborative model experiment
            logger.info('-------------------- ' + local_params['model'] + ' --------------------')
            run_farm_collaborative_experiment(training_df, validation_df, **local_params)
            logger.success(f"Experiment for {estacao} successfully concluded.")

        elif local_params['model'] == 'lag-farm-collaborative':
            # run period collaborative model experiment
            logger.info('-------------------- ' + local_params['model'] + ' --------------------')
            run_lag_farm_collaborative_experiment(training_df, validation_df, **local_params)
            logger.success(f"Experiment for {estacao} successfully concluded.")

        elif local_params['model'] == 'latent-factor':
            # run latent factor model experiment
            logger.info('-------------------- ' + local_params['model'] + ' --------------------')
            logger.info('-------------------- ' + local_params['solver'] + ' --------------------')
            run_latent_factor_experiment(training_df, validation_df, **local_params)
            logger.success(f"Experiment for {estacao} successfully concluded.")
        else:
            logger.error('--------------------  NO MODEL SET ! -------------------- ')