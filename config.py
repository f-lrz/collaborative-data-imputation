
from pathlib import Path

# Get the absolute path of the current script
current_path = Path(__file__).resolve().parent

params = {'uri': 'http://127.0.0.1:5000',
            'exp_name': 'Imputation Bahia',
            'path': current_path,
            'model': 'period-collaborative', #'lag-farm-collaborative', # 'latent-factor', #'period-collaborative', 'grouped-average', 'farm-collaborative
            'seed': 42,
            'nord_pool': 'data/dataset_multivariado_bahia.csv',
            'test_size' : 0.3,
            'block':False,
            'blocksize': 168,
            #####################
            'group_by': 'periodId',   # grouped-average
            #####################
            'based_on': 'others-lags-farms',  # period/farm/lag-farm collaborative filtering
            'neighbors': 50,
            'min_common_farms': 2,
            'min_common_periods': 50,
            'nr_lags': 4,
            'lookup': 'both',
            'other_farms': True,
            #####################
            'latent_dimensions': 2,  # latent factor
            'n_epochs': 50,
            'warm_start': True,
            'lambda_reg_U': 5.0,
            'lambda_reg_P': 5.0,
            'learning_rate': None,
            'solver': 'als'
            }
