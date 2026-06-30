import numpy as np
from loguru import logger
from sortedcontainers import SortedList
from mlflow.pyfunc import PythonModel

class PeriodCollaborativeFiltering(PythonModel):
    """
    Class for computing period similarities using collaborative filtering.
    """

    def __init__(self, period2farm, periodfarm2power, lst_periods, K, min_common_farms):
        """
        Initialize the PeriodCollaborativeFiltering instance.
        """
        assert isinstance(period2farm, dict), 'period2farm must be a dictionary'
        assert isinstance(periodfarm2power, dict), 'periodfarm2power must be a dictionary'
        assert isinstance(lst_periods, list), 'lst_periods must be a list'
        assert isinstance(K, int), 'K must be an integer'
        assert isinstance(min_common_farms, int), 'min_common_farms must be an integer'
        assert K > 0, 'K must be a positive integer'
        assert min_common_farms > 0, 'min_common_farms must be a positive integer'

        # Store the inputs
        self.period2farm = period2farm
        self.periodfarm2power = periodfarm2power
        self.lst_periods = lst_periods
        self.K = K
        self.min_common_farms = min_common_farms
        self.neighbors = {}
        self.averages = {}
        self.deviations = {}
        self.sigmas = {}

    def calculate_avg_and_deviation(self, period, farms):
        """
        Calculate average power and deviation for a given period and its corresponding farms.
        """
        powers = {farm: self.periodfarm2power[(period, farm)] for farm in farms}  # power values for the period
        avg_power = np.mean(list(powers.values()))  # average power for the period
        dev_power = {farm: (power - avg_power) for farm, power in powers.items()}  # deviation from the average power
        dev_power_values = np.array(list(dev_power.values()))  # deviation values
        sigma_power = np.sqrt(dev_power_values.dot(dev_power_values))  # standard deviation
        return avg_power, dev_power, sigma_power

    def pearson_similarity(self, common_farms, dev_power_i, dev_power_j):
        " Compute Pearson similarity using only common farms/variables. "
        # Extrai apenas os desvios das variáveis (farms) em comum naquele instante de tempo
        dev_i_common = np.array([dev_power_i[farm] for farm in common_farms])
        dev_j_common = np.array([dev_power_j[farm] for farm in common_farms])
        
        covariance = np.sum(dev_i_common * dev_j_common)
        
        # Calcula o sigma local estritamente sobre as variáveis em comum
        sigma_i_local = np.sqrt(np.sum(dev_i_common**2))
        sigma_j_local = np.sqrt(np.sum(dev_j_common**2))
        
        if sigma_i_local == 0 or sigma_j_local == 0:
            return 0.0
            
        weigth_ij = covariance / (sigma_i_local * sigma_j_local)
        return weigth_ij

    def compute_period_similarities(self):
        """
        Compute period similarities based on collaborative filtering.
        """
        logger.info('Start Data Imputation based on Period Similarity')
        
        # 1. PRÉ-CÁLCULO: Calcula médias e desvios de TODOS os períodos de uma vez
        for period in self.lst_periods:
            farms = self.period2farm[period]
            avg_power, dev_power, sigma_power = self.calculate_avg_and_deviation(period, farms)
            self.averages[period] = avg_power
            self.deviations[period] = dev_power
            self.sigmas[period] = sigma_power # Opcional agora
            
        # Iterate over the periods
        count = 0
        for period_i in self.lst_periods:
            farms_i_set = set(self.period2farm[period_i])
            dev_power_i = self.deviations[period_i]
            
            sl = SortedList()
            for period_j in self.lst_periods:
                if period_j != period_i:
                    farms_j_set = set(self.period2farm[period_j])
                    common_farms = farms_i_set & farms_j_set 
                    
                    if len(common_farms) >= self.min_common_farms:
                        dev_power_j = self.deviations[period_j]
                        
                        # Usa a nova função de similaridade corrigida
                        w_ij = self.pearson_similarity(common_farms, dev_power_i, dev_power_j)
                        sl.add((-w_ij, period_j))
                        
                        if len(sl) > self.K:
                            del sl[-1]
                            
            self.neighbors[period_i] = sl
            
            if count % 1000 == 0:  # Diminuí o print para dar um feedback mais rápido
                logger.info(f'Total processed periods: {count}')
            count += 1
            
        logger.info(f'Total processed periods: {count}')
        return self.neighbors, self.averages, self.deviations, self.sigmas

    def compute_predictions(self, period_i, farm_m):
            """
            Compute power prediction for a given period and farm using collaborative filtering.
            """
            numerator = 0
            denominator = 0
            for neg_w, period_j in self.neighbors[period_i]:
                try:
                    numerator += -neg_w * self.deviations[period_j][farm_m]
                    denominator += abs(neg_w)
                except KeyError:
                    pass
            if denominator == 0:
                prediction = self.averages[period_i]
            else:
                prediction = self.averages[period_i] + numerator / denominator
            prediction = min(1, prediction)  # max power is 1.0 in normalized data
            prediction = max(0, prediction)  # min power is 0.0
            return prediction


    def predict(self, power_dict):
        """
        Make power predictions for a dictionary of period-farm pairs.
        """
        assert isinstance(power_dict, dict), "Input power_dict must be a dictionary."
        predictions = []
        targets = []
        periods = []
        farms = []
        
        # Cache para armazenar a média histórica de treino por estação e evitar lentidão
        fallback_averages = {}
        
        for (period_i, farm_m), target_power in power_dict.items():
            # Se o período for inédito/futuro (out-of-sample), ele não terá vizinhos.
            # Capturamos isso e aplicamos a média histórica de treino da estação correspondente.
            if period_i not in self.neighbors:
                if farm_m not in fallback_averages:
                    vals = [v for (p, f), v in self.periodfarm2power.items() if f == farm_m]
                    fallback_averages[farm_m] = np.mean(vals) if vals else 0.0
                prediction = fallback_averages[farm_m]
            else:
                # Se o período já existe no histórico, segue o fluxo normal
                prediction = self.compute_predictions(period_i, farm_m)
                
            predictions.append(prediction)
            targets.append(target_power)
            periods.append(period_i)
            farms.append(farm_m)
            
        return predictions, targets, periods, farms


    @staticmethod
    def compute_rmse(predictions, targets):
        " Compute the root mean squared error (RMSE) between predictions and targets. "
        assert isinstance(predictions, list), "Input predictions must be a list."
        assert isinstance(targets, list), "Input targets must be a list."
        assert len(predictions) == len(targets), "Length of predictions and targets must be the same."
        predictions = np.array(predictions)
        targets = np.array(targets)
        rmse = np.sqrt(np.mean((predictions - targets)**2))
        return rmse
