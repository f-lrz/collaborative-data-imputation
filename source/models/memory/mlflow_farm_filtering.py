from mlflow.pyfunc import PythonModel
import numpy as np
from sortedcontainers import SortedList
from loguru import logger

class FarmCollaborativeFiltering(PythonModel):
    " Collaborative Filtering model based on wind farms similarity. "

    def __init__(self, farm2period, periodfarm2power, lst_farms, K, min_common_periods):
        " Initialize the FarmCollaborativeFiltering instance. "
        # Check if the inputs are of the correct type
        assert isinstance(farm2period, dict), "Input farm2period must be a dictionary."
        assert isinstance(periodfarm2power, dict), "Input periodfarm2power must be a dictionary."
        assert isinstance(lst_farms, list), "Input lst_farms must be a list."
        assert isinstance(K, int), "Input K must be an integer."
        assert isinstance(min_common_periods, int), "Input min_common_periods must be an integer."
        assert K > 0, 'K must be a positive integer'
        assert min_common_periods > 0, 'min_common_farms must be a positive integer'

        # Store the inputs
        self.farm2period = farm2period
        self.periodfarm2power = periodfarm2power
        self.lst_farms = lst_farms
        self.K = K
        self.min_common_periods = min_common_periods
        self.neighbors = {}
        self.averages = {}
        self.deviations = {}
        self.sigmas = {}

    def calculate_avg_and_deviation(self, farm, periods):
        " Calculate the average, deviation, and sigma of power for a given farm. "
        powers = {period: self.periodfarm2power[(period, farm)] for period in periods}
        # Compute the average power
        avg_power = np.mean(list(powers.values()))
        # Compute the deviation of power
        dev_power = {period: (power - avg_power) for period, power in powers.items()}
        dev_power_values = np.array(list(dev_power.values()))
        # Compute the sigma of power
        sigma_power = np.sqrt(dev_power_values.dot(dev_power_values))
        return avg_power, dev_power, sigma_power

    def pearson_similarity(self, common_periods, dev_power_i, dev_power_j):
        " Compute Pearson similarity using only common periods. "
        # Extrai apenas os desvios dos períodos em comum
        dev_i_common = np.array([dev_power_i[p] for p in common_periods])
        dev_j_common = np.array([dev_power_j[p] for p in common_periods])
        
        covariance = np.sum(dev_i_common * dev_j_common)
        
        # Calcula os sigmas locais estritamente sobre os períodos em comum
        sigma_i_local = np.sqrt(np.sum(dev_i_common**2))
        sigma_j_local = np.sqrt(np.sum(dev_j_common**2))
        
        if sigma_i_local == 0 or sigma_j_local == 0:
            return 0.0
            
        weigth_ij = covariance / (sigma_i_local * sigma_j_local)
        return weigth_ij

    def compute_similarities(self):
        " Compute the similarities between wind farms. "
        logger.info('Start Data Imputation based on Wind Farms Similarity')
        
        # 1. PRÉ-CÁLCULO: Calcula as médias e desvios de todas as variáveis de uma vez só 
        for farm in self.lst_farms:
            periods = self.farm2period[farm]
            avg_power, dev_power, sigma_power = self.calculate_avg_and_deviation(farm, periods)
            self.averages[farm] = avg_power
            self.deviations[farm] = dev_power
            self.sigmas[farm] = sigma_power

        # Iterate over the wind farms
        count=0
        for farm_i in self.lst_farms:
            periods_i_set = set(self.farm2period[farm_i])
            dev_power_i = self.deviations[farm_i]
            
            sl = SortedList()
            for farm_j in self.lst_farms:
                if farm_j != farm_i:
                    periods_j_set = set(self.farm2period[farm_j])
                    common_periods = periods_i_set & periods_j_set 
                    
                    if len(common_periods) >= self.min_common_periods:  
                        dev_power_j = self.deviations[farm_j]
                        
                        # Passa apenas os desvios. O cálculo do sigma é feito lá dentro usando apenas as datas comuns.
                        w_ij = self.pearson_similarity(common_periods, dev_power_i, dev_power_j)
                        
                        sl.add((-w_ij, farm_j))
                        if len(sl) > self.K:
                            del sl[-1]
            self.neighbors[farm_i] = sl
            if count % 5 == 0:
                logger.info('Wind Farms Processed: ' + str(count))
            count += 1
            
        return self.neighbors, self.averages, self.deviations, self.sigmas

    def compute_predictions(self, farm_i, period_m):
            " Compute the prediction for a given farm and period. "
            # Iterate over the neighbors of farm_i
            # Compute the numerator and denominator for the prediction
            numerator = 0
            denominator = 0
            for neg_w, farm_j in self.neighbors[farm_i]:
                try:
                    # Check if farm_j has a power value for period_m
                    # If not, skip the farm
                    numerator += -neg_w * self.deviations[farm_j][period_m]
                    denominator += abs(neg_w)
                except KeyError:
                    pass
            # Compute the prediction
            if denominator == 0:
                # If denominator is 0, use the average power of farm_i
                prediction = self.averages[farm_i]
            else:
                # Compute the prediction
                prediction = self.averages[farm_i] + numerator / denominator
            prediction = min(1, prediction)  # max power is 1.0 in normalized data
            prediction = max(0, prediction)  # min power is 0.0
            return prediction

    def predict(self, power_dict):
        " Predict using the model. "
        assert isinstance(power_dict, dict), "Input power_dict must be a dictionary."
        predictions = []
        targets = []
        periods = []
        farms = []
        for (period_m, farm_i), target_power in power_dict.items():
            if farm_i in self.lst_farms:
                prediction = self.compute_predictions(farm_i, period_m)
                predictions.append(prediction)
                targets.append(target_power)
                periods.append(period_m)
                farms.append(farm_i)
        return predictions, targets, periods, farms

    @staticmethod
    def compute_rmse(predictions, targets):
        " Compute the root mean squared error. "
        assert isinstance(predictions, list), "Input predictions must be a list."
        assert isinstance(targets, list), "Input targets must be a list."
        assert len(predictions) == len(targets), "Length of predictions and targets must be the same."
        predictions = np.array(predictions)
        targets = np.array(targets)
        rmse = np.sqrt(np.mean((predictions - targets)**2))
        return rmse
