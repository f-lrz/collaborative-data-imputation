import numpy as np
from sortedcontainers import SortedList
from loguru import logger
from mlflow.pyfunc import PythonModel
from loguru import logger

class LagFarmDataImputation(PythonModel):
    " Define a class for Lag-Farm Data Imputation. "

    def __init__(self, farm2period, periodfarm2power, lst_farms, lst_farms_lags, K, min_common_periods, other_farms=True):
        " Initialize the LagFarmDataImputation object. "

        assert isinstance(farm2period, dict), "Input farm2period must be a dictionary."
        assert isinstance(periodfarm2power, dict), "Input periodfarm2power must be a dictionary."
        assert isinstance(lst_farms, list), "Input lst_farms must be a list."
        assert isinstance(lst_farms_lags, list), "Input lst_farms_lags must be a list."
        assert isinstance(K, int), "Input K must be an integer."
        assert isinstance(min_common_periods, int), "Input min_common_periods must be an integer."
        assert isinstance(other_farms, bool), "Input other_farms must be a boolean."

        # Set parameters
        self.farm2period = farm2period  # Dictionary mapping farms to periods
        self.periodfarm2power = periodfarm2power  # Dictionary mapping periods and farms to power values
        self.lst_farms = lst_farms  # List of wind farms
        self.lst_farms_lags = lst_farms_lags  # List of wind farms with lags
        self.K = K  # Number of neighbors
        self.min_common_periods = min_common_periods  # Minimum number of common periods
        self.other_farms = other_farms  # Include other farms
        self.neighbors = {}
        self.averages = {}
        self.deviations = {}
        self.sigmas = {}

    def calculate_avg_and_deviation(self, farm, periods):
        " Calculate average power and deviation for a given farm and periods. "
        # Get the power values for the farm for all periods
        powers = {period: self.periodfarm2power[(period, farm)] for period in periods}
        # Compute the average power
        avg_power = np.mean(list(powers.values()))
        # Compute the deviation of power
        dev_power = {period: (power - avg_power) for period, power in powers.items()}
        # Compute the deviation values
        dev_power_values = np.array(list(dev_power.values()))
        # Compute the sigma of power
        sigma_power = np.sqrt(dev_power_values.dot(dev_power_values))
        return avg_power, dev_power, sigma_power

    def pearson_similarity(self, common_periods, dev_power_i, sigma_power_i,  dev_power_j, sigma_power_j):
        " Compute Pearson similarity between two wind farms. "
        covariance = sum(dev_power_i[period] * dev_power_j[period] for period in common_periods)  # covariance
        weigth_ij = covariance / (sigma_power_i * sigma_power_j)  # Pearson similarity
        return weigth_ij

    def compute_similarities(self):
        " Compute the similarities between wind farms (lagged and non-lagged). "
        count=0
        logger.info('Start Data Imputation based on Wind Farms Similarity')
        logger.info('List of Wind Farms ' + str(self.lst_farms))
        
        # PRÉ-CALCULO PARA TODAS AS VARIÁVEIS (BASE + LAGS)
        for farm in self.lst_farms_lags:
            if farm in self.farm2period:
                periods = self.farm2period[farm]
                avg_p, dev_p, sigma_p = self.calculate_avg_and_deviation(farm, periods)
                self.averages[farm] = avg_p
                self.deviations[farm] = dev_p
                self.sigmas[farm] = sigma_p
        
        # Iterate over the wind farms
        for farm_i in self.lst_farms:
            periods_i = self.farm2period[farm_i]  # Get the periods of farm_i
            periods_i_set = set(periods_i)  # Convert periods_i to a set
            dev_power_i = self.deviations[farm_i]
            sigma_power_i = self.sigmas[farm_i]
            # Create a SortedList to store the similarities
            sl = SortedList()
            if not self.other_farms:  # Include only lags of the same wind farm
                lst_farms_j = [name for name in self.lst_farms_lags if str(name).split('_')[0] == str(farm_i).split('_')[0]]
            else:
                lst_farms_j = self.lst_farms_lags
            #lst_farms_j = [name for name in self.lst_farms_lags if self.other_farms or name.split('_')[0] == farm_i.split('_')[0]]
            for farm_j in lst_farms_j:
                if farm_j != farm_i:
                    periods_j = self.farm2period[farm_j]
                    periods_j_set = set(periods_j)
                    common_periods = periods_i_set & periods_j_set  # Find the common periods
                    if len(common_periods) >= self.min_common_periods:  # Check if the number of common periods is greater than min_common_periods
                        # Busca os valores já pré-calculados
                        dev_power_j = self.deviations[farm_j]
                        sigma_power_j = self.sigmas[farm_j]
                        w_ij = self.pearson_similarity(common_periods, dev_power_i, sigma_power_i, dev_power_j, sigma_power_j)
                        # Add the similarity to the SortedList
                        sl.add((-w_ij, farm_j))
                        # Keep only the K most similar wind farms
                        if len(sl) > self.K:
                            del sl[-1]
            self.neighbors[farm_i] = sl
            if count % 5 == 0:
                logger.info('Wind Farms Processed: ' + str(count))
            count+=1
        return self.neighbors, self.averages, self.deviations, self.sigmas

    def compute_predictions(self, farm_i, period_m):
        " Compute the predictions for a given farm and period. "
        numerator = 0
        denominator = 0
        for neg_w, farm_j in self.neighbors[farm_i]:
            try:
                numerator += -neg_w * self.deviations[farm_j][period_m]
                denominator += abs(neg_w)
            except KeyError:
                pass
        if denominator == 0:
            prediction = self.averages[farm_i]
        else:
            prediction = self.averages[farm_i] + numerator / denominator
        prediction = min(1, prediction)  # max power is 1.0 in normalized data
        prediction = max(0, prediction)  # min power is 0.0
        return prediction

    def predict(self, power_dict):
        " Make predictions for a given dictionary of power values. "
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
        " Compute the Root Mean Squared Error (RMSE) between predictions and targets. "
        assert isinstance(predictions, list), "Input predictions must be a list."
        assert isinstance(targets, list), "Input targets must be a list."
        assert len(predictions) == len(targets), "Length of predictions and targets must be the same."
        predictions = np.array(predictions)
        targets = np.array(targets)
        rmse = np.sqrt(np.mean((predictions - targets)**2))
        return rmse
