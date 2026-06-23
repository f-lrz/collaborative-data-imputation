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

    def pearson_similarity(self, common_periods, dev_power_i, sigma_power_i,  dev_power_j, sigma_power_j):
        " Compute Pearson similarity between two wind farms. "
        covariance = sum(dev_power_i[period] * dev_power_j[period] for period in common_periods)
        weigth_ij = covariance / (sigma_power_i * sigma_power_j)
        return weigth_ij

    def compute_period_similarities(self):
        """
        Compute period similarities based on collaborative filtering.
        """
        # Iterate over the periods
        count=0
        for period_i in self.lst_periods:
            farms_i = self.period2farm[period_i]  # Get the farms for period_i
            farms_i_set = set(farms_i)  # Convert farms_i to a set
            avg_power_i, dev_power_i, sigma_power_i = self.calculate_avg_and_deviation(period_i, farms_i)
            self.averages[period_i] = avg_power_i  # Store the average power of period_i
            self.deviations[period_i] = dev_power_i  # Store the deviation of power of period
            self.sigmas[period_i] = sigma_power_i  # Store the sigma of power of period_i
            # Create a SortedList to store the similarities
            sl = SortedList()
            for period_j in self.lst_periods:
                if period_j != period_i:  # Skip the same period
                    farms_j = self.period2farm[period_j]  # Get the farms for period_j
                    farms_j_set = set(farms_j)  # Convert farms_j to a set
                    common_farms = farms_i_set & farms_j_set  # Get the common farms
                    if len(common_farms) >= self.min_common_farms:  # Check if the number of common farms is greater than min_common_farms
                        _, dev_power_j, sigma_power_j = self.calculate_avg_and_deviation(period_j, farms_j)
                        # Compute the Pearson similarity between period_i and period_j
                        w_ij = self.pearson_similarity(common_farms, dev_power_i, sigma_power_i,  dev_power_j, sigma_power_j)
                        sl.add((-w_ij, period_j))  # Add the similarity to the SortedList
                        if len(sl) > self.K:  # Keep only the K most similar periods
                            del sl[-1]  # Delete the last element
            self.neighbors[period_i] = sl  # Store the K most similar periods
            # Log the progress
            if count % 5000 == 0:
                logger.info(f'Total processed periods: {count}')
            count+=1
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
