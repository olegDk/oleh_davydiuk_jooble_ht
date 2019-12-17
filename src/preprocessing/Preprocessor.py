import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler

pd.options.mode.chained_assignment = None

class Preprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, scaler_type: str  = 'standard'):
        """
        :param scaler_type: string indicating scaling type
        """
        super().__init__()
        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame = None, y: np.ndarray = None):
        """
        Fitting train statistics for scaling
        :param X: features dataframe
        :param y: labels np.ndarray undefined for transformers
        """

        # Getting features type
        self.features_type = X.iloc[0, 0].split(',')[0]
        # Splitting features string into integer features for each vacancy
        X_features = X['features'].str.split(',', expand=True)
        # Dropping unnecessary feature type column (with constant value) for further scaler fitting
        X_features.drop(X_features.columns[0], axis=1, inplace=True)

        return self.scaler.fit(X_features.values.astype(np.int))

    def transform(self, X: pd.DataFrame = None) -> pd.DataFrame:
        """
        Transform new data using statistics fitted to train set,
        also engineering new features: max_feature_2_index, max_feature_2_abs_mean_diff
        :param X: features dataframe
        :return: transformed features dataframe
        """

        # Splitting features string into integer features for each vacancy
        X_features = X['features'].str.split(',', expand=True)
        # Dropping unnecessary feature type column (with constant value) for further scaler transforming
        X_features.drop(X_features.columns[0], axis=1, inplace=True)
        # Further transformations and feature engineering using np.ndarray instead of pandas to increase speed
        X_features_values = X_features.values.astype(np.int)

        # Getting index of max feature for each vacancy
        max_indices = np.argmax(X_features_values, axis=1)
        # Getting max feature of each vacancy for further calculations
        max_values = np.amax(X_features_values, axis=1)
        # Mean of largest feature's column for each vacancy
        max_column_per_row_means = np.mean(X_features_values[:, max_indices].T, axis=1)
        # Calulating abs deviation of max feature from it's mean for each vacancy
        absolute_deviation = np.abs(np.subtract(max_values, max_column_per_row_means))

        # Scaling using fitted train statistics
        X_features_scaled = self.scaler.transform(X_features_values)

        # Converting to pd.DataFrame
        X_features_scaled_df = pd.DataFrame(X_features_scaled, index=X_features.index)
        columns_map = self.get_feature_column_names(list(X_features_scaled_df.columns))
        X_features_scaled_df.rename(index=str, columns=columns_map, inplace=True)

        # Adding required calculated columns
        X_features_scaled_df[f'max_feature_{self.features_type}_index'] = max_indices.astype(np.int)
        X_features_scaled_df[f'max_feature_{self.features_type}_abs_mean_diff'] = absolute_deviation.astype(np.double)

        return X_features_scaled_df

    def get_feature_column_names(self, columns: list) -> dict:
        """
        Maps indices of transformed dataframe to meaningful feature names
        :param columns: old feature names
        :return: dictionary that maps old feature names to new feature names
        """
        new_columns = [f'feature_{self.features_type}_stand_{i}' for i in columns]

        return dict(zip(columns, new_columns))
