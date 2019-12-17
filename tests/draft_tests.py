import pandas as pd
from pathlib import Path
from src.preprocessing.Preprocessor import Preprocessor

data_path = Path(
            'data',
            'train.tsv'
        )

X = pd.read_csv(data_path, delim_whitespace=True, index_col='id_job')

data_path_test = Path(
            'data',
            'test.tsv'
        )

X_test = pd.read_csv(data_path_test, delim_whitespace=True, index_col='id_job')

preprocessor = Preprocessor(scaler_type='standard')
preprocessor.fit(X=X)

X_test_preprocessed = preprocessor.transform(X=X_test)

X_test_preprocessed.to_csv('data/test_proc.tsv', sep=' ')
