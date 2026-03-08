import sys
from functools import lru_cache

@lru_cache(maxsize=1)
def get_sklearn_deps():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    return RandomForestClassifier, roc_auc_score, train_test_split

if __name__ == '__main__':
    rf, auc, split = get_sklearn_deps()
    print('ok', rf.__name__, auc.__name__, split.__name__)
