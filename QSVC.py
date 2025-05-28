### Import packages
import sys, getopt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler #StandardScaler is sensitive to outlier

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel


### Globals
# For reproducibility
np.random.seed(42)

# Fixed feature sizes
NUM_FEATURES = 3
NUM_QUBITS = NUM_FEATURES
NUM_TARGETS = 1

# Quantum circuit parameters
FEATURE_MAP_REPS_LIST = [1, 2, 3, 4, 5]
REGU_PARA_LIST = [0.1, 1, 10, 100]
ENTANGLEMENT_LIST = ['linear', 'full', 'circular']

# Training hyperparameters
#LEARNING_RATE = 0.01
#BATCH_SIZE = 30
#NUM_EPOCHS = 100 # Adjust as needed

# K-fold cross-validation parameters
N_REPEATS = 10
TEST_SIZE = 1

# Data conf
CLASSIFIER_THRESHOLD = 19


def prepare_dataset_k_fold(X, y, train_indices, test_indices):
    # Separate train/test split
    X_train_raw, X_test_raw = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    # Separate element column from the actual features
    element_test = X_test_raw[:, 0]
    element_train = X_train_raw[:, 0]

    # Drop the element column (first column)
    X_train = X_train_raw[:, 1:]
    X_test = X_test_raw[:, 1:]

    full_X = np.vstack([X_train, X_test])

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(full_X)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test, element_test, element_train


def reconfig_quantum_kernel_qsvc(feature_dimension, C, reps, entangle):
    """
        Create a quantum kernel
        qsvc = QSVC(C=20.0, quantum_kernel=kernel)

        Args:
            feature_dimension: Dimension of the feature space.
            reps: Number of repetitions of quantum circuit.
            C: Regularization parameter.
               The strength of the regularization is inversely proportional to C.
               Must be strictly positive. The penalty is a squared l2.
            entangle: Entanglement type of the feature map.

        Returns:
            qsvc: quantum kernel
    """
    feature_map = ZZFeatureMap(feature_dimension=feature_dimension, reps=reps, entanglement=entangle, insert_barriers=True)
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    qsvc = QSVC(C=C, quantum_kernel=kernel)
    return qsvc


def train_qsvc(qsvc, X_train, y_train, X_test):
    """
        Train based on X_train/y_train (after scaling), return prediction from X_test

        Args:
            qsvc: quantum kernel
            X_train:
            y_train
            X_test

        Returns:
    """
    qsvc.fit(X_train, np.concatenate(y_train))
    return qsvc.predict(X_train), qsvc.predict(X_test)

def get_arguments(argvs):
    _entangle = ''
    _feature_map_reps = ''
    _regu_para = ''
    try:
        opts, args = getopt.getopt(argvs, "h:e:f:r:", ["entangle=", "feature_map_reps=", "_regu_para="])
    except getopt.GetoptError:
        print('QSVC.py -e <entangle> -f <feature_map_reps> -r <regu_para>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('QSVC.py -e <entangle> -f <feature_map_reps> -r <regu_para>')
            sys.exit()
        elif opt in ("-e", "--entangle"):
            _entangle = arg
        elif opt in ("-f", "--feature_map_reps"):
            _feature_map_reps = int(arg)
        elif opt in ("-r", "--regu_para"):
            _regu_para = float(arg)
    return _entangle, _feature_map_reps, _regu_para


if __name__ == "__main__":
    date = '28_19_25_1'
    tmp1, tmp2, tmp3 = get_arguments(sys.argv[1:])
    if tmp1 != '':
        ENTANGLEMENT_LIST = [tmp1]
    if tmp2 != '':
        FEATURE_MAP_REPS_LIST = [tmp2]
    if tmp3 != '':
        REGU_PARA_LIST = [tmp3]
    print(f"\nFEATURE_MAP_REPS_LIST={FEATURE_MAP_REPS_LIST} "
          f"REGU_PARA_LIST={REGU_PARA_LIST} "
          f"ENTANGLEMENT_LIST={ENTANGLEMENT_LIST} "
          f"date={date}")
    if len(FEATURE_MAP_REPS_LIST) == 1:
        FEATURE_MAP_REPS_LIST_NAME = FEATURE_MAP_REPS_LIST[0]
    else:
        FEATURE_MAP_REPS_LIST_NAME = FEATURE_MAP_REPS_LIST
    if len(REGU_PARA_LIST) == 1:
        REGU_PARA_LIST_NAME = REGU_PARA_LIST[0]
    else:
        REGU_PARA_LIST_NAME = REGU_PARA_LIST
    if len(ENTANGLEMENT_LIST) == 1:
        ENTANGLEMENT_LIST_NAME = ENTANGLEMENT_LIST[0]
    else:
        ENTANGLEMENT_LIST_NAME = ENTANGLEMENT_LIST
    file_name = (f'QSVC/result/FMR_{FEATURE_MAP_REPS_LIST_NAME}_'
                 f'R_{REGU_PARA_LIST_NAME}_E_{ENTANGLEMENT_LIST_NAME}_{date}.csv')

    print("\n--- Loading and Preprocessing Data ---")

    dataset_name = "qml_training-validation-data.csv"
    df = pd.read_csv(dataset_name)
    X = df[['Element', 'el_neg', 'B/GPa', 'Volume/A^3']].values
    y = df['SFE/mJm^-3'].values

    # Group data by classifier threshold
    for i in range(0, len(y)):
        if y[i] > CLASSIFIER_THRESHOLD:
            y[i] = 0
        else:
            y[i] = 1

    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    y = y_scaler.fit_transform(y.reshape(-1, 1))

    print('Total number of data: ', X.shape[0])
    rkf = RepeatedKFold(n_splits=X.shape[0] // TEST_SIZE, n_repeats=N_REPEATS)

    df = pd.DataFrame(columns=['C', 'reps', 'entanglement',
                               'element test', 'actual test', 'predicted test',
                               'element train', 'actual train', 'predicted train',
                               'R2 test', 'R2 train'])
    i = 0

    print("\n--- Start K-Fold Loop ---")

    for train_indices, test_indices in rkf.split(X):
        X_train, y_train, X_test, y_test, element_test, element_train = prepare_dataset_k_fold(X, y, train_indices, test_indices)
        for C_value in REGU_PARA_LIST:
            for feature_map_reps in FEATURE_MAP_REPS_LIST:
                for entanglement in ENTANGLEMENT_LIST:
                    print(f'C:{C_value} feature_map_reps:{feature_map_reps} entanglement:{entanglement}')
                    # conf kernel
                    qsvc = reconfig_quantum_kernel_qsvc(feature_dimension=NUM_FEATURES,
                                                        C=C_value,
                                                        reps=feature_map_reps,
                                                        entangle=entanglement)

                    # train
                    predict_train, predict_test = train_qsvc(qsvc, X_train, y_train, X_test)

                    # some conversions
                    all_preds = np.array(predict_test)
                    all_targets = np.array(y_test)
                    all_preds = y_scaler.inverse_transform(all_preds.reshape(-1, 1))
                    all_targets = y_scaler.inverse_transform(all_targets.reshape(-1, 1))

                    all_preds_train = np.array(predict_train)
                    all_targets_train = np.array(y_train)
                    all_preds_train = y_scaler.inverse_transform(all_preds_train.reshape(-1, 1))
                    all_targets_train = y_scaler.inverse_transform(all_targets_train.reshape(-1, 1))

                    # save data
                    new_row = {'C': C_value,
                               'reps': feature_map_reps,
                               'entanglement': entanglement,
                               'element test': element_test,
                               'actual test': np.array(all_targets).flatten(),
                               'predicted test': np.array(all_preds).flatten(),
                               'element train': element_train,
                               'actual train': np.array(all_targets_train).flatten(),
                               'predicted train': np.array(all_preds_train).flatten(),
                               #'R2 test': r2_score(y_test, predict_test),
                               'R2 train': r2_score(y_train, predict_train),
                               }
                    df.loc[len(df)] = new_row
                    with np.printoptions(linewidth=10000):
                        df.to_csv(file_name, index=False)  # update csv every loop
                    df.at[0, "info"] = [f"DATASET: {dataset_name}, "
                                        f"CLASSIFIER_THRESHOLD = {CLASSIFIER_THRESHOLD}"]
                    i += 1
