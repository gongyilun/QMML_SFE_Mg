### Import packages
import sys, getopt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler #StandardScaler is sensitive to outlier

from qiskit.circuit.library import PauliFeatureMap, RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import L_BFGS_B
# from IPython.display import clear_output

### Globals
# For reproducibility
np.random.seed(42)

# Fixed feature sizes
NUM_FEATURES = 3

# Quantum circuit parameters
IF_PAULI_FEATURE_MAP_LIST = [True, False]
FEATURE_MAP_REPS_LIST = [1, 2, 3, 4, 5]
ANSATZ_REPS_LIST = [1, 2, 3, 4, 5]
ENTANGLEMENT_LIST = ['linear', 'full', 'circular']

# Training hyperparameters
LOSS_FUNCTION = 'cross_entropy'

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


def reconfig_quantum_kernel_vqc(if_pauli_feature_map, feature_reps, ansatz_reps, entangle, objective_func_vals):
    """
        Create a quantum kernel
        vqc = VQC(C=20.0, epsilon=0.2, quantum_kernel=kernel)

        Args:
            if_pauli_feature_map: If True, use a Pauli feature map, else ZZ feature map.
            feature_reps: Number of repetitions of quantum circuit for a feature map.
            ansatz_reps: Number of repetitions of quantum circuit for the ansatz.
            entangle: Entanglement type of the feature map.
            objective_func_vals: List to store the objective function values during training.

        Returns:
            qsvr: quantum kernel
    """
    # Define callback function for plotting the training progress
    def callback_graph(weights, obj_func_eval):
        # clear_output(wait=True)
        objective_func_vals.append(obj_func_eval)
        print(f"Iteration: {len(objective_func_vals)}, Objective function value: {obj_func_eval}")
        #plt.rcParams["figure.figsize"] = (12, 6)
        #plt.xlabel("Iteration")
        #plt.ylabel("Objective function value")
        #plt.plot(range(len(objective_func_vals)), objective_func_vals)
        #plt.show()

    # Configure feature map
    if if_pauli_feature_map:
        feature_map = PauliFeatureMap(feature_dimension=NUM_FEATURES, reps=feature_reps, entanglement=entangle)
    else:
        feature_map = ZZFeatureMap(feature_dimension=NUM_FEATURES, reps=feature_reps, entanglement=entangle)

    # Configure ansatz
    ansatz = RealAmplitudes(num_qubits=NUM_FEATURES, reps=ansatz_reps)

    # Configure optimizer
    optimizer = L_BFGS_B(ftol=0.000001)

    return VQC(feature_map=feature_map,
               ansatz=ansatz,
               optimizer=optimizer,
               callback=callback_graph,
               loss=LOSS_FUNCTION,
               )


def train_vqc(vqc, X_train, y_train, X_test):
    """
        Train based on X_train/y_train (after scaling), return prediction from X_test

        Args:
            vqc: quantum kernel
            X_train:
            y_train
            X_test

        Returns:
    """
    vqc.fit(X_train, np.concatenate(y_train))
    return vqc.predict(X_train), vqc.predict(X_test)


def get_arguments(argvs):
    _entangle = ''
    _feature_map_reps = ''
    _ansatz_reps = ''
    _if_pauli_feature = True
    try:
        opts, args = getopt.getopt(argvs, "h:e:f:a:p:", ["entangle=", "feature_map_reps=", "ansatz_reps=", "if_pauli_feature="])
    except getopt.GetoptError:
        print('VQC.py -e <entangle> -f <feature_map_reps> -a <ansatz_reps> -p <if_pauli_feature>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('VQC.py -e <entangle> -f <feature_map_reps> -a <ansatz_reps> -p <if_pauli_feature>')
            sys.exit()
        elif opt in ("-e", "--entangle"):
            _entangle = arg
        elif opt in ("-f", "--feature_map_reps"):
            _feature_map_reps = int(arg)
        elif opt in ("-a", "--ansatz_reps"):
            _ansatz_reps = int(arg)
        elif opt in ("-p", "--if_pauli_feature"):
            _if_pauli_feature = eval(arg)
    return _entangle, _feature_map_reps, _ansatz_reps, _if_pauli_feature


if __name__ == "__main__":
    date = '05_31_25_0'
    tmp1, tmp2, tmp3, tmp4 = get_arguments(sys.argv[1:])
    if tmp1 != '':
        ENTANGLEMENT_LIST = [tmp1]
    if tmp2 != '':
        FEATURE_MAP_REPS_LIST = [tmp2]
    if tmp3 != '':
        ANSATZ_REPS_LIST = [tmp3]
    if tmp4 != '':
        IF_PAULI_FEATURE_MAP_LIST = [tmp4]
    print(f"\nFEATURE_MAP_REPS_LIST={FEATURE_MAP_REPS_LIST} "
          f"ANSATZ_REPS_LIST={ANSATZ_REPS_LIST} "
          f"ENTANGLEMENT_LIST={ENTANGLEMENT_LIST} "
          f"IF_PAULI_FEATURE_MAP_LIST={IF_PAULI_FEATURE_MAP_LIST} "
          f"date={date}")
    if len(FEATURE_MAP_REPS_LIST) == 1:
        FEATURE_MAP_REPS_LIST_NAME = FEATURE_MAP_REPS_LIST[0]
    else:
        FEATURE_MAP_REPS_LIST_NAME = FEATURE_MAP_REPS_LIST
    if len(ANSATZ_REPS_LIST) == 1:
        ANSATZ_REPS_LIST_NAME = ANSATZ_REPS_LIST[0]
    else:
        ANSATZ_REPS_LIST_NAME = ANSATZ_REPS_LIST
    if len(ENTANGLEMENT_LIST) == 1:
        ENTANGLEMENT_LIST_NAME = ENTANGLEMENT_LIST[0]
    else:
        ENTANGLEMENT_LIST_NAME = ENTANGLEMENT_LIST
    if len(IF_PAULI_FEATURE_MAP_LIST) == 1:
        IF_PAULI_FEATURE_MAP_LIST_NAME = IF_PAULI_FEATURE_MAP_LIST[0]
    else:
        IF_PAULI_FEATURE_MAP_LIST_NAME = IF_PAULI_FEATURE_MAP_LIST
    IF_PAULI_FEATURE_MAP_LIST_NAME = str(IF_PAULI_FEATURE_MAP_LIST_NAME).replace('False', 'ZZ').replace('True', 'Pauli')
    file_name = f'VQC/result/FMR_{FEATURE_MAP_REPS_LIST_NAME}_AR_{ANSATZ_REPS_LIST_NAME}_E_{ENTANGLEMENT_LIST_NAME}_P_{IF_PAULI_FEATURE_MAP_LIST_NAME}_{date}.csv'
    print('\nTo be saved to', file_name)

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

    df = pd.DataFrame(columns=['feature_map_name', 'feature_map_reps', 'ansatz_reps', 'entanglement',
                               'element test', 'actual test', 'predicted test',
                               'element train', 'actual train', 'predicted train',
                               'R2 test', 'R2 train'])
    i = 0

    print("\n--- Start K-Fold Loop ---")

    for train_indices, test_indices in rkf.split(X):
        X_train, y_train, X_test, y_test, element_test, element_train = prepare_dataset_k_fold(X, y, train_indices, test_indices)
        for if_pauli_feature_map in IF_PAULI_FEATURE_MAP_LIST:
            for feature_map_reps in FEATURE_MAP_REPS_LIST:
                for ansatz_reps in ANSATZ_REPS_LIST:
                    for entanglement in ENTANGLEMENT_LIST:
                        if if_pauli_feature_map:
                            feature_map_name = 'Pauli'
                        else:
                            feature_map_name = 'ZZ'
                        print(f'\nfeature_map_name:{feature_map_name} '
                              f'feature_map_reps:{feature_map_reps} '
                              f'ansatz_reps:{ansatz_reps} '
                              f'entanglement:{entanglement} '
                              f'element test:{element_test[0]} ')
                        # conf kernel
                        objective_func_vals = []
                        vqc = reconfig_quantum_kernel_vqc(if_pauli_feature_map=if_pauli_feature_map,
                                                           feature_reps=feature_map_reps,
                                                           ansatz_reps=ansatz_reps,
                                                           entangle=entanglement,
                                                           objective_func_vals=objective_func_vals,
                                                           )

                        # train
                        predict_train, predict_test = train_vqc(vqc, X_train, y_train, X_test)

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
                        new_row = {'feature_map_name': feature_map_name,
                                   'feature_map_reps': feature_map_reps,
                                   'ansatz_reps': ansatz_reps,
                                   'entanglement': entanglement,
                                   'element test': element_test,
                                   'actual test': np.array(all_targets).flatten(),
                                   'predicted test': np.array(all_preds).flatten(),
                                   'element train': element_train,
                                   'actual train': np.array(all_targets_train).flatten(),
                                   'predicted train': np.array(all_preds_train).flatten(),
                                   # 'R2 test': r2_score(y_test, predict_test),
                                   'R2 train': r2_score(y_train, predict_train),
                                   }
                        df.loc[len(df)] = new_row
                        with np.printoptions(linewidth=10000):
                            df.to_csv(file_name, index=False)  # update csv every loop
                        df.at[0, "info"] = [f"DATASET: {dataset_name}, LOSS: {LOSS_FUNCTION}"
                                            f"CLASSIFIER_THRESHOLD = {CLASSIFIER_THRESHOLD}"]
                        i += 1
