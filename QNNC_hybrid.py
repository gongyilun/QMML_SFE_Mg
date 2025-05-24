import sys, getopt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliFeatureMap, RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_machine_learning.utils.loss_functions import L2Loss # Qiskit ML loss, or use PyTorch's
from qiskit.quantum_info import SparsePauliOp

### Globals
# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Fixed feature sizes
NUM_FEATURES = 3
NUM_QUBITS = NUM_FEATURES
NUM_TARGETS = 1

# Quantum circuit parameters
FEATURE_MAP_REPS_LIST = [1, 2, 3, 4, 5, 6]
ANSATZ_REPS_LIST = [1, 2, 3, 4, 5, 6]
ENTANGLEMENT_LIST = ['linear', 'full', 'circular']

# Training hyperparameters
LEARNING_RATE = 0.01
BATCH_SIZE = 30
NUM_EPOCHS = 100 # Adjust as needed

# K-fold cross-validation parameters
N_REPEATS = 10
TEST_SIZE = 1      # Leave-one-out cross-validation (LOOCV) is suggested since the sample size is too small

# Data conf
CLASSIFIER_THRESHOLD = 17

def get_qnn_torch_model(entangle, feature_map_reps, ansatz_reps):
    ### Feature Map, Ansatz, then QNN Constructor
    # a. Feature Map: Encodes NUM_FEATURES into NUM_QUBITS
    # ParameterVector for input features
    input_params = ParameterVector("x", NUM_FEATURES)

    feature_map_template = PauliFeatureMap(
        feature_dimension=NUM_FEATURES, # This tells the template how many input parameters it structurally needs
        reps=feature_map_reps,
        entanglement=entangle
    )

    # Assign the *specific* input parameters from the vector to the template's parameter slots
    # This creates a new circuit instance containing parameters ONLY from input_params (size NUM_FEATURES)
    feature_map = feature_map_template.assign_parameters(input_params)
    print(f"Assigned feature map parameters: {feature_map.num_parameters}")

    # Create a template to find out how many parameters it needs structurally
    ansatz_template = RealAmplitudes(NUM_QUBITS, reps=ansatz_reps, entanglement=entangle)
    # ParameterVector for trainable weights - sized based on the template's structural parameters
    num_ansatz_params = ansatz_template.num_parameters # This was correctly calculated as 12
    weight_params = ParameterVector("θ", num_ansatz_params)

    # Create the ansatz circuit instance by assigning the weight parameters to the template
    ansatz = ansatz_template.assign_parameters(weight_params)
    print(f"Assigned ansatz parameters: {ansatz.num_parameters}")

    # c. Combine into a full quantum circuit
    qc = QNNCircuit(
        feature_map=feature_map,
        ansatz=ansatz_template,
    )

    # example 5.2 from Qiskit guide on binary classification
    parity = lambda x: "{:b}".format(x).count("1") % 2
    output_shape = 2  # parity = 0, 1

    sampler = Sampler()

    qnn = SamplerQNN(
        circuit=qc,
        interpret=parity,
        output_shape=output_shape,
        sampler=sampler,
        sparse=False,
        input_gradients=False,         # Set to True if you need gradients w.r.t. inputs
    )

    # --- 4. TorchConnector ---
    # Wrap the QNN into a PyTorch module
    initial_weights = 0.01 * (2 * np.random.rand(qnn.num_weights) - 1)
    qnn_torch_model = TorchConnector(qnn, initial_weights=torch.tensor(initial_weights, dtype=torch.float32))

    return qnn_torch_model


class HybridModel(nn.Module):
    def __init__(self, qnn_model):
        super().__init__()
        # Example: Add classical layers if needed
        # self.classical_pre = nn.Linear(NUM_FEATURES, NUM_FEATURES) # If you want to pre-process features
        self.qnn = qnn_model
        # Example: Add classical layers after the QNN
        # self.classical_post = nn.Linear(qnn_model.output_shape[0], NUM_TARGETS) # output_shape[0] is num_observables
                                                                                # If qnn_model.output_shape is (1,), then it's 1.

    def forward(self, x):
        # x = self.classical_pre(x) # If using classical_pre
        x = self.qnn(x)
        # x = self.classical_post(x) # If using classical_post
        return x


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

def get_arguments(argvs):
    _entangle = ''
    _feature_map_reps = ''
    _ansatz_reps = ''
    try:
        opts, args = getopt.getopt(argvs, "h:e:f:a:", ["entangle=", "feature_map_reps=", "ansatz_reps="])
    except getopt.GetoptError:
        print('QNNC_hybrid.py -e <entangle> -f <feature_map_reps> -a <ansatz_reps>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('QNNC_hybrid.py -e <entangle> -f <feature_map_reps> -a <ansatz_reps>')
            sys.exit()
        elif opt in ("-e", "--entangle"):
            _entangle = arg
        elif opt in ("-f", "--feature_map_reps"):
            _feature_map_reps = int(arg)
        elif opt in ("-a", "--ansatz_reps"):
            _ansatz_reps = int(arg)
    return _entangle, _feature_map_reps, _ansatz_reps


if __name__ == "__main__":
    date = '24_19_25_1'
    tmp1, tmp2, tmp3 = get_arguments(sys.argv[1:])
    if tmp1 != '':
        ENTANGLEMENT_LIST = [tmp1]
    if tmp2 != '':
        FEATURE_MAP_REPS_LIST = [tmp2]
    if tmp3 != '':
        ANSATZ_REPS_LIST = [tmp3]
    print(f"\nFEATURE_MAP_REPS_LIST={FEATURE_MAP_REPS_LIST} "
          f"ANSATZ_REPS_LIST={ANSATZ_REPS_LIST} ENTANGLEMENT_LIST={ENTANGLEMENT_LIST} date={date}")
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
    file_name = f'QNNC/result/FMR_{FEATURE_MAP_REPS_LIST_NAME}_AR_{ANSATZ_REPS_LIST_NAME}_E_{ENTANGLEMENT_LIST_NAME}_{date}.csv'

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

    #y_scaler = MinMaxScaler(feature_range=(-1, 1))
    #y = y_scaler.fit_transform(y.reshape(-1, 1))

    print('Total number of data: ', X.shape[0])
    rkf = RepeatedKFold(n_splits=X.shape[0] // TEST_SIZE, n_repeats=N_REPEATS)

    df = pd.DataFrame(columns=['entanglement', 'feature_map_reps', 'ansatz_reps',
                               'element test', 'actual test', 'predicted test',
                               'element train', 'actual train', 'predicted train',
                               'R2 test', 'R2 train'])
    i = 0

    LOSS = nn.CrossEntropyLoss()  # use torch.long
    # LOSS = nn.MSELoss() # haven't tried
    # LOSS = nn.BCELoss() # use torch.float

    print("\n--- Start K-Fold Loop ---")

    for train_indices, test_indices in rkf.split(X):
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, y_train, X_test, y_test, element_test, element_train = prepare_dataset_k_fold(X, y, train_indices, test_indices)

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.long)

        # Create DataLoaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_dataset = TensorDataset(X_test_t, y_test_t)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        print(f"Training data shape: X_train_t: {X_train_t.shape}, y_train_t: {y_train_t.shape}")
        print(f"Testing data shape: X_test_t: {X_test_t.shape}, y_test_t: {y_test_t.shape}")


        # For binary classification (0 or 1 target):
        # You might scale the QNN output (e.g., (output + 1) / 2 to get [0,1]) and then use nn.BCELoss()
        # Or use nn.BCEWithLogitsLoss() if your QNN output is treated as logits (less common for direct EstimatorQNN output).

        for entanglement in ENTANGLEMENT_LIST:
            for feature_map_reps in FEATURE_MAP_REPS_LIST:
                for ansatz_reps in ANSATZ_REPS_LIST:
                    # Build model
                    # model = qnn_torch_model # for purely quantum nn
                    model = HybridModel(get_qnn_torch_model(entangle=entanglement,
                                                            feature_map_reps=feature_map_reps,
                                                            ansatz_reps=ansatz_reps))  # Classical modifications in the HybridQNN class
                    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

                    print(f"\n--- Starting Training {i}th---")
                    train_losses = []
                    test_losses = []

                    for epoch in range(NUM_EPOCHS):
                        # Training phase
                        model.train()
                        running_loss = 0.0
                        for batch_X, batch_y in train_loader:
                            optimizer.zero_grad()  # Clear gradients
                            outputs = model(batch_X)  # Forward pass
                            loss = LOSS(outputs, batch_y)  # Calculate loss
                            loss.backward()  # Backward pass (compute gradients)
                            optimizer.step()  # Update weights
                            running_loss += loss.item() * batch_X.size(0)

                        epoch_loss = running_loss / len(train_loader.dataset)
                        train_losses.append(epoch_loss)

                        # Validation/Test phase
                        model.eval()
                        test_loss = 0.0
                        with torch.no_grad():  # Disable gradient calculations
                            for batch_X_test, batch_y_test in test_loader:
                                outputs_test = model(batch_X_test)
                                loss_test = LOSS(outputs_test, batch_y_test)
                                test_loss += loss_test.item() * batch_X_test.size(0)

                        epoch_test_loss = test_loss / len(test_loader.dataset)
                        test_losses.append(epoch_test_loss)

                        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {epoch_loss:.4f}, Test Loss: {epoch_test_loss:.4f}")

                    print("--- Training Finished ---")

                    # --- 9. Plotting Training History (Optional) ---
                    # plt.figure(figsize=(10, 5))
                    # plt.plot(train_losses, label='Training Loss')
                    # plt.plot(test_losses, label='Test Loss')
                    # plt.title('Training and Test Loss Over Epochs')
                    # plt.xlabel('Epoch')
                    # plt.ylabel('MSE Loss')
                    # plt.legend()
                    # plt.grid(True)
                    # plt.show()

                    # --- 10. Evaluation on Test (and Training) Set ---
                    model.eval()
                    all_preds = []
                    all_targets = []
                    all_preds_train = []
                    all_targets_train = []
                    with torch.no_grad():
                        for batch_X_test, batch_y_test in test_loader:
                            outputs_test = model(batch_X_test)
                            all_preds.extend(outputs_test.cpu().numpy())
                            all_targets.extend(batch_y_test.cpu().numpy())
                        for batch_X_train, batch_y_train in train_loader:
                            outputs_train = model(batch_X_train)
                            all_preds_train.extend(outputs_train.cpu().numpy())
                            all_targets_train.extend(batch_y_train.cpu().numpy())

                    all_preds = np.array(all_preds)
                    all_targets = np.array(all_targets)
                    all_preds_temp = []
                    for item in all_preds:
                        if item[1] > item[0]:
                            all_preds_temp.append(1)
                        else:
                            all_preds_temp.append(0)
                    all_preds = np.array(all_preds_temp)

                    all_preds_train = np.array(all_preds_train)
                    all_targets_train = np.array(all_targets_train)
                    all_preds_temp = []
                    for item in all_preds_train:
                        if item[1] > item[0]:
                            all_preds_temp.append(1)
                        else:
                            all_preds_temp.append(0)
                    all_preds_train = np.array(all_preds_temp)

                    # Example: Scatter plot for regression
                    # if NUM_TARGETS == 1: # Simple plot if single target variable
                    # plt.figure(figsize=(8, 8))
                    # plt.scatter(all_targets, all_preds, alpha=0.5)
                    # plt.plot([min(all_targets.min(), all_preds.min()), max(all_targets.max(), all_preds.max())],
                    #         [min(all_targets.min(), all_preds.min()), max(all_targets.max(), all_preds.max())],
                    #         'k--', lw=2, label='Ideal')
                    # plt.xlabel('Actual Values')
                    # plt.ylabel('Predicted Values')
                    # plt.title('Actual vs. Predicted Values on Test Set')
                    # plt.legend()
                    # plt.grid(True)
                    # plt.show()

                    # Further evaluation metrics can be added here (e.g., R-squared for regression, accuracy for classification)
                    final_mse = mean_squared_error(all_targets, all_preds)
                    final_r2 = r2_score(all_targets, all_preds)
                    print(f"\n--- Final Test Set Evaluation ---")
                    print(f"Mean Squared Error (MSE): {final_mse:.4f}")
                    print(f"R-squared (R2 Score): {final_r2:.4f}")

                    final_train_mse = mean_squared_error(all_targets_train, all_preds_train)
                    final_train_r2 = r2_score(all_targets_train, all_preds_train)
                    print(f"\n--- Final Train Set Evaluation ---")
                    print(f"Mean Squared Error (MSE): {final_train_mse:.4f}")
                    print(f"R-squared (R2 Score): {final_train_r2:.4f}")

                    print(f"\n--- Done for entanglement: {entanglement}, feature_map_reps: {feature_map_reps}, ansatz_reps: {ansatz_reps} ---")

                    # Add to dataframe
                    new_row = {'entanglement': entanglement,
                               'feature_map_reps': feature_map_reps,
                               'ansatz_reps': ansatz_reps,
                               'element test': element_test,
                               'actual test': np.array(all_targets).flatten(),
                               'predicted test': np.array(all_preds).flatten(),
                               'element train': element_train,
                               'actual train': np.array(all_targets_train).flatten(),
                               'predicted train': np.array(all_preds_train).flatten(),
                               'R2 test': final_r2,
                               'R2 train': final_train_r2,
                               }
                    df.loc[len(df)] = new_row
                    df.to_csv(file_name, index=False)  # update csv every loop
                    df.at[0, "info"] = [f"DATASET: {dataset_name}, LEARNING_RATE = {LEARNING_RATE}, "
                                        f"BATCH_SIZE = {BATCH_SIZE}, NUM_EPOCHS = {NUM_EPOCHS}, LOSS: {LOSS},"
                                        f"CLASSIFIER_THRESHOLD = {CLASSIFIER_THRESHOLD}"]
