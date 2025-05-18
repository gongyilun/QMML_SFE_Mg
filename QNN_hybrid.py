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
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
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
FEATURE_MAP_REPS = 1
ANSATZ_REPS = 3

# Training hyperparameters
LEARNING_RATE = 0.01
BATCH_SIZE = 30
NUM_EPOCHS = 100 # Adjust as needed

# K-fold cross-validation parameters
N_REPEATS = 500
TEST_SIZE = 1


### Feature Map, Ansatz, then QNN Constructor
# a. Feature Map: Encodes NUM_FEATURES into NUM_QUBITS
# ParameterVector for input features
input_params = ParameterVector("x", NUM_FEATURES)

feature_map_template = PauliFeatureMap(
    feature_dimension=NUM_FEATURES, # This tells the template how many input parameters it structurally needs
    reps=FEATURE_MAP_REPS,
    entanglement='linear'
)

# Assign the *specific* input parameters from the vector to the template's parameter slots
# This creates a new circuit instance containing parameters ONLY from input_params (size NUM_FEATURES)
feature_map = feature_map_template.assign_parameters(input_params)
print(f"Assigned feature map parameters: {feature_map.num_parameters}")

# Create a template to find out how many parameters it needs structurally
ansatz_template = RealAmplitudes(NUM_QUBITS, reps=ANSATZ_REPS, entanglement="linear")
# ParameterVector for trainable weights - sized based on the template's structural parameters
num_ansatz_params = ansatz_template.num_parameters # This was correctly calculated as 12
weight_params = ParameterVector("θ", num_ansatz_params)

# Create the ansatz circuit instance by assigning the weight parameters to the template
ansatz = ansatz_template.assign_parameters(weight_params)
print(f"Assigned ansatz parameters: {ansatz.num_parameters}")

# c. Combine into a full quantum circuit
qc = QuantumCircuit(NUM_QUBITS)
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)

print(f"Total circuit parameters in qc: {qc.num_parameters}")


# d. Define Observable(s)
# For a single output, measure the expectation value of Pauli Z on the first qubit
# The output of EstimatorQNN will be in the range [-1, 1] for Pauli observables
from qiskit.quantum_info import SparsePauliOp
observable = SparsePauliOp.from_list([("Z" + "I" * (NUM_QUBITS - 1), 1.0)])
# If you have multiple qubits and want to combine their measurements, you can define multiple observables
# or a more complex one. For instance, if NUM_TARGETS > 1 or you want a richer output from QNN:
# observables = [SparsePauliOp(f"{'I'*i}Z{'I'*(NUM_QUBITS-1-i)}") for i in range(NUM_QUBITS)]
# This would give NUM_QUBITS outputs from the QNN.

# --- 3. EstimatorQNN ---
# Uses Qiskit's Estimator primitive for expectation value computations
# By default, Estimator uses a local statevector simulator.
# For real hardware or more advanced simulation, configure the Estimator.
estimator = Estimator()

qnn = EstimatorQNN(
    circuit=qc,
    estimator=estimator,
    input_params=input_params,
    weight_params=weight_params, # Parameters for trainable weights
    observables=observable,      # Observable to measure
    input_gradients=False       # Set to True if you need gradients w.r.t. inputs
)

# --- 4. TorchConnector ---
# Wrap the QNN into a PyTorch module
initial_weights = 0.01 * (2 * np.random.rand(qnn.num_weights) - 1)
qnn_torch_model = TorchConnector(qnn, initial_weights=torch.tensor(initial_weights, dtype=torch.float32))


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


if __name__ == "__main__":
    date = '05_15_25_0'
    file_name = f'QNN/result/{date}.csv'

    print("\n--- Loading and Preprocessing Data ---")

    df = pd.read_csv("qml_training-validation-data.csv")
    X = df[['Element', 'el_neg', 'B/GPa', 'Volume/A^3']].values
    y = df['SFE/mJm^-3'].values

    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    y = y_scaler.fit_transform(y.reshape(-1, 1))

    print('Total number of data: ', X.shape[0])
    rkf = RepeatedKFold(n_splits=X.shape[0] // TEST_SIZE, n_repeats=N_REPEATS)

    df = pd.DataFrame(columns=['element test', 'actual test', 'predicted test',
                               'element train', 'actual train', 'predicted train',
                               'R2 test', 'R2 train'])
    i = 0

    print("\n--- Start K-Fold Loop ---")

    for train_indices, test_indices in rkf.split(X):
        # model = qnn_torch_model # for purely quantum nn
        model = HybridModel(qnn_torch_model)  # classical modifications in the HybridQNN class

        ## Feature Scaling (important for many QML algorithms)
        ## The output of PauliFeatureMap is sensitive to input scale.
        ## Typically, inputs are scaled to [0, pi] or [-1, 1] or similar.
        ## If your feature map expects angles (like many rotations), scaling to [0, pi] is common.
        ## PauliFeatureMap often works well with inputs in [-1, 1] or [0, 1].

        # scaler_X = MinMaxScaler(feature_range=(-1, 1)) # Or (0, np.pi) if your feature map implies angles
        # X = scaler_X.fit_transform(X)
        # y_scaler = MinMaxScaler(feature_range=(-1, 1))
        # y = y_scaler.fit_transform(y.reshape(-1,1))

        ## Target variable scaling (if it's a regression task and target has wide range)
        ## For QNN output in [-1,1], you might want to scale y_data to this range or use a final classical layer to adapt.
        ## For simplicity, let's assume y_data is already in a suitable range or we handle it with loss/activation.
        ## If y_data is e.g. 0 or 1 for classification, adjust loss function accordingly.

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, y_train, X_test, y_test, element_test, element_train = prepare_dataset_k_fold(X, y, train_indices, test_indices)

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32)

        # Create DataLoaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_dataset = TensorDataset(X_test_t, y_test_t)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        print(f"Training data shape: X_train_t: {X_train_t.shape}, y_train_t: {y_train_t.shape}")
        print(f"Testing data shape: X_test_t: {X_test_t.shape}, y_test_t: {y_test_t.shape}")

        # --- 7. Loss Function and Optimizer ---
        # For regression with QNN output in [-1, 1], MSELoss is common.
        # The QNN output (expectation value) is typically in [-1, 1].
        # If your target y_data is not in this range, you might need to:
        #   1. Scale y_data to [-1, 1].
        #   2. Add a classical layer after the QNN to map the output to the desired range.
        #   3. Use a different observable strategy if direct mapping is hard.

        # criterion = nn.MSELoss()
        criterion = nn.HuberLoss()  # maybe less prone to extreme values

        # For binary classification (0 or 1 target):
        # You might scale the QNN output (e.g., (output + 1) / 2 to get [0,1]) and then use nn.BCELoss()
        # Or use nn.BCEWithLogitsLoss() if your QNN output is treated as logits (less common for direct EstimatorQNN output).

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
                loss = criterion(outputs, batch_y)  # Calculate loss
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
                    loss_test = criterion(outputs_test, batch_y_test)
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
        all_preds = y_scaler.inverse_transform(all_preds.reshape(-1, 1))
        all_targets = y_scaler.inverse_transform(all_targets.reshape(-1, 1))

        all_preds_train = np.array(all_preds_train)
        all_targets_train = np.array(all_targets_train)
        all_preds_train = y_scaler.inverse_transform(all_preds_train.reshape(-1, 1))
        all_targets_train = y_scaler.inverse_transform(all_targets_train.reshape(-1, 1))

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

        # Add to dataframe
        new_row = {'element test': element_test,
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

    # To make predictions on new data:
    # new_data_np = np.array([[val1, val2, val3], ...]) # Your new data
    # new_data_scaled = scaler_X.transform(new_data_np) # Don't forget to scale
    # new_data_t = torch.tensor(new_data_scaled, dtype=torch.float32)
    # with torch.no_grad():
    #     predictions = model(new_data_t)
    # print(f"Predictions for new data: {predictions.numpy()}")


