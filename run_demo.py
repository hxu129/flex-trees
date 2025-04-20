#!/usr/bin/env python
# coding: utf-8

"""
Demo: Interpretable Client Decision Tree Aggregation (ICDTA4FL) with FLEXible

This script demonstrates the ICDTA4FL process using the `flex-trees` library, based on the paper
"An interpretable client decision tree aggregation process for Federated Learning".
"""
# Import necessary libraries from `flex`, `flextrees`, and standard Python packages.
import numpy as np
import os
from datetime import datetime

# FLEX imports
from flex.data import FedDataDistribution, FedDatasetConfig, Dataset
from flex.pool import FlexPool

# flextrees imports for datasets and ICDTA4FL process
from flextrees.datasets import nursery, adult, credit2, bank # Available datasets
from flextrees.pool import (
    # Initialization and Config
    init_server_model_dtfl,
    deploy_local_model_config_dtfl,
    # Local Training
    train_local_model,
    # Filtering Steps (Algorithm 1, Steps 2-5)
    collect_clients_trees,
    set_local_trees_to_server,
    send_all_trees_to_client,
    evaluate_global_trees,
    collect_local_evaluations_from_clients,
    aggregate_thresholds_and_select,
    set_selected_trees_to_server,
    # Aggregation Steps (Algorithm 1, Step 6-9)
    collect_clients_weights_dtfl,
    aggregate_dtfl_prunning, # Use the pruning version
    set_aggregated_weights_dtfl,
    # Evaluation Steps (Algorithm 1, Step 10+)
    deploy_global_model,
    evaluate_global_model,
    evaluate_server_model_dtfl,
    aggregate_client_dts
)

# Configuration parameters
# Experiment Parameters
N_CLIENTS = 2 # Number of clients (e.g., 2, 5, 10, 20, 50)
DATASET_FUNC = adult # Choose dataset function (nursery, adult, car, credit2)
DATA_DISTRIBUTION = 'iid' # 'iid' or 'non-iid'

# Model Parameters
# MODEL_TYPE = 'id3'  # Choose 'id3', 'cart', or 'c45'
MODEL_TYPE = 'cart'  # Choose 'id3', 'cart', or 'c45'
# MODEL_TYPE = 'c45'  # Choose 'id3', 'cart', or 'c45'
MAX_DEPTH = 2 if MODEL_TYPE == 'id3' else 2 # ID3 depth related to features, CART/C45 fixed
CRITERION = 'entropy' if MODEL_TYPE == 'id3' else 'gini'
RULES_THRESHOLD = 99999999 # Max rules during aggregation filtering

# Filtering Parameters (Algorithm 1, Step 5)
FILTERING_METHOD = 'mean'
PERCENTILE_VALUE = 75 # Used only if FILTERING_METHOD is 'percentile'
ACC_THRESHOLD = 0.60
F1_THRESHOLD = 0.50

# Results Saving
RESULTS_DIR = "results_icdtafl"
EXPLANATIONS_DIR = "explanations_icdtafl"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Generate a unique filename for this run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
RESULTS_FILENAME = os.path.join(
    RESULTS_DIR,
    f"results_{DATASET_FUNC.__name__}_{MODEL_TYPE}_clients{N_CLIENTS}_{DATA_DISTRIBUTION}_{timestamp}.csv"
)
print(f"Results will be saved to: {RESULTS_FILENAME}")

# Data Preprocessing Flag
NEEDS_CATEGORICAL = MODEL_TYPE in ['id3', 'c45']

def main():
    # Data Preparation
    print(f"Loading dataset: {DATASET_FUNC.__name__}")
    # Load data, potentially getting feature names if needed by the model/preprocessing
    try:
        # Datasets like nursery, adult return feature names
        train_data, test_data, features_names = DATASET_FUNC(ret_feature_names=True, categorical=NEEDS_CATEGORICAL)
        print(f"Loaded features: {features_names}")
    except ValueError:
        # Other datasets might not return feature names
        train_data, test_data = DATASET_FUNC(categorical=NEEDS_CATEGORICAL)
        # Create dummy feature names if needed (e.g., for ID3 depth calculation)
        num_features = train_data.X_data.shape[1]
        features_names = [f'x{i}' for i in range(num_features)] + ['label']
        print(f"Loaded {num_features} features.")

    # Recalculate max_depth for ID3 if it depends on the number of features
    max_depth = MAX_DEPTH
    if MODEL_TYPE == 'id3':
        max_depth = len(features_names[:-1]) // 2
        print(f"Recalculated MAX_DEPTH for ID3: {max_depth}")

    print(f"\nFederating data ({DATA_DISTRIBUTION} distribution across {N_CLIENTS} clients)...")
    if DATA_DISTRIBUTION == 'iid':
        federated_data = FedDataDistribution.iid_distribution(
            centralized_data=train_data,
            n_nodes=N_CLIENTS
        )
    else:
        # Example for Non-IID (Dirichlet distribution simulation)
        alpha = 0.5 # Smaller alpha = more non-iid
        weights = np.random.dirichlet(np.repeat(alpha, N_CLIENTS), 1)[0]
        config_nidd = FedDatasetConfig(seed=0, n_nodes=N_CLIENTS,
                                    replacement=False,
                                    weights=weights)
        federated_data = FedDataDistribution.from_config(
            centralized_data=train_data,
            config=config_nidd
        )
    print("Data federated.")

    # Initialize Federated Environment
    print("\nInitializing Federated Pool...")
    
    # Prepare the configuration dictionary for init_server_model_dtfl
    icdtafl_config = {
        'local_model_params': {
            'max_depth': max_depth,
            'criterion': CRITERION,
            'splitter': 'best',
            'model_type': MODEL_TYPE,
        },
        'global_model_params': {
            'rules_threshold': RULES_THRESHOLD
        },
        'filtering_params': {
            'acc_threshold': ACC_THRESHOLD,
            'f1_threshold': F1_THRESHOLD,
            'filter_method': FILTERING_METHOD,
            'filter_value': PERCENTILE_VALUE if FILTERING_METHOD == 'percentile' else 0
        }
    }
    
    pool = FlexPool.client_server_pool(federated_data, init_server_model_dtfl, config=icdtafl_config)

    clients = pool.clients
    aggregator = pool.aggregators
    server = pool.servers

    print("Federated Pool initialized.")
    print(f"Number of clients: {len(clients)}")
    print(f"Number of aggregators: {len(aggregator)}")
    print(f"Number of servers: {len(server)}")

    # Run ICDTA4FL Training Pipeline
    # Step 1 (Client): Train local DTs
    print("\n--- Starting Training Pipeline ---")
    print("Step 1: Deploying config and training local models...")
    server.map(deploy_local_model_config_dtfl, dst_pool=clients)
    clients.map(train_local_model) # This also extracts rules into client's FlexModel
    print("Step 1: Local models trained and rules extracted.")

    # Step 2-5 (Server/Client/Server): Filter weak trees
    print("\nStep 2-5: Filtering weak decision trees...")
    # a. Server collects trees from clients (via aggregator)
    aggregator.map(collect_clients_trees, dst_pool=clients)
    aggregator.map(aggregate_client_dts)
    aggregator.map(set_local_trees_to_server, dst_pool=server)
    print("  - Server collected local trees.")

    # b. Server sends all trees to all clients
    server.map(send_all_trees_to_client, dst_pool=clients)
    print("  - Server sent trees to clients for evaluation.")

    # c. Clients evaluate all trees on their local test data
    clients.map(evaluate_global_trees)
    print("  - Clients evaluated trees.")

    # d. Server collects evaluation metrics (via aggregator)
    aggregator.map(collect_local_evaluations_from_clients, dst_pool=clients)
    print("  - Server collecting evaluations.")

    # e. Aggregator selects trees based on the filter
    server_model = server._models.get('server')
    filter_params = server_model['filtering_params']
    print(f"  - Applying filter: {filter_params['filter_method']}")
    aggregator.map(
        aggregate_thresholds_and_select,
        acc_threshold=filter_params['acc_threshold'],
        f1_threshold=filter_params['f1_threshold'],
        func_str=filter_params['filter_method'],
        func_kwargs=filter_params['filter_value']
    )
    aggregator.map(set_selected_trees_to_server, dst_pool=server)

    # Retrieve selected indices from the server model
    server_model = server._models.get('server')  # Get server model again to ensure fresh data
    selected_indices = server_model.get('selected_trees', list(range(N_CLIENTS)))
    if isinstance(selected_indices, np.ndarray):
        selected_indices = selected_indices.tolist()
    print(f"Step 2-5: Filtering complete. Selected {len(selected_indices)} trees (Indices: {selected_indices}).")

    # Step 6-9 (Server): Aggregate rules and build global DT
    print("\nStep 6-9: Aggregating rules and building global model...")
    # Collect rules ONLY from clients corresponding to selected trees
    aggregator.map(collect_clients_weights_dtfl, dst_pool=clients)
    print("  - Collected rules from clients (will be filtered by aggregator).")

    # Aggregate using the pruning version, passing selected indices
    aggregator.map(aggregate_dtfl_prunning, selected_indexes=selected_indices)
    print("  - Aggregated rules from selected trees.")

    # Set the final global model on the server
    aggregator.map(set_aggregated_weights_dtfl, dst_pool=server)
    print("Step 6-9: Global model created and set on server.")

    # Optional: Server evaluates on global test set
    if test_data:
        print("\n--- Server Evaluation ---")
        X_test_server, y_test_server = test_data.to_numpy()
        server.map(evaluate_server_model_dtfl, test_data=X_test_server, test_labels=y_test_server, filename=RESULTS_FILENAME)
        print("Server evaluation complete.")
    else:
        print("\n--- Server Evaluation ---")
        print("No global test data provided for server evaluation.")

    # Perform Prediction (Evaluate Global Model on Clients)
    print("\n--- Client Evaluation of Global Model ---")
    # Deploy the final global model from server to clients
    server.map(deploy_global_model, dst_pool=clients)
    print("  - Deployed global model to clients.")

    # Clients evaluate the global model on their local test data
    clients.map(evaluate_global_model, filename=RESULTS_FILENAME)
    print("Client evaluation complete.")
    print(f"Detailed results saved in: {RESULTS_FILENAME}")
    print(f"Explanations (if generated) might be in: {EXPLANATIONS_DIR} (check primitives_dtfl.py for exact path logic)")

    print("\n--- ICDTA4FL Demo Finished ---")

if __name__ == "__main__":
    main() 