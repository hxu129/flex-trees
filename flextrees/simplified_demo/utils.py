"""
Helper functions for visualization and result display
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import os

def visualize_decision_tree(tree, feature_names=None, class_names=None, max_depth=None, 
                           save_path=None, title="Decision Tree Visualization"):
    """Visualize a single decision tree"""
    plt.figure(figsize=(20, 10))
    plot_tree(tree, feature_names=feature_names, class_names=class_names, 
             filled=True, rounded=True, max_depth=max_depth)
    plt.title(title, fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Decision tree image saved to: {save_path}")
    
    plt.show()

def visualize_rules(branches_df, top_n=10, save_path=None, title="Rules Visualization"):
    """
    Visualize the top N rules from the rule set
    
    Parameters:
        branches_df: Rules DataFrame
        top_n: Number of rules to display
        save_path: Path to save the image
        title: Image title
    """
    # Ensure rules count doesn't exceed actual rules
    n_rules = min(top_n, len(branches_df))
    
    # Select top n_rules rules
    top_rules = branches_df.iloc[:n_rules]
    
    # Extract probabilities and support for each rule
    probas = top_rules['probas'].values
    support = top_rules['branch_probability'].values
    
    # Create rule descriptions
    rule_descriptions = []
    for idx, row in top_rules.iterrows():
        features = [col for col in row.index if ('upper' in col or 'lower' in col) 
                    and not pd.isna(row[col]) and row[col] != np.inf]
        
        if not features:
            rule_descriptions.append("Root Node Rule")
            continue
            
        conditions = []
        for feature in features:
            # Extract feature name and boundary type
            feature_name, bound_type = feature.split('_')
            value = row[feature]
            
            # Skip invalid values
            if pd.isna(value) or value == np.inf:
                continue
                
            # Format condition
            if bound_type == 'upper':
                conditions.append(f"{feature_name} ≤ {value:.3f}")
            else:
                conditions.append(f"{feature_name} > {value:.3f}")
                
        rule_descriptions.append(" AND ".join(conditions))
    
    # Create bar charts for each rule's probability distribution
    fig, axs = plt.subplots(n_rules, 1, figsize=(12, n_rules * 2.5), constrained_layout=True)
    if n_rules == 1:
        axs = [axs]
        
    for i, (proba, rule, supp) in enumerate(zip(probas, rule_descriptions, support)):
        # Ensure proba is a numpy array
        if isinstance(proba, list):
            proba = np.array(proba)
        
        class_labels = [f"Class {j}" for j in range(len(proba))]
        axs[i].barh(class_labels, proba, color='skyblue')
        axs[i].set_title(f"Rule {i+1}: {rule}\nSupport: {supp:.3f}", fontsize=10)
        axs[i].set_xlim(0, 1)
        
        # Add probability values on the bars
        for j, p in enumerate(proba):
            axs[i].text(p + 0.01, j, f"{p:.3f}", va='center')
    
    plt.suptitle(title, fontsize=14)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Rules visualization saved to: {save_path}")
    
    plt.show()

def compare_models(client_models, global_model, metrics, title="Model Comparison"):
    """
    Compare client models and global model performance
    
    Parameters:
        client_models: List of client models
        global_model: Global model
        metrics: Performance metrics dictionary with 'acc' and 'f1'
        title: Chart title
    """
    n_clients = len(client_models)
    
    # Extract client model performance
    client_acc = [model['local_acc'] for model in client_models]
    client_f1 = [model['local_f1'] for model in client_models]
    
    # Global model performance
    global_acc = metrics['acc']
    global_f1 = metrics['f1']
    
    # Create comparison charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    client_indices = np.arange(n_clients)
    width = 0.35
    
    ax1.bar(client_indices, client_acc, width, label='Client Local Models')
    ax1.axhline(y=global_acc, color='r', linestyle='-', label='Global Model')
    ax1.set_xlabel('Client ID')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Comparison')
    ax1.set_xticks(client_indices)
    ax1.set_xticklabels([f'Client {i+1}' for i in range(n_clients)])
    ax1.set_ylim(0, 1)
    ax1.legend()
    
    # F1 score comparison
    ax2.bar(client_indices, client_f1, width, label='Client Local Models')
    ax2.axhline(y=global_f1, color='r', linestyle='-', label='Global Model')
    ax2.set_xlabel('Client ID')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score Comparison')
    ax2.set_xticks(client_indices)
    ax2.set_xticklabels([f'Client {i+1}' for i in range(n_clients)])
    ax2.set_ylim(0, 1)
    ax2.legend()
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def print_tree_paths(tree, feature_names=None, max_paths=10):
    """
    Print decision tree paths (from root to leaf)
    
    Parameters:
        tree: Decision tree model
        feature_names: List of feature names
        max_paths: Maximum number of paths to print
    """
    # Ensure feature names exist
    if feature_names is None:
        feature_names = [f"Feature{i}" for i in range(tree.tree_.n_features)]
    
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    
    # Store all paths
    paths = []
    
    def dfs(node_id, path):
        # If leaf node
        if children_left[node_id] == -1 and children_right[node_id] == -1:
            class_distribution = tree.tree_.value[node_id][0]
            class_distribution = class_distribution / class_distribution.sum()
            predicted_class = np.argmax(class_distribution)
            paths.append((path, class_distribution, predicted_class))
            return
        
        # Left subtree path
        if children_left[node_id] != -1:
            feature_name = feature_names[feature[node_id]]
            threshold_value = threshold[node_id]
            left_path = path + [(feature_name, "<=", threshold_value)]
            dfs(children_left[node_id], left_path)
        
        # Right subtree path
        if children_right[node_id] != -1:
            feature_name = feature_names[feature[node_id]]
            threshold_value = threshold[node_id]
            right_path = path + [(feature_name, ">", threshold_value)]
            dfs(children_right[node_id], right_path)
    
    # Start DFS from the root node
    dfs(0, [])
    
    # Print paths
    print(f"Decision tree has {len(paths)} paths, showing the first {min(max_paths, len(paths))}:")
    for i, (path, class_distribution, predicted_class) in enumerate(paths[:max_paths]):
        print(f"\nPath {i+1} (Predicts Class {predicted_class}):")
        for step in path:
            feature, op, value = step
            print(f"  {feature} {op} {value:.3f}")
        
        print("  Class probability distribution:")
        for j, prob in enumerate(class_distribution):
            print(f"   Class {j}: {prob:.3f}")
    
    return paths 