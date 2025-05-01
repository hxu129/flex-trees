import json
from gfn_trees import SimpleConjunctionSet, compare_trees, SimpleBranch
import numpy as np
from gfn_trees import jensenshannon
import pandas as pd # Import pandas

def extract_rules_from_json_tree(json_tree):
    """Extract rules (SimpleBranch) from a tree in the JSON BFS format.

    Args:
        json_tree (list): A list of node dictionaries representing the tree in BFS order.

    Returns:
        tuple: A tuple containing:
            - list[SimpleBranch]: The list of extracted branches.
            - list[str]: The list of feature names (conditions).
            - list[str]: The list of class names.
    """
    if not json_tree:
        return [], [], []

    feature_names_map = {}
    classes_map = {}
    feature_count = 0
    class_count = 0

    # First pass: Identify all unique features and classes
    for node in json_tree:
        if node is None:
            continue
        if node.get("role") == "C" and node.get("triples"):
            feature_str = node["triples"][0]
            if feature_str not in feature_names_map:
                feature_names_map[feature_str] = feature_count
                feature_count += 1
        elif node.get("role") == "D":
            # Sort and join triples to form a unique class label string
            label_str = " ".join(sorted(node.get("triples", [])))
            if not label_str: # Handle empty decision node as a specific class
                label_str = "__EMPTY__"
            if label_str not in classes_map:
                classes_map[label_str] = class_count
                class_count += 1

    feature_names = list(feature_names_map.keys())
    classes_ = list(classes_map.keys())

    # Find leaf nodes ('D' nodes)
    leaf_indices = [
        i for i, node in enumerate(json_tree)
        if node is not None and node.get("role") == "D"
    ]

    branches = []

    # For each leaf node
    for leaf_idx in leaf_indices:
        leaf_node = json_tree[leaf_idx]

        # Determine class label and create probability vector
        label_str = " ".join(sorted(leaf_node.get("triples", [])))
        if not label_str:
             label_str = "__EMPTY__"
        class_idx = classes_map[label_str]
        probas = np.zeros(len(classes_))
        probas[class_idx] = 1.0

        # Create branch
        branch = SimpleBranch(
            feature_names=feature_names,
            classes_=classes_,
            label_probas=list(probas),
            number_of_samples=1  # Default number of samples
        )

        # Trace path from leaf to root
        node_id = leaf_idx
        while node_id > 0:
            parent_id = (node_id - 1) // 2
            if parent_id < 0 or parent_id >= len(json_tree) or json_tree[parent_id] is None:
                 print(f"Warning: Invalid parent index {parent_id} found while tracing from node {node_id}. Stopping trace.")
                 break # Stop tracing if parent is invalid

            parent_node = json_tree[parent_id]

            # Expect parent to be a 'C' node for branching
            if parent_node.get("role") != "C" or not parent_node.get("triples"):
                 print(f"Warning: Expected 'C' node with triples at index {parent_id}, but found {parent_node}. Stopping trace.")
                 break # Stop if parent isn't a valid condition node

            feature_str = parent_node["triples"][0]
            if feature_str not in feature_names_map:
                 print(f"Warning: Feature '{feature_str}' from parent {parent_id} not found in map. Skipping condition.")
                 node_id = parent_id # Move to parent but skip adding condition
                 continue

            feature_idx = feature_names_map[feature_str]

            # Determine if current node is left or right child
            is_left_child = (node_id == 2 * parent_id + 1)

            # Add condition (0.5 threshold for binary features)
            threshold = 0.5
            if is_left_child:
                bound = "upper"  # Left child means feature <= 0.5 (absent)
            else:
                bound = "lower"  # Right child means feature > 0.5 (present)

            branch.add_condition(feature_idx, threshold, bound)

            # Move to parent
            node_id = parent_id

        branches.append(branch)

    return branches, feature_names, classes_

# Load the trees from the JSON file
# Use the path provided in the context
json_file_path = 'generated_subtrees.json'
try:
    with open(json_file_path, 'r') as f:
        all_json_trees = json.load(f)
except FileNotFoundError:
    print(f"Error: JSON file not found at {json_file_path}")
    print("Please ensure the path is correct relative to where the script is run.")
    exit()

# Check if there are enough trees to compare
if len(all_json_trees) < 2:
    print("Need at least two trees in the JSON file to compare.")
    exit()

# Process trees and store results
processed_trees_data = []
for i, json_tree in enumerate(all_json_trees):
    print(f"Processing tree {i+1}...", end="")
    branches, feature_names, classes_ = extract_rules_from_json_tree(json_tree)
    
    if not branches:
        print(f" Skipped (No branches extracted).")
        processed_trees_data.append(None) # Placeholder for skipped trees
        continue
        
    print(f" Found {len(feature_names)} features, {len(classes_)} classes, {len(branches)} branches.")
    
    # Convert branches to DataFrame
    cs = SimpleConjunctionSet(feature_names=feature_names, amount_of_branches_threshold=np.inf)
    # aggregate_branches expects a list of lists of branches
    cs.aggregate_branches([[branch for branch in branches]], classes_)
    cs.buildConjunctionSet() # This step might not be strictly necessary if aggregate does the job
    branches_df = cs.get_conjunction_set_df()

    processed_trees_data.append({
        'branches': branches,
        'feature_names': feature_names,
        'classes': classes_,
        'df': branches_df
    })

    # Example: Print the first few branches for the first valid tree processed
    if i == 0 and not branches_df.empty:
         print("\n  Example Branches DataFrame for Tree 1:")
         print(branches_df.head(3).to_string())
         print("  Feature Names:", feature_names)
         print("  Class Names:", classes_)

# --- Compare two trees (e.g., the first two valid ones found) ---

tree_data_1 = None
tree_data_2 = None
idx1, idx2 = -1, -1

# Find the first valid processed tree
for i, data in enumerate(processed_trees_data):
    if data is not None and not data['df'].empty:
        tree_data_1 = data
        idx1 = i
        break

# Find the second valid processed tree
for i, data in enumerate(processed_trees_data[idx1+1:], start=idx1+1):
    if data is not None and not data['df'].empty:
        tree_data_2 = data
        idx2 = i
        break

if tree_data_1 and tree_data_2:
    print(f"\nComparing Tree {idx1 + 1} and Tree {idx2 + 1}...")

    # Ensure features and classes are compatible or find common subset
    # For simplicity, we assume they *should* be based on the extraction method
    # A robust implementation might involve aligning features/classes
    if tree_data_1['feature_names'] != tree_data_2['feature_names']:
        print("Warning: Feature names differ between trees. Comparison might be inaccurate.")
        # Potential alignment logic here...
    if tree_data_1['classes'] != tree_data_2['classes']:
        print("Warning: Class names differ between trees. Comparison might be inaccurate.")
        # Potential alignment logic here...

    # Use feature_names and classes_ from the first tree for consistency
    # (Assuming they should ideally be the same across compared trees)
    feature_names_for_comparison = tree_data_1['feature_names']
    classes_for_comparison = tree_data_1['classes']

    # Define bounds (defaulting to [0, 1] as compare_trees handles None)
    bounds_for_comparison = [[0, 1] for _ in range(len(feature_names_for_comparison))] # Let compare_trees use default [0, 1]

    tree_similarity = compare_trees(
        tree1=tree_data_1['df'],
        tree2=tree_data_2['df'],
        feature_names=feature_names_for_comparison,
        classes_=classes_for_comparison,
        bounds=bounds_for_comparison
    )
    # ------------------------------------

    print(f"\nStructural Similarity between Tree {idx1 + 1} Branch 0 and Tree {idx2 + 1} Branch 0: {tree_similarity:.4f}")


else:
    print("\nCould not find two valid trees with extracted branches to compare.")

