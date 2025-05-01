import json
import numpy as np

def process_json_trees(json_file_path):
    """
    Process JSON trees into a format compatible with gfn_trees.compare_trees
    
    Args:
        json_file_path (str): Path to JSON file containing trees
        
    Returns:
        list: List of encoded trees, each tree is a list of node lists
    """
    # Load JSON trees
    try:
        with open(json_file_path, 'r') as f:
            all_json_trees = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}")
        return []
    
    # Check if there are trees to process
    if not all_json_trees:
        print("No trees found in the JSON file.")
        return []
    
    # First pass: Collect all unique feature names across all trees
    all_features = set()
    all_classes = set()
    
    for json_tree in all_json_trees:
        if not json_tree:
            continue
            
        for node in json_tree:
            if node is None:
                continue
                
            # FIXME num of ele
            # Collect feature names from condition nodes
            if node.get("role") == "C" and node.get("triples"):
                feature_str = node["triples"][0]
                all_features.add(feature_str)
                
            # Collect class names from decision nodes
            elif node.get("role") == "D":
                label_str = " ".join(sorted(node.get("triples", [])))
                if not label_str:
                    label_str = "__EMPTY__"
                all_classes.add(label_str)
    
    # Create unified feature and class mappings
    feature_names = sorted(list(all_features))
    classes_ = sorted(list(all_classes))
    
    feature_map = {feature: idx for idx, feature in enumerate(feature_names)}
    class_map = {cls: idx for idx, cls in enumerate(classes_)}
    
    # Process each tree
    processed_trees = []
    
    for tree_idx, json_tree in enumerate(all_json_trees):
        if not json_tree:
            # Empty placeholder for empty trees
            processed_trees.append([])
            continue
            
        # Initialize tree with empty nodes
        # FIXME 没看懂
        max_node_idx = max(enumerate(json_tree), key=lambda x: 0 if x[0] is None else x[0])[0]
        # FIXME num of classes
        tree = [[np.nan, np.nan, np.nan, -1, 0, np.nan, np.nan, np.nan] for _ in range(max_node_idx + 1)]
        
        # Fill in the tree nodes
        for idx, node in enumerate(json_tree):
            if node is None:
                continue
                
            if idx >= len(tree):
                # Extend tree if needed (shouldn't happen if max_node_idx calculated correctly)
                tree.extend([[np.nan, np.nan, np.nan, -1, 0, np.nan, np.nan, np.nan] 
                           for _ in range(idx - len(tree) + 1)])
            
            # Process condition node
            if node.get("role") == "C" and node.get("triples"):
                feature_str = node["triples"][0]
                if feature_str in feature_map:
                    feature_idx = feature_map[feature_str]
                    tree[idx][0] = 0  # Condition node
                    tree[idx][1] = feature_idx
                    tree[idx][2] = 0.5  # Binary threshold
                    tree[idx][3] = -1
                    tree[idx][4] = 0
            
            # Process decision/leaf node
            elif node.get("role") == "D":
                label_str = " ".join(sorted(node.get("triples", [])))
                if not label_str:
                    label_str = "__EMPTY__"
                    
                class_idx = class_map[label_str]
                
                # Create probability vector (one-hot)
                probas = [0.0] * len(classes_)
                probas[class_idx] = 1.0
                
                tree[idx][0] = 1  # Leaf node
                tree[idx][1] = -1
                tree[idx][2] = -1
                tree[idx][3] = -1
                tree[idx][4] = 0
                
                # Fill in probabilities (indices 5 onwards)
                for i, prob in enumerate(probas):
                    if 5 + i < len(tree[idx]):
                        tree[idx][5 + i] = prob
                    else:
                        tree[idx].append(prob)
        
        processed_trees.append(tree)
    
    print(f"Processed {len(processed_trees)} trees.")
    print(f"Unified feature set: {feature_names}")
    print(f"Unified class set: {classes_}")
    
    return processed_trees, feature_names, classes_

if __name__ == "__main__":
    json_file_path = 'generated_subtrees.json'
    trees, features, classes = process_json_trees(json_file_path)
    
    # Compare first two trees if available
    if len(trees) >= 2 and trees[0] and trees[1]:
        from gfn_trees import compare_trees
        
        bounds = [(0, 1) for _ in range(len(features))]
        similarity = compare_trees(
            trees[0], trees[1], 
            features, classes,
            bounds, comp_dist=True, dist_weight=0.5
        )
        
        print(trees[0])
        print(trees[1])
        print(f"\nStructural Similarity between Tree 1 and Tree 2: {similarity:.4f}")

