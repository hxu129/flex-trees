import json
import numpy as np
import os
import pickle

def save_mappings(feature_map, class_map, output_dir='.'):
    """保存特征映射和类别映射到文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'feature_map.pkl'), 'wb') as f:
        pickle.dump(feature_map, f)
        
    with open(os.path.join(output_dir, 'class_map.pkl'), 'wb') as f:
        pickle.dump(class_map, f)
    
    # 同时保存一个人类可读的JSON版本
    with open(os.path.join(output_dir, 'feature_map.json'), 'w') as f:
        json.dump({k: int(v) for k, v in feature_map.items()}, f, indent=2)
        
    with open(os.path.join(output_dir, 'class_map.json'), 'w') as f:
        json.dump({k: int(v) for k, v in class_map.items()}, f, indent=2)
    
def load_mappings(input_dir='.'):
    """从文件加载特征映射和类别映射"""
    feature_map_path = os.path.join(input_dir, 'feature_map.pkl')
    class_map_path = os.path.join(input_dir, 'class_map.pkl')
    
    if not os.path.exists(feature_map_path) or not os.path.exists(class_map_path):
        raise FileNotFoundError("特征映射或类别映射文件不存在")
    
    with open(feature_map_path, 'rb') as f:
        feature_map = pickle.load(f)
        
    with open(class_map_path, 'rb') as f:
        class_map = pickle.load(f)
        
    return feature_map, class_map

def process_json_trees(json_file_path, save_maps=True, output_dir='.'):
    """
    Process JSON trees into a format compatible with gfn_trees.compare_trees
    
    Args:
        json_file_path (str): Path to JSON file containing trees
        save_maps (bool): Whether to save feature and class mappings to files
        output_dir (str): Directory to save mappings
        
    Returns:
        tuple: A tuple containing:
            - list: List of encoded trees, each tree is a list of node lists
            - list: Feature names
            - list: Class names
    """
    # Load JSON trees
    try:
        with open(json_file_path, 'r') as f:
            all_json_trees = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}")
        return [], [], []
    
    # Check if there are trees to process
    if not all_json_trees:
        print("No trees found in the JSON file.")
        return [], [], []
    
    # First pass: Collect all unique feature names across all trees
    all_features = set()
    all_classes = set()
    
    for json_tree in all_json_trees:
        if not json_tree:
            continue
            
        for node in json_tree:
            if node is None:
                continue
                
            # Collect feature names from condition nodes
            if node.get("role") == "C" and node.get("triples"):
                for feature_str in node.get("triples", []):
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
    
    # Save mappings if requested
    if save_maps:
        save_mappings(feature_map, class_map, output_dir)
    
    # Process each tree
    processed_trees = []
    
    for tree_idx, json_tree in enumerate(all_json_trees):
        if not json_tree:
            # Empty placeholder for empty trees
            processed_trees.append([])
            continue
            
        # Initialize tree with empty nodes
        # Find the maximum node index in the tree
        max_node_idx = max(enumerate(json_tree), key=lambda x: x[0] if x[1] is not None else 0)[0]
        
        # Create node list with correct size for number of classes
        # 5 fixed elements + class probabilities
        tree = [[np.nan, np.nan, np.nan, -1, 0] + [np.nan] * len(classes_) for _ in range(max_node_idx + 1)]
        
        # Fill in the tree nodes
        for idx, node in enumerate(json_tree):
            if node is None:
                continue
                
            if idx >= len(tree):
                # Extend tree if needed (shouldn't happen if max_node_idx calculated correctly)
                tree.extend([[np.nan, np.nan, np.nan, -1, 0] + [np.nan] * len(classes_) 
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
    print(f"Unified feature set ({len(feature_names)}): {feature_names[:5]}...")
    print(f"Unified class set ({len(classes_)}): {classes_}")
    
    return processed_trees, feature_names, classes_

def create_similarity_calculator(json_file_path, save_maps=True, output_dir='.'):
    """
    创建一个函数，用于计算输入树与JSON文件中所有树的平均相似度
    
    Args:
        json_file_path (str): JSON文件路径，包含要比较的树
        save_maps (bool): 是否保存特征和类别映射到文件
        output_dir (str): 保存映射的目录
        
    Returns:
        function: 一个函数，接受一个树作为输入，返回其与JSON文件中所有树的平均相似度
    """
    # 预处理JSON文件中的树
    processed_trees, feature_names, classes_ = process_json_trees(json_file_path, save_maps, output_dir)
    
    # 过滤掉空树
    valid_trees = [tree for tree in processed_trees if tree]
    
    if not valid_trees:
        print("Warning: No valid trees found in JSON file.")
        
    print(f"Loaded {len(valid_trees)} valid trees for similarity calculation.")
    
    # 创建比较函数
    def calculate_average_similarity(input_tree, comp_dist=True, dist_weight=0.5):
        """
        计算输入树与JSON文件中所有树的平均相似度
        
        Args:
            input_tree (list): 输入树，格式与gfn_trees.py中的tree1/tree2相同
            comp_dist (bool): 是否比较标签分布
            dist_weight (float): 标签分布差异的权重
            
        Returns:
            float: 平均相似度
        """
        from gfn_trees import compare_trees
        
        if not valid_trees:
            print("No valid trees to compare with.")
            return 0.0
            
        # 定义特征边界（默认为0-1）
        bounds = [(0, 1) for _ in range(len(feature_names))]
        
        # 计算输入树与每棵树的相似度
        similarities = []
        for tree in valid_trees:
            similarity = compare_trees(
                tree1=input_tree,
                tree2=tree,
                feature_names=feature_names,
                classes_=classes_,
                bounds=bounds,
                comp_dist=comp_dist,
                dist_weight=dist_weight
            )
            similarities.append(similarity)
            
        # 计算平均相似度
        average_similarity = sum(similarities) / len(similarities)
        
        return average_similarity
    
    return calculate_average_similarity

if __name__ == "__main__":
    json_file_path = 'generated_subtrees.json'
    
    # 示例1：直接处理并比较两棵树
    trees, features, classes = process_json_trees(json_file_path, save_maps=True)
    
    # 比较第一棵和第二棵树（如果存在）
    if len(trees) >= 2 and trees[0] and trees[1]:
        from gfn_trees import compare_trees
        
        bounds = [(0, 1) for _ in range(len(features))]
        similarity = compare_trees(
            trees[0], trees[1], 
            features, classes,
            bounds, comp_dist=True, dist_weight=0.5
        )
        
        print(f"树节点数量: {len(trees[0])} 类别数量: {len(classes)}")
        print(f"\n第1棵和第2棵树的结构相似度: {similarity:.4f}")
    
    # 示例2：创建相似度计算器函数并使用
    print("\n创建相似度计算器...")
    similarity_calculator = create_similarity_calculator(json_file_path)
    
    # 如果有有效的树，用第一棵树测试计算器
    if trees and trees[0]:
        avg_similarity = similarity_calculator(trees[0])
        print(f"第1棵树与所有树的平均相似度: {avg_similarity:.4f}")

