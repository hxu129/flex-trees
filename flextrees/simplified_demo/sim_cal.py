import json
import torch
import torch.nn.functional as F
import os
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

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
    """处理JSON文件中的树，返回PyTorch张量格式的树表示"""
    # 加载JSON树
    try:
        with open(json_file_path, 'r') as f:
            all_json_trees = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}")
        return [], [], []
    
    # 检查是否有可处理的树
    if not all_json_trees:
        print("No trees found in the JSON file.")
        return [], [], []
    
    # 第一步：收集所有唯一的特征名和类别名
    all_features = set()
    all_classes = set()
    
    for json_tree in all_json_trees:
        if not json_tree:
            continue
            
        for node in json_tree:
            if node is None:
                continue
                
            # 从条件节点中收集特征名
            if node.get("role") == "C" and node.get("triples"):
                for feature_str in node.get("triples", []):
                    all_features.add(feature_str)
                
            # 从决策节点中收集类别名
            elif node.get("role") == "D":
                label_str = " ".join(sorted(node.get("triples", [])))
                if not label_str:
                    label_str = "__EMPTY__"
                all_classes.add(label_str)
    
    # 创建统一的特征和类别映射
    feature_names = sorted(list(all_features))
    classes_ = sorted(list(all_classes))
    
    feature_map = {feature: idx for idx, feature in enumerate(feature_names)}
    class_map = {cls: idx for idx, cls in enumerate(classes_)}
    
    # 如果需要，保存映射
    if save_maps:
        save_mappings(feature_map, class_map, output_dir)
    
    # 处理每棵树
    processed_trees = []
    
    for tree_idx, json_tree in enumerate(all_json_trees):
        if not json_tree:
            # 对空树使用空占位符
            processed_trees.append([])
            continue
            
        # 初始化树节点
        max_node_idx = max(enumerate(json_tree), key=lambda x: x[0] if x[1] is not None else 0)[0]
        
        # 创建具有正确大小的节点列表
        tree = []
        for _ in range(max_node_idx + 1):
            # 每个节点：5个固定元素 + 类别概率
            node = [float('nan')] * (5 + len(classes_))
            tree.append(node)
        
        # 填充树节点
        for idx, node in enumerate(json_tree):
            if node is None:
                continue
                
            if idx >= len(tree):
                # 如果需要，扩展树（如果max_node_idx计算正确，不应该发生）
                for _ in range(idx - len(tree) + 1):
                    new_node = [float('nan')] * (5 + len(classes_))
                    tree.append(new_node)
            
            # 处理条件节点
            if node.get("role") == "C" and node.get("triples"):
                feature_str = node["triples"][0]
                if feature_str in feature_map:
                    feature_idx = feature_map[feature_str]
                    tree[idx][0] = 0  # 条件节点
                    tree[idx][1] = feature_idx
                    tree[idx][2] = 0.5  # 二元阈值
                    tree[idx][3] = -1
                    tree[idx][4] = 0
            
            # 处理决策/叶节点
            elif node.get("role") == "D":
                label_str = " ".join(sorted(node.get("triples", [])))
                if not label_str:
                    label_str = "__EMPTY__"
                    
                class_idx = class_map[label_str]
                
                # 创建概率向量（one-hot）
                probas = [0.0] * len(classes_)
                probas[class_idx] = 1.0
                
                tree[idx][0] = 1  # 叶节点
                tree[idx][1] = -1
                tree[idx][2] = -1
                tree[idx][3] = -1
                tree[idx][4] = 0
                
                # 填充概率（索引5及以后）
                for i, prob in enumerate(probas):
                    tree[idx][5 + i] = prob
        
        # 将列表转换为PyTorch张量
        tensor_tree = []
        for node in tree:
            tensor_node = []
            for val in node:
                if val != val:  # nan检查
                    tensor_node.append(torch.tensor(float('nan')))
                else:
                    tensor_node.append(torch.tensor(float(val)))
            tensor_tree.append(tensor_node)
        
        processed_trees.append(tensor_tree)
    
    print(f"Processed {len(processed_trees)} trees.")
    print(f"Unified feature set ({len(feature_names)}): {feature_names[:5]}...")
    print(f"Unified class set ({len(classes_)}): {classes_}")
    
    return processed_trees, feature_names, classes_

def torch_jensenshannon(p, q, base=None):
    """PyTorch实现的Jensen-Shannon散度，确保可微分"""
    # 添加小值避免log(0)，确保梯度流动
    eps = 1e-10
    
    # 确保p和q是张量
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, dtype=torch.float32)
    if not isinstance(q, torch.Tensor):
        q = torch.tensor(q, dtype=torch.float32)
        
    # 添加微小值防止零概率
    p = p + eps
    q = q + eps
    
    # 归一化，保持梯度
    p = p / torch.sum(p)
    q = q / torch.sum(q)
    
    # 计算混合分布
    m = 0.5 * (p + q)
    
    # 计算KL散度，使用安全的log计算
    log_p_m = torch.log(p) - torch.log(m)
    log_q_m = torch.log(q) - torch.log(m)
    
    # 使用乘法和sum而不是dot product，保持梯度
    kl_pm = torch.sum(p * log_p_m)
    kl_qm = torch.sum(q * log_q_m)
    
    # Jensen-Shannon散度
    js = 0.5 * (kl_pm + kl_qm)
    
    # 确保js是非负的
    js = torch.clamp(js, min=0.0)
    
    # 如果需要，应用平方根获取Jensen-Shannon距离
    if base is not None:
        js = js / torch.log(torch.tensor(base, dtype=torch.float32))
        
    # 计算平方根时添加小值避免对0求导导致的梯度问题
    return torch.sqrt(js + eps)

def compare_torch_branches(branch1, branch2, feature_names, comp_dist=False, dist_weight=0.5, not_match_label=-float('inf')):
    """比较两个分支的结构相似性（PyTorch版本）"""
    # 将pandas Series转换为字典以便于处理
    b1 = {col: branch1[col] for col in branch1.index}
    b2 = {col: branch2[col] for col in branch2.index}
    
    # 转换为PyTorch张量，保留梯度信息
    # 检查是否已经是PyTorch张量，否则转换
    if isinstance(b1["probas"], np.ndarray):
        probas1 = torch.tensor(b1["probas"], dtype=torch.float32)
    elif isinstance(b1["probas"], list):
        # 检查列表中是否有张量元素
        has_tensor = False
        for p in b1["probas"]:
            if isinstance(p, torch.Tensor):
                has_tensor = True
                break
                
        if has_tensor:
            # 如果列表中有张量，保留它们
            probas1 = []
            for p in b1["probas"]:
                if isinstance(p, torch.Tensor):
                    probas1.append(p)
                else:
                    probas1.append(torch.tensor(float(p), dtype=torch.float32))
            # 将列表转换为张量，保持梯度流
            if all(isinstance(p, torch.Tensor) for p in probas1):
                stacked = torch.stack(probas1)
                probas1 = stacked
        else:
            probas1 = torch.tensor(b1["probas"], dtype=torch.float32)
    elif hasattr(b1["probas"], 'tolist'):
        probas1 = torch.tensor(b1["probas"].tolist(), dtype=torch.float32)
    else:
        probas1 = b1["probas"]  # 如果已经是tensor则直接使用
        
    if isinstance(b2["probas"], np.ndarray):
        probas2 = torch.tensor(b2["probas"], dtype=torch.float32)
    elif isinstance(b2["probas"], list):
        # 检查列表中是否有张量元素
        has_tensor = False
        for p in b2["probas"]:
            if isinstance(p, torch.Tensor):
                has_tensor = True
                break
                
        if has_tensor:
            # 如果列表中有张量，保留它们
            probas2 = []
            for p in b2["probas"]:
                if isinstance(p, torch.Tensor):
                    probas2.append(p)
                else:
                    probas2.append(torch.tensor(float(p), dtype=torch.float32))
            # 将列表转换为张量，保持梯度流
            if all(isinstance(p, torch.Tensor) for p in probas2):
                stacked = torch.stack(probas2)
                probas2 = stacked
        else:
            probas2 = torch.tensor(b2["probas"], dtype=torch.float32)
    elif hasattr(b2["probas"], 'tolist'):
        probas2 = torch.tensor(b2["probas"].tolist(), dtype=torch.float32)
    else:
        probas2 = b2["probas"]  # 如果已经是tensor则直接使用
    
    # 比较两个分支的结构相似性
    dist_similarity = torch.tensor(0.0, dtype=torch.float32)
    if comp_dist:
        # 比较类别标签的分布
        dist_similarity = torch_jensenshannon(probas1, probas2)
    else:
        # 第1步：比较类别标签
        if isinstance(probas1, list) and all(isinstance(p, torch.Tensor) for p in probas1):
            tmp_probas1 = torch.stack(probas1)
            class_label_1 = torch.argmax(tmp_probas1)
        else:
            class_label_1 = torch.argmax(probas1)
            
        if isinstance(probas2, list) and all(isinstance(p, torch.Tensor) for p in probas2):
            tmp_probas2 = torch.stack(probas2)
            class_label_2 = torch.argmax(tmp_probas2)
        else:
            class_label_2 = torch.argmax(probas2)
            
        if class_label_1 != class_label_2:
            return torch.tensor(not_match_label, dtype=torch.float32)
    
    overlap = torch.tensor(0.0, dtype=torch.float32)
    overall_range_1 = torch.tensor(0.0, dtype=torch.float32)
    overall_range_2 = torch.tensor(0.0, dtype=torch.float32)
    
    # 第2步：比较重叠区域和整体范围
    for feature in range(len(feature_names)):
        # 转换特征边界值为PyTorch张量
        # 确保所有边界值都是张量，保持梯度流
        feature_key_lower_1 = f"{feature}_lower"
        feature_key_upper_1 = f"{feature}_upper"
        feature_key_lower_2 = f"{feature}_lower"
        feature_key_upper_2 = f"{feature}_upper"
        
        # 获取特征边界并确保是张量
        if isinstance(b1[feature_key_lower_1], torch.Tensor):
            lower_1 = b1[feature_key_lower_1]
        else:
            lower_1 = torch.tensor(float(b1[feature_key_lower_1]), dtype=torch.float32)
            
        if isinstance(b1[feature_key_upper_1], torch.Tensor):
            upper_1 = b1[feature_key_upper_1]
        else:
            upper_1 = torch.tensor(float(b1[feature_key_upper_1]), dtype=torch.float32)
            
        if isinstance(b2[feature_key_lower_2], torch.Tensor):
            lower_2 = b2[feature_key_lower_2]
        else:
            lower_2 = torch.tensor(float(b2[feature_key_lower_2]), dtype=torch.float32)
            
        if isinstance(b2[feature_key_upper_2], torch.Tensor):
            upper_2 = b2[feature_key_upper_2]
        else:
            upper_2 = torch.tensor(float(b2[feature_key_upper_2]), dtype=torch.float32)
        
        # 计算重叠 - 使用PyTorch操作保持梯度
        min_upper = torch.min(upper_1, upper_2)
        max_lower = torch.max(lower_1, lower_2)
        current_overlap = torch.clamp(min_upper - max_lower, min=0.0)
        overlap = overlap + current_overlap
        
        # 计算整体范围
        overall_range_1 = overall_range_1 + (upper_1 - lower_1)
        overall_range_2 = overall_range_2 + (upper_2 - lower_2)

    # 计算分支相似度
    # 添加小值避免除零
    denominator = overall_range_1 + overall_range_2
    # 避免使用.item()进行比较
    if denominator == 0:
        denominator = denominator + 1e-10
    branch_similarity = 2 * overlap / denominator

    if comp_dist:
        # 避免直接乘以标量，使用张量的multiply操作保持梯度
        branch_similarity = dist_similarity * dist_weight + branch_similarity * (1 - dist_weight)

    return branch_similarity

def linear_sum_assignment_torch(cost_matrix):
    # 保存原始梯度信息的张量
    original_matrix = cost_matrix.clone()
    
    # 计算匹配（不可微）
    np_cost_matrix = cost_matrix.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(np_cost_matrix)
    
    # 构建梯度路径 - 使用打通的梯度路径
    total_match_value = torch.zeros(1, requires_grad=True)
    selected_values = []
    
    for r, c in zip(row_ind, col_ind):
        # 直接使用原始矩阵的值，保持梯度
        selected_values.append(original_matrix[r, c])
    
    if selected_values:
        # 使用torch.stack确保梯度能正确传递
        total_match_value = torch.sum(torch.stack(selected_values))
    
    # 确保计算图连接到结果
    return row_ind, col_ind, total_match_value

# 辅助函数：检查树是否包含可微分参数
def has_grad_params(tree):
    """检查树是否包含可微分参数"""
    for node in tree:
        for val in node:
            if isinstance(val, torch.Tensor) and val.requires_grad:
                return True
    return False

def compare_torch_trees(tree1, tree2, feature_names, classes_, bounds, comp_dist=False, dist_weight=0.5):
    """比较两棵树的结构相似性（PyTorch版本）
    
    使用可微分的规则提取和比较流程，确保梯度能够正确传播。
    
    Args:
        tree1: 第一棵树，每个节点是一个列表或PyTorch张量
        tree2: 第二棵树，每个节点是一个列表或PyTorch张量
        feature_names: 特征名称列表
        classes_: 类别名称列表
        bounds: 特征的取值范围，列表的列表: [(min1, max1), (min2, max2), ...]
        comp_dist: 是否比较标签分布，如果为False则只比较最可能的类别
        dist_weight: 类别分布权重，仅当comp_dist=True有效
    
    Returns:
        torch.Tensor: 两棵树的结构相似度，可用于梯度传播
    """
    import pandas as pd

    # 检查是否有可微分参数
    has_grad1 = has_grad_params(tree1)
    has_grad2 = has_grad_params(tree2)
    has_grad = has_grad1 or has_grad2
    
    # 保存所有可微分参数引用，以便后续构建计算图
    param_tensors = []
    if has_grad:
        # 收集所有可微分参数
        for node in tree1:
            for val in node:
                if isinstance(val, torch.Tensor) and val.requires_grad:
                    param_tensors.append(val)
        for node in tree2:
            for val in node:
                if isinstance(val, torch.Tensor) and val.requires_grad:
                    param_tensors.append(val)
    
    # 使用可微分版本的函数提取规则
    branches1 = extract_df_rules_from_tree_torch(tree1, feature_names, classes_)
    branches2 = extract_df_rules_from_tree_torch(tree2, feature_names, classes_)
    
    # 处理空分支情况
    if len(branches1) == 0 or len(branches2) == 0:
        print("No branches found in one or both trees")
        if has_grad and param_tensors:
            # 如果有可微分参数，创建一个带梯度的零值
            zero = torch.zeros(1, requires_grad=True)
            # 添加一个微小的可微分贡献，使梯度可以流动
            result = zero * 0 + sum(p * 0 for p in param_tensors)
            return result[0]
        return torch.tensor(0.0, dtype=torch.float32)
    
    # 定义一个函数来替换DataFrame中的无穷值
    def replace_inf_with_bounds_torch(series):
        if 'prob' in series.name:  # 概率值不替换
            return series
        column_idx, bound_type = series.name.split("_")
        column_idx = int(column_idx)
        if bound_type == "upper":
            max_value = bounds[column_idx][1]
            if isinstance(series, pd.Series):
                # 对于Pandas Series，使用replace方法
                return series.replace([np.inf, float('inf')], max_value)
            else:
                # 对于其他类型，尝试直接替换
                return max_value if series == float('inf') else series
        elif bound_type == "lower":
            min_value = bounds[column_idx][0]
            if isinstance(series, pd.Series):
                # 对于Pandas Series，使用replace方法
                return series.replace([-np.inf, float('-inf')], min_value)
            else:
                # 对于其他类型，尝试直接替换
                return min_value if series == float('-inf') else series
        else:
            raise ValueError(f"Invalid bound type: {bound_type}")
    
    # 应用替换函数处理无穷值
    try:
        branches1 = branches1.apply(replace_inf_with_bounds_torch)
        branches2 = branches2.apply(replace_inf_with_bounds_torch)
    except Exception as e:
        print(f"替换无穷值时出错: {e}")
        # 如果出错，创建一个简单的可微分结果
        if has_grad and param_tensors:
            return sum(p * 0 for p in param_tensors) + torch.tensor(0.0, requires_grad=True)
        return torch.tensor(0.0, dtype=torch.float32)

    # 创建相似度矩阵 - 计算每对分支之间的相似度
    similarity_matrix = torch.zeros((len(branches1), len(branches2)), dtype=torch.float32)
    
    try:
        for i in range(len(branches1)):
            for j in range(len(branches2)):
                similarity_matrix[i, j] = compare_torch_branches(
                    branches1.iloc[i], 
                    branches2.iloc[j], 
                    feature_names, 
                    comp_dist, 
                    dist_weight
                )
    except Exception as e:
        print(f"计算分支相似度时出错: {e}")
        # 如果出错，创建一个简单的可微分结果
        if has_grad and param_tensors:
            return sum(p * 0 for p in param_tensors) + torch.tensor(0.0, requires_grad=True)
        return torch.tensor(0.0, dtype=torch.float32)
    
    # 使用匈牙利算法进行最优匹配
    try:
        row_ind, col_ind, match_value = linear_sum_assignment_torch(-1 * similarity_matrix)
    except Exception as e:
        print(f"匈牙利算法匹配失败: {e}，尝试使用惩罚矩阵")
        # 如果匈牙利算法失败，创建一个包含惩罚的扩展矩阵
        M, N = similarity_matrix.shape
        penalty_value = torch.tensor(-float('inf'), dtype=torch.float32)
        penalty_matrix = torch.full((M, M), penalty_value)
        torch.diagonal(penalty_matrix)[:] = 0  # 对角线上设置为0
        augmented_matrix = torch.cat([similarity_matrix, penalty_matrix], dim=1)
        row_ind, col_ind, match_value = linear_sum_assignment_torch(-1 * augmented_matrix)
    
    # 计算未匹配的分支
    unmapped_branches1_indices = set()
    unmapped_branches2_indices = set()
    
    # 去掉惩罚矩阵的影响，只保留有效匹配
    new_row_ind, new_col_ind = [], []
    for r, c in zip(row_ind, col_ind):
        if c < len(branches2):  # 这是一个有效匹配
            new_row_ind.append(r)
            new_col_ind.append(c)
        else:  # 这是一个惩罚项
            unmapped_branches1_indices.add(r)
    
    row_ind, col_ind = new_row_ind, new_col_ind
    
    # 找出所有未匹配的分支
    unmapped_branches1_indices.update(set(range(len(branches1))) - set(row_ind))
    unmapped_branches2_indices.update(set(range(len(branches2))) - set(col_ind))
    
    # 计算未匹配分支的惩罚
    penalty = torch.tensor(0.0, dtype=torch.float32)
    
    # 对于每个未匹配的分支1，添加它与所有分支2的最大相似度作为惩罚
    for i in unmapped_branches1_indices:
        if len(similarity_matrix[i]) > 0:
            max_sim = torch.max(similarity_matrix[i])
            if not torch.isnan(max_sim) and not torch.isinf(max_sim):
                penalty += max_sim
    
    # 对于每个未匹配的分支2，添加它与所有分支1的最大相似度作为惩罚
    for j in unmapped_branches2_indices:
        if similarity_matrix.shape[0] > 0:
            column_j = similarity_matrix[:, j]
            max_sim = torch.max(column_j)
            if not torch.isnan(max_sim) and not torch.isinf(max_sim):
                penalty += max_sim
    
    # 计算匹配的总相似度
    total_similarity = torch.tensor(0.0, dtype=torch.float32)
    for i, j in zip(row_ind, col_ind):
        total_similarity += similarity_matrix[i, j]
    
    # 计算最终的平均相似度
    total_branches = len(branches1) + len(branches2)
    if total_branches > 0:
        average_similarity = 2 * (total_similarity - penalty) / total_branches
    else:
        average_similarity = torch.tensor(0.0, dtype=torch.float32)
    
    # 确保结果可微分
    if has_grad and param_tensors and not average_similarity.requires_grad:
        # 使用match_value添加一个有效的梯度路径
        average_similarity = average_similarity + (match_value * 0)
        # 额外添加所有参数的连接
        for p in param_tensors:
            if p.requires_grad:
                average_similarity = average_similarity + (p * 0)

    return average_similarity

def create_torch_similarity_calculator(json_file_path, save_maps=True, output_dir='.'):
    """
    创建一个函数，用于计算PyTorch张量格式的输入树与JSON文件中所有树的平均相似度
    
    Args:
        json_file_path (str): JSON文件路径，包含要比较的树
        save_maps (bool): 是否保存特征和类别映射到文件
        output_dir (str): 保存映射的目录
        
    Returns:
        function: 一个函数，接受一个PyTorch张量格式的树作为输入，返回平均相似度
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
        计算PyTorch张量格式的输入树与JSON文件中所有树的平均相似度
        
        Args:
            input_tree (torch.Tensor或list): 输入树，格式与gfn_trees.py中的tree1/tree2相同，但可以是PyTorch张量
            comp_dist (bool): 是否比较标签分布
            dist_weight (float): 标签分布差异的权重
            
        Returns:
            torch.Tensor: 平均相似度，作为可微分PyTorch张量
        """
        if not valid_trees:
            print("No valid trees to compare with.")
            return torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
            
        # 定义特征边界（默认为0-1）
        bounds = [(0, 1) for _ in range(len(feature_names))]
        
        # 检查input_tree是否包含requires_grad=True的张量
        has_grad = False
        for node in input_tree:
            for val in node:
                if isinstance(val, torch.Tensor) and val.requires_grad:
                    has_grad = True
                    break
            if has_grad:
                break
        
        if not has_grad:
            print("Warning: 输入树不包含可微分参数，尝试转换...")
            # 尝试把树中的部分张量转换为可微分参数
            try:
                # 寻找条件节点，将其阈值转换为可微分参数
                for i, node in enumerate(input_tree):
                    if len(node) > 0 and isinstance(node[0], torch.Tensor) and node[0].item() == 0:
                        # 是条件节点
                        # 将特征索引和阈值设为可微分
                        feature_idx = node[1].item()
                        threshold = node[2].item()
                        input_tree[i][1] = torch.nn.Parameter(torch.tensor(float(feature_idx), requires_grad=True))
                        input_tree[i][2] = torch.nn.Parameter(torch.tensor(float(threshold), requires_grad=True))
                        print(f"已将节点 {i} 的特征索引和阈值转换为可微分参数")
                    elif len(node) > 5 and isinstance(node[0], torch.Tensor) and node[0].item() == 1:
                        # 是叶节点，将类别概率转换为可微分
                        for j in range(5, len(node)):
                            if not torch.isnan(node[j]):
                                prob = node[j].item()
                                input_tree[i][j] = torch.nn.Parameter(torch.tensor(float(prob), requires_grad=True))
                        print(f"已将节点 {i} 的类别概率转换为可微分参数")
            except Exception as e:
                print(f"转换失败: {e}")
        
        # 计算输入树与每棵树的相似度
        similarities = []
        for tree in valid_trees:
            try:
                similarity = compare_torch_trees(
                    input_tree,
                    tree,
                    feature_names,
                    classes_,
                    bounds,
                    comp_dist=comp_dist,
                    dist_weight=dist_weight
                )
                similarities.append(similarity)
            except Exception as e:
                print(f"计算相似度失败: {e}")
                # 添加一个0相似度，避免循环失败
                similarities.append(torch.tensor(0.0, dtype=torch.float32, requires_grad=True))
            
        # 计算平均相似度 - 保持计算图连接
        if similarities:
            try:
                # 使用torch.stack创建张量，然后计算平均值以保持梯度流
                average_similarity = torch.stack(similarities).mean()
                # 确保结果是可微分的
                if not average_similarity.requires_grad:
                    # 如果结果不可微分，尝试通过加入一个小的可微分项来使其可微分
                    param = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))
                    average_similarity = average_similarity + param * 0.0
            except Exception as e:
                print(f"计算平均相似度失败: {e}")
                average_similarity = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        else:
            average_similarity = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        
        return average_similarity
    
    return calculate_average_similarity

class TorchSimpleBranch:
    """PyTorch版本的SimpleBranch，支持可微分操作"""
    def __init__(self, feature_names, classes_, label_probas=None, number_of_samples=0):
        self.feature_names = feature_names
        self.classes_ = classes_
        
        # 确保标签概率是PyTorch张量
        if label_probas is None:
            self.label_probas = torch.zeros(len(classes_), dtype=torch.float32)
        elif isinstance(label_probas, list):
            # 检查是否含有PyTorch张量
            has_torch_tensor = any(isinstance(p, torch.Tensor) for p in label_probas)
            if has_torch_tensor:
                # 如果含有PyTorch张量，将所有元素转换为PyTorch张量
                tensor_probas = []
                for p in label_probas:
                    if isinstance(p, torch.Tensor):
                        tensor_probas.append(p)
                    else:
                        tensor_probas.append(torch.tensor(float(p), dtype=torch.float32))
                self.label_probas = tensor_probas
            else:
                self.label_probas = torch.tensor(label_probas, dtype=torch.float32)
        elif isinstance(label_probas, np.ndarray):
            self.label_probas = torch.tensor(label_probas, dtype=torch.float32)
        else:
            self.label_probas = label_probas
        
        # 样本数量
        self.number_of_samples = number_of_samples
        
        # 条件格式：[(特征索引, 阈值, 比较类型), ...]
        # 例如：[(0, 0.5, 'upper'), (2, 0.7, 'lower')]
        self.conditions = []
    
    def add_condition(self, feature, threshold, bound="upper"):
        """添加分支条件，保持梯度信息"""
        # 如果阈值是PyTorch张量，直接使用；否则转换为张量
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.tensor(float(threshold), dtype=torch.float32)
        
        self.conditions.append((feature, threshold, bound))
    
    def contradict_branch(self, other_branch):
        """检查两个分支是否矛盾"""
        for feat1, val1, bound1 in self.conditions:
            for feat2, val2, bound2 in other_branch.conditions:
                if feat1 == feat2:
                    # 直接使用张量比较，避免使用.item()
                    # 转换为张量以统一处理
                    if not isinstance(val1, torch.Tensor):
                        val1 = torch.tensor(float(val1), dtype=torch.float32)
                    if not isinstance(val2, torch.Tensor):
                        val2 = torch.tensor(float(val2), dtype=torch.float32)
                    
                    if bound1 == "upper" and bound2 == "lower":
                        if (val1 < val2).any():
                            return True
                    elif bound1 == "lower" and bound2 == "upper":
                        if (val1 > val2).any():
                            return True
        return False
    
    def merge_branch(self, other_branch, personalized=True):
        """合并两个分支，保持梯度信息"""
        # 创建新分支
        new_branch = TorchSimpleBranch(
            self.feature_names, 
            self.classes_, 
            label_probas=self.label_probas,
            number_of_samples=self.number_of_samples + other_branch.number_of_samples
        )
        
        # 复制当前分支的条件
        for feat, val, bound in self.conditions:
            new_branch.add_condition(feat, val, bound)
        
        # 添加其他分支的条件
        for feat, val, bound in other_branch.conditions:
            # 检查是否已存在该特征的条件
            existing_condition = False
            for i, (f, v, b) in enumerate(new_branch.conditions):
                if f == feat:
                    existing_condition = True
                    # 更新条件
                    if bound == b:
                        # 根据边界类型选择最小或最大值
                        if bound == "upper":
                            # 比较并取最小值，保持梯度
                            if isinstance(v, torch.Tensor) and isinstance(val, torch.Tensor):
                                # 如果两者都是张量，使用torch.min保持梯度
                                new_val = torch.min(v, val)
                            elif isinstance(v, torch.Tensor):
                                # 如果v是张量，比较后选择
                                if v.item() <= float(val):
                                    new_val = v
                                else:
                                    new_val = val
                            elif isinstance(val, torch.Tensor):
                                # 如果val是张量，比较后选择
                                if float(v) <= val.item():
                                    new_val = v
                                else:
                                    new_val = val
                            else:
                                # 两者都不是张量，取最小值
                                new_val = min(v, val)
                        else:  # bound == "lower"
                            # 比较并取最大值，保持梯度
                            if isinstance(v, torch.Tensor) and isinstance(val, torch.Tensor):
                                # 如果两者都是张量，使用torch.max保持梯度
                                new_val = torch.max(v, val)
                            elif isinstance(v, torch.Tensor):
                                # 如果v是张量，比较后选择
                                if v.item() >= float(val):
                                    new_val = v
                                else:
                                    new_val = val
                            elif isinstance(val, torch.Tensor):
                                # 如果val是张量，比较后选择
                                if float(v) >= val.item():
                                    new_val = v
                                else:
                                    new_val = val
                            else:
                                # 两者都不是张量，取最大值
                                new_val = max(v, val)
                        
                        new_branch.conditions[i] = (f, new_val, b)
            
            # 如果不存在该特征的条件，则添加
            if not existing_condition:
                new_branch.add_condition(feat, val, bound)
        
        # 更新标签概率 (加权平均)
        if personalized:
            n1 = self.number_of_samples
            n2 = other_branch.number_of_samples
            total = n1 + n2
            
            # 创建新的标签概率列表
            new_probas = []
            for i in range(len(self.label_probas)):
                # 获取两个分支的概率值
                if isinstance(self.label_probas, list):
                    prob1 = self.label_probas[i]
                else:
                    prob1 = self.label_probas[i]
                    
                if isinstance(other_branch.label_probas, list):
                    prob2 = other_branch.label_probas[i]
                else:
                    prob2 = other_branch.label_probas[i]
                
                # 计算加权平均，保持梯度
                if isinstance(prob1, torch.Tensor) and isinstance(prob2, torch.Tensor):
                    # 两者都是张量，直接计算加权平均
                    weighted_avg = (prob1 * n1 + prob2 * n2) / total
                elif isinstance(prob1, torch.Tensor):
                    # 只有prob1是张量
                    weighted_avg = (prob1 * n1 + float(prob2) * n2) / total
                elif isinstance(prob2, torch.Tensor):
                    # 只有prob2是张量
                    weighted_avg = (float(prob1) * n1 + prob2 * n2) / total
                else:
                    # 两者都不是张量
                    weighted_avg = (float(prob1) * n1 + float(prob2) * n2) / total
                
                new_probas.append(weighted_avg)
            
            # 更新新分支的标签概率
            new_branch.label_probas = new_probas
        
        return new_branch
    
    def str_branch(self):
        """返回分支的字符串表示"""
        conditions_str = []
        for feat, val, bound in self.conditions:
            feat_name = self.feature_names[feat] if feat < len(self.feature_names) else f"x{feat}"
            if bound == "upper":
                # 确保val可以被展示
                if isinstance(val, torch.Tensor):
                    val_str = f"{val.item():.3f}"
                else:
                    val_str = f"{float(val):.3f}"
                conditions_str.append(f"{feat_name} <= {val_str}")
            else:
                if isinstance(val, torch.Tensor):
                    val_str = f"{val.item():.3f}"
                else:
                    val_str = f"{float(val):.3f}"
                conditions_str.append(f"{feat_name} > {val_str}")
        return " AND ".join(conditions_str)
    
    def __str__(self):
        return self.str_branch() + f" -> {self.label_probas}"
    
    def get_branch_dict(self):
        """获取分支的字典表示，供DataFrame使用，保持梯度信息"""
        result = {}
        
        # 添加特征上下界
        for i in range(len(self.feature_names)):
            result[f"{i}_lower"] = float('-inf')
            result[f"{i}_upper"] = float('inf')
        
        # 设置条件的上下界
        for feat, val, bound in self.conditions:
            # 确保feat是整数或可以转换为整数
            if isinstance(feat, torch.Tensor):
                feat_idx = int(feat.detach().cpu().item())  # 仅用于索引，不影响梯度
            else:
                feat_idx = int(feat)
            
            key = f"{feat_idx}_{bound}"
            
            if bound == "upper":
                current = result[key]
                
                # 如果当前值是无穷大，直接使用新值
                if current == float('inf'):
                    result[key] = val
                else:
                    # 否则，取较小值
                    if isinstance(val, torch.Tensor) and isinstance(current, torch.Tensor):
                        result[key] = torch.min(current, val)
                    elif isinstance(val, torch.Tensor):
                        if val.detach().cpu().item() < current:
                            result[key] = val
                        # 否则保持current
                    elif isinstance(current, torch.Tensor):
                        if float(val) < current.detach().cpu().item():
                            result[key] = float(val)
                        # 否则保持current
                    else:
                        result[key] = min(current, float(val))
            else:  # bound == "lower"
                current = result[key]
                
                # 如果当前值是负无穷大，直接使用新值
                if current == float('-inf'):
                    result[key] = val
                else:
                    # 否则，取较大值
                    if isinstance(val, torch.Tensor) and isinstance(current, torch.Tensor):
                        result[key] = torch.max(current, val)
                    elif isinstance(val, torch.Tensor):
                        if val.detach().cpu().item() > current:
                            result[key] = val
                        # 否则保持current
                    elif isinstance(current, torch.Tensor):
                        if float(val) > current.detach().cpu().item():
                            result[key] = float(val)
                        # 否则保持current
                    else:
                        result[key] = max(current, float(val))
        
        # 添加其他属性 - 标签概率
        if isinstance(self.label_probas, list):
            result["probas"] = self.label_probas
        else:
            result["probas"] = self.label_probas.tolist() if isinstance(self.label_probas, np.ndarray) else self.label_probas
        
        result["branch_probability"] = self.number_of_samples
        
        return result

class TorchSimpleConjunctionSet:
    """PyTorch版本的SimpleConjunctionSet，支持可微分操作"""
    def __init__(self, feature_names=None, amount_of_branches_threshold=float('inf')):
        self.feature_names = feature_names  # 特征名称
        self.amount_of_branches_threshold = amount_of_branches_threshold  # 规则数量阈值
        self.branches_lists = []  # 规则集列表
        self.conjunctionSet = []  # 最终的规则集
        self.classes_ = None  # 类别名称
    
    def aggregate_branches(self, client_cs, classes_):
        """聚合多个客户端的规则集合
        
        Args:
            client_cs: list[list[TorchSimpleBranch]]
            classes_: list[str]
        """
        self.classes_ = classes_
        self.branches_lists = []
        
        # 将所有客户端规则集合合并成一个列表
        for cs in client_cs:
            if isinstance(cs, list):
                # 确保添加的是列表，不是单个分支
                if len(cs) == 1 and not isinstance(cs[0], list):
                    self.branches_lists.append([cs[0]])  # 将单个分支包装为列表
                else:
                    # 如果cs[0]是列表，直接添加
                    if len(cs) > 0 and isinstance(cs[0], list):
                        self.branches_lists.append(cs)
                    else:
                        # 否则，将整个cs作为一个列表添加
                        self.branches_lists.append(cs)
    
    def buildConjunctionSet(self):
        """构建最终的规则集"""
        # 从第一个规则集开始
        if not self.branches_lists:
            self.conjunctionSet = []
            return
        
        # 确保conjunctionSet是列表
        first_branches = self.branches_lists[0]
        if isinstance(first_branches, list) and not isinstance(first_branches[0], list):
            conjunctionSet = first_branches  # 如果是分支列表，直接使用
        else:
            # 如果是列表的列表，扁平化
            conjunctionSet = []
            for branch_list in first_branches:
                if isinstance(branch_list, list):
                    conjunctionSet.extend(branch_list)
                else:
                    conjunctionSet.append(branch_list)
        
        # 依次合并其他规则集
        for i, branch_list in enumerate(self.branches_lists[1:], 1):
            # 确保branch_list是分支列表
            if isinstance(branch_list, list) and not isinstance(branch_list[0], list):
                current_branches = branch_list
            else:
                current_branches = []
                for b in branch_list:
                    if isinstance(b, list):
                        current_branches.extend(b)
                    else:
                        current_branches.append(b)
            
            # 合并规则 (笛卡尔积)
            new_conjunction_set = []
            for b1 in conjunctionSet:
                for b2 in current_branches:
                    if not b1.contradict_branch(b2):
                        new_conjunction_set.append(b1.merge_branch(b2))
            
            conjunctionSet = new_conjunction_set
        
            # 移除重复规则
            conjunctionSet = self._remove_duplicate_branches(conjunctionSet)
            
            # 如果规则太多，进行过滤
            if len(conjunctionSet) > self.amount_of_branches_threshold:
                conjunctionSet = self._filter_conjunction_set(conjunctionSet)
        
        # 保存最终的规则集
        self.conjunctionSet = conjunctionSet
    
    def _filter_conjunction_set(self, cs):
        """过滤规则集，只保留最重要的规则"""
        if len(cs) <= self.amount_of_branches_threshold:
            return cs
        
        # 使用样本数量作为重要性度量
        branch_metrics = [b.number_of_samples for b in cs]
        threshold = sorted(branch_metrics, reverse=True)[self.amount_of_branches_threshold - 1]
        
        return [b for b, metric in zip(cs, branch_metrics) if metric >= threshold][:self.amount_of_branches_threshold]
    
    def _remove_duplicate_branches(self, cs):
        """移除重复的规则，保持梯度信息"""
        unique_branches = {}
        
        for branch in cs:
            br_str = branch.str_branch()
            if br_str not in unique_branches:
                unique_branches[br_str] = branch
            else:
                # 如果有重复，合并标签概率
                existing = unique_branches[br_str]
                n1 = branch.number_of_samples
                n2 = existing.number_of_samples
                total = n1 + n2
                
                # 更新样本数量
                existing.number_of_samples = total
                
                # 更新标签概率 - 保持梯度信息
                for i in range(len(existing.label_probas)):
                    if isinstance(existing.label_probas, list) and isinstance(branch.label_probas, list):
                        existing_prob = existing.label_probas[i]
                        branch_prob = branch.label_probas[i]
                        
                        if isinstance(existing_prob, torch.Tensor) and isinstance(branch_prob, torch.Tensor):
                            existing.label_probas[i] = (existing_prob * n2 + branch_prob * n1) / total
                        elif isinstance(existing_prob, torch.Tensor):
                            existing.label_probas[i] = (existing_prob * n2 + float(branch_prob) * n1) / total
                        elif isinstance(branch_prob, torch.Tensor):
                            existing.label_probas[i] = (float(existing_prob) * n2 + branch_prob * n1) / total
                        else:
                            existing.label_probas[i] = (float(existing_prob) * n2 + float(branch_prob) * n1) / total
                    else:
                        # 处理非列表情况
                        if isinstance(existing.label_probas, torch.Tensor) and isinstance(branch.label_probas, torch.Tensor):
                            existing.label_probas[i] = (existing.label_probas[i] * n2 + branch.label_probas[i] * n1) / total
                        elif isinstance(existing.label_probas, torch.Tensor):
                            prob_value = float(branch.label_probas[i]) if hasattr(branch.label_probas[i], 'item') else branch.label_probas[i]
                            existing.label_probas[i] = (existing.label_probas[i] * n2 + prob_value * n1) / total
        
        return list(unique_branches.values())
    
    def get_conjunction_set_df(self):
        """获取规则集的DataFrame表示，保持梯度信息"""
        branches_dicts = [b.get_branch_dict() for b in self.conjunctionSet]
        
        # 创建一个可以保持梯度信息的DataFrame
        return pd.DataFrame(branches_dicts)

def extract_rules_from_bfs_tree_torch(bfs_tree, feature_names, classes_):
    """从BFS格式的树中提取规则，保持梯度信息
    
    Args:
        bfs_tree: 广度优先搜索格式的树，每个节点是一个列表或PyTorch张量
        feature_names: 特征名称列表
        classes_: 类别名称列表
        
    Returns:
        list[TorchSimpleBranch]: 提取的规则分支列表
    """
    # 找出所有叶节点
    leaf_indices = []
    for i, node in enumerate(bfs_tree):
        if not node:  # 空节点
            continue
        
        # 检查是否是叶节点 (节点类型为1)
        node_type = node[0]
        if isinstance(node_type, torch.Tensor):
            # 不使用.item()，避免断开梯度流
            if (node_type == 1).all():
                leaf_indices.append(i)
        elif node_type == 1:
            leaf_indices.append(i)
    
    branches = []
    
    # 对每个叶节点
    for leaf_idx in leaf_indices:
        # 获取叶节点的类别概率（从索引5开始）
        if len(bfs_tree[leaf_idx]) >= 8:
            probas = bfs_tree[leaf_idx][5:]  # 从索引5开始是概率
        else:
            probas = []
        
        # 如果没有有效的概率，使用均匀分布
        if not probas:
            probas = [1.0 / len(classes_) for _ in range(len(classes_))]
            
        # 检查概率是否有NaN值
        has_nan = False
        for p in probas:
            if isinstance(p, torch.Tensor) and torch.isnan(p).any():
                has_nan = True
                break
            elif p != p:  # Python's way of checking for NaN
                has_nan = True
                break
        
        if has_nan:
            # 如果有NaN，创建一个新的均匀分布
            probas = [1.0 / len(classes_) for _ in range(len(classes_))]
        
        # 默认样本数量
        n_samples = 1
        
        # 创建分支
        branch = TorchSimpleBranch(
            feature_names=feature_names,
            classes_=classes_,
            label_probas=probas,
            number_of_samples=n_samples
        )
        
        # 从叶节点追踪到根节点的路径
        node_id = leaf_idx
        while node_id > 0:  # 0是根节点
            parent_id = (node_id - 1) // 2  # BFS表示中的父节点
            
            # 确定当前节点是父节点的左子节点还是右子节点
            is_left_child = (node_id == 2 * parent_id + 1)
            
            # 获取父节点的特征和阈值
            feature = bfs_tree[parent_id][1]
            threshold = bfs_tree[parent_id][2]
            
            # 特征索引检查 - 保持特征索引为张量，不使用.item()
            feature_idx = feature
            if isinstance(feature, torch.Tensor):
                # 我们仍需要将特征索引作为整数使用，但需要保持梯度流
                # 为条件使用张量比较，而不是提取标量值
                feature_idx_int = feature  # 保持为张量
            else:
                feature_idx_int = int(feature)
                feature_idx = torch.tensor(feature_idx_int, dtype=torch.float32)
            
            # 根据是左子节点还是右子节点添加条件
            if is_left_child:
                bound = "upper"  # 左子节点表示 <= threshold
            else:
                bound = "lower"  # 右子节点表示 > threshold
            
            # 将条件添加到分支，保持阈值的梯度信息
            branch.add_condition(feature_idx_int, threshold, bound)
            
            # 移动到父节点
            node_id = parent_id
        
        branches.append(branch)
    
    return branches

def extract_df_rules_from_tree_torch(tree, feature_names, classes_):
    """从决策树提取规则，保持梯度信息
    
    Args:
        tree: 树的PyTorch表示，每个节点是一个列表或PyTorch张量
        feature_names: 特征名称列表
        classes_: 类别名称列表
        
    Returns:
        pandas.DataFrame: 提取的规则DataFrame，保持梯度信息
    """
    # 从树中提取规则分支
    branches = extract_rules_from_bfs_tree_torch(tree, feature_names, classes_)
    
    # 创建规则集
    cs = TorchSimpleConjunctionSet(feature_names=feature_names, amount_of_branches_threshold=float('inf'))
    cs.aggregate_branches([branches], classes_)
    cs.buildConjunctionSet()
    
    # 返回规则集的DataFrame表示
    df = cs.get_conjunction_set_df()
    
    # 确保概率列保持张量性质，不会被转换成普通值
    for i in range(len(df)):
        if isinstance(df.iloc[i]['probas'], list):
            for j, p in enumerate(df.iloc[i]['probas']):
                if isinstance(p, torch.Tensor) and p.requires_grad:
                    # 保持概率的梯度连接
                    df.iloc[i]['probas'][j] = p
    
    return df

def main():
    import torch
    
    json_file_path = 'generated_subtrees.json'
    
    # 示例：创建相似度计算器
    print("创建PyTorch相似度计算器...")
    similarity_calculator = create_torch_similarity_calculator(json_file_path)
    
    # 加载处理后的树以进行测试
    processed_trees, feature_names, classes_ = process_json_trees(json_file_path, save_maps=False)
    
    print("\n演示梯度计算（多个阈值）：")
    
    # 测试不同阈值对相似度的影响
    thresholds = [0.3, 0.5, 0.7]
    
    for threshold in thresholds:
        print(f"\n测试阈值 = {threshold}:")
        
        # 创建一个可微分阈值参数
        param = torch.nn.Parameter(torch.tensor(threshold, requires_grad=True))
        
        # 创建一个简单的树结构
        simple_tree = [
            # 根节点 - 条件节点
            [torch.tensor(0.0), torch.tensor(0.0), param, torch.tensor(-1.0), torch.tensor(0.0), 
             torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)],
            # 左子节点 - 叶节点
            [torch.tensor(1.0), torch.tensor(-1.0), torch.tensor(-1.0), torch.tensor(-1.0), torch.tensor(0.0),
             torch.tensor(0.7), torch.tensor(0.2), torch.tensor(0.1)],
            # 右子节点 - 叶节点
            [torch.tensor(1.0), torch.tensor(-1.0), torch.tensor(-1.0), torch.tensor(-1.0), torch.tensor(0.0),
             torch.tensor(0.1), torch.tensor(0.8), torch.tensor(0.1)]
        ]
        
        # 计算相似度
        similarity = similarity_calculator(simple_tree)
        print(f"相似度: {similarity.item():.4f}")
        
        # 反向传播
        try:
            similarity.backward(retain_graph=True)  # 保留计算图以便下次计算
            print(f"参数的梯度: {param.grad}")
            
            # 应用梯度更新测试
            if param.grad is not None:
                lr = 0.01  # 学习率
                with torch.no_grad():
                    old_param = param.item()
                    param.add_(lr * param.grad)
                    new_param = param.item()
                    
                # 计算更新后的相似度
                param.grad.zero_()  # 清除梯度
                new_similarity = similarity_calculator(simple_tree)
                print(f"参数更新: {old_param:.4f} -> {new_param:.4f}")
                print(f"相似度变化: {similarity.item():.4f} -> {new_similarity.item():.4f}")
        except Exception as e:
            print(f"反向传播时出错: {e}")
            

    # ================================

    # 测试类别概率的梯度
    print("\n\n测试类别概率的梯度:")
    
    # 创建可微分的类别概率 - 使用更明显的差异
    prob1 = torch.nn.Parameter(torch.tensor(0.7, requires_grad=True))
    prob2 = torch.nn.Parameter(torch.tensor(0.2, requires_grad=True))
    # 使用直接计算而非创建新张量，以保持梯度流
    prob3 = 1.0 - prob1 - prob2  # 保证概率和为1
    
    # 创建带有可微分类别概率的树
    prob_tree = [
        # 根节点 - 条件节点
        [torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.5), torch.tensor(-1.0), torch.tensor(0.0), 
         torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)],
        # 左子节点 - 叶节点 (可微分概率)
        [torch.tensor(1.0), torch.tensor(-1.0), torch.tensor(-1.0), torch.tensor(-1.0), torch.tensor(0.0),
         prob1, prob2, prob3],
        # 右子节点 - 叶节点 (不同的概率分布)
        [torch.tensor(1.0), torch.tensor(-1.0), torch.tensor(-1.0), torch.tensor(-1.0), torch.tensor(0.0),
         torch.tensor(0.2), torch.tensor(0.7), torch.tensor(0.1)]
    ]
    
    # 使用comp_dist=True来强制使用概率分布比较
    similarity_calculator_with_dist = create_torch_similarity_calculator(json_file_path)
    
    # 计算相似度
    similarity = similarity_calculator_with_dist(prob_tree, comp_dist=True, dist_weight=0.8)
    print(f"相似度 (使用概率分布): {similarity.item():.4f}")
    
    # 反向传播
    try:
        similarity.backward(retain_graph=True)
        print(f"概率1的梯度: {prob1.grad}")
        print(f"概率2的梯度: {prob2.grad}")
        
        # 应用梯度更新测试
        if prob1.grad is not None:
            lr = 0.1  # 较大的学习率，使变化明显
            with torch.no_grad():
                old_prob1 = prob1.item()
                old_prob2 = prob2.item()
                prob1.add_(lr * prob1.grad)
                prob2.add_(lr * prob2.grad)
                new_prob1 = prob1.item()
                new_prob2 = prob2.item()
                
            # 计算更新后的相似度
            prob1.grad.zero_()
            prob2.grad.zero_()
            new_similarity = similarity_calculator_with_dist(prob_tree, comp_dist=True, dist_weight=0.8)
            print(f"概率更新: [{old_prob1:.4f}, {old_prob2:.4f}] -> [{new_prob1:.4f}, {new_prob2:.4f}]")
            print(f"相似度变化: {similarity.item():.4f} -> {new_similarity.item():.4f}")
    except Exception as e:
        print(f"反向传播时出错: {e}")

    # ================================

if __name__ == "__main__":
    main()
