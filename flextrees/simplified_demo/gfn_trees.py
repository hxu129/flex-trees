import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
import pandas as pd

class SimpleBranch:
    def __init__(self, feature_names, classes_, label_probas=None, number_of_samples=0):
        self.feature_names = feature_names
        self.classes_ = classes_
        self.label_probas = label_probas if label_probas is not None else []
        self.number_of_samples = number_of_samples
        # 条件格式：(特征索引, 阈值, 比较类型)
        self.conditions = []  # 例如：[(0, 0.5, 'upper'), (2, 0.7, 'lower')]
    
    def add_condition(self, feature, threshold, bound="upper"):
        """添加分支条件"""
        self.conditions.append((feature, threshold, bound))
    
    def contradict_branch(self, other_branch):
        """检查两个分支是否矛盾"""
        for feat1, val1, bound1 in self.conditions:
            for feat2, val2, bound2 in other_branch.conditions:
                if feat1 == feat2:
                    if bound1 == "upper" and bound2 == "lower":
                        if val1 < val2:
                            return True
                    elif bound1 == "lower" and bound2 == "upper":
                        if val1 > val2:
                            return True
        return False
    
    def merge_branch(self, other_branch, personalized=True):
        """合并两个分支"""
        # 创建新分支
        new_branch = SimpleBranch(
            self.feature_names, 
            self.classes_, 
            label_probas=self.label_probas.copy(),
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
                        if bound == "upper":
                            new_val = min(v, val)
                        else:
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
            for i in range(len(self.label_probas)):
                new_branch.label_probas[i] = (self.label_probas[i] * n1 + 
                                             other_branch.label_probas[i] * n2) / total
        
        return new_branch
    
    def str_branch(self):
        """返回分支的字符串表示"""
        conditions_str = []
        for feat, val, bound in self.conditions:
            feat_name = self.feature_names[feat] if feat < len(self.feature_names) else f"x{feat}"
            if bound == "upper":
                conditions_str.append(f"{feat_name} <= {val:.3f}")
            else:
                conditions_str.append(f"{feat_name} > {val:.3f}")
        return " AND ".join(conditions_str)
    
    def __str__(self):
        return self.str_branch() + f" -> {self.label_probas}"
    
    def get_branch_dict(self):
        """获取分支的字典表示，供DataFrame使用"""
        result = {}
        
        # 添加特征上下界
        for i in range(len(self.feature_names)):
            result[f"{i}_lower"] = float('-inf')
            result[f"{i}_upper"] = float('inf')
        
        # 设置条件的上下界
        for feat, val, bound in self.conditions:
            if bound == "upper":
                result[f"{feat}_upper"] = min(result[f"{feat}_upper"], val)
            else:
                result[f"{feat}_lower"] = max(result[f"{feat}_lower"], val)
        
        # 添加其他属性 - 确保probas是numpy数组而不是列表
        result["probas"] = np.array(self.label_probas)
        result["branch_probability"] = self.number_of_samples
        
        return result

class SimpleConjunctionSet:
    def __init__(self, feature_names=None, amount_of_branches_threshold=np.inf):
        self.feature_names = feature_names # 特征名称
        self.amount_of_branches_threshold = amount_of_branches_threshold # 如果规则太多，进行过滤
        self.branches_lists = [] # 规则集列表
        self.conjunctionSet = [] # 最终的规则集
        self.classes_ = None # 类别名称
    
    def aggregate_branches(self, client_cs, classes_):
        """聚合多个客户端的规则集合，用来由多个branch生成一个conjunctionSet的后续branch
        client_cs: list[list[SimpleBranch]]
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
                    # 如果cs[0]是列表(如果客户端返回的是列表的列表)，直接添加
                    if len(cs) > 0 and isinstance(cs[0], list):
                        self.branches_lists.append(cs)
                    else:
                        # 否则，将整个cs作为一个列表添加
                        self.branches_lists.append(cs)
    
    def buildConjunctionSet(self):
        """构建最终的规则集，用来由多个branch生成一个conjunctionSet"""
        # 从第一个规则集开始
        if not self.branches_lists:
            self.conjunctionSet = []
            return
        
        # 确保conjunctionSet是列表
        first_branches = self.branches_lists[0]
        if isinstance(first_branches, list) and not isinstance(first_branches[0], list):
            conjunctionSet = first_branches  # 如果是分支列表，直接使用
            # conjunctionset is just a list of branches
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
            print(f"合并规则集 {i+1}，当前规则数: {len(conjunctionSet)}")
            
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
            
            # 合并规则 with Cartision product
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
                print(f"规则数量太多，过滤后剩余: {len(conjunctionSet)}")
        
        # Save the final conjunctionSet
        self.conjunctionSet = conjunctionSet
        print(f"最终规则数: {len(self.conjunctionSet)}")
    
    def _filter_conjunction_set(self, cs):
        """过滤规则集，只保留最重要的规则"""
        if len(cs) <= self.amount_of_branches_threshold:
            return cs
        
        # 使用样本数量作为重要性度量
        branch_metrics = [b.number_of_samples for b in cs]
        threshold = sorted(branch_metrics, reverse=True)[self.amount_of_branches_threshold - 1]
        
        return [b for b, metric in zip(cs, branch_metrics) if metric >= threshold][:self.amount_of_branches_threshold]
    
    def _remove_duplicate_branches(self, cs):
        """移除重复的规则"""
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
                
                # 更新标签概率
                for i in range(len(existing.label_probas)):
                    existing.label_probas[i] = (existing.label_probas[i] * n2 + 
                                              branch.label_probas[i] * n1) / total
        
        return list(unique_branches.values())
    
    def get_conjunction_set_df(self):
        """获取规则集的DataFrame表示"""
        branches_dicts = [b.get_branch_dict() for b in self.conjunctionSet]
        return pd.DataFrame(branches_dicts).round(decimals=5)

def extract_rules_from_bfs_tree(bfs_tree, feature_names, classes_):
    """Extract rules from a tree organized in BFS format"""
    # Find all leaf nodes - with numpy compatibility
    leaf_indices = [
        i for i, node in enumerate(bfs_tree)
        if isinstance(node, (list, np.ndarray)) and len(node) > 0 and node[0] == 1
    ]
    
    branches = []
    
    # For each leaf node
    for leaf_idx in leaf_indices:
        # Get the class probabilities from the leaf node (indices 5,6,7)
        if len(bfs_tree[leaf_idx]) >= 8:
            probas = bfs_tree[leaf_idx][5:] # Starting from index 5 there are the probabilities
        else:
            probas = []
        
        # If no valid probabilities, use uniform distribution
        if probas is None or len(probas) == 0:
            probas = [1.0 / len(classes_)] * len(classes_)

        # If nan is in the probas, raise an error
        if any(np.isnan(probas)):
            raise ValueError("NaN found in probabilities")
        
        # TODO: 在未来我们可能可以根据其他的性质来确定到达这个branch的样本数量
        n_samples = 1  # Default number of samples
        
        # Create branch
        branch = SimpleBranch(
            feature_names=feature_names,
            classes_=classes_,
            label_probas=list(probas),
            number_of_samples=n_samples
        )
        
        # Trace path from leaf to root
        node_id = leaf_idx
        while node_id > 0:  # 0 is the root node
            parent_id = (node_id - 1) // 2  # Parent in BFS representation
            
            # Determine if current node is left or right child of parent
            is_left_child = (node_id == 2 * parent_id + 1)
            
            # Get parent's feature and threshold
            feature = bfs_tree[parent_id][1]
            threshold = bfs_tree[parent_id][2]
            
            
            # Add condition based on whether this is a left or right child
            if is_left_child:
                bound = "upper"  # Left child means <= threshold
            else:
                bound = "lower"  # Right child means > threshold
            
            # Add condition to branch
            branch.add_condition(int(feature), threshold, bound)
            
            # Move to parent
            node_id = parent_id
        
        branches.append(branch)
    
    return branches
    

def extract_df_rules_from_tree(tree, feature_names, classes_):
    """从决策树提取规则"""
    branches = extract_rules_from_bfs_tree(tree, feature_names, classes_)
    cs = SimpleConjunctionSet(feature_names=feature_names, amount_of_branches_threshold=np.inf)
    cs.aggregate_branches([branches], classes_)
    cs.buildConjunctionSet()
    return cs.get_conjunction_set_df()

def compare_trees(tree1, tree2, feature_names, classes_, bounds, comp_dist=False, dist_weight=0.5):
    """比较两棵树的structural similarity
    tree1: list[list[int]]
    tree2: list[list[int]]
    feature_names: list[str]
    classes_: list[str]
    bounds: list[(float, float)] the bounds of the features; note that they should be in the same order as the feature_names
    comp_dist: bool, 是否计算标签分布差异，如果为False，那么如果两个branch的概率经过argmax后不相同，那么就认为这两个branch不匹配，条件非常严格
    dist_weight: float, 标签分布差异的权重
    """
    # Convert trees to numpy arrays first for consistent handling
    # This ensures all NaN values are correctly represented as np.nan
    # TODO: in the future if the running time is too long, we may modify the convert_to_numpy function
    def convert_to_numpy(tree):
        """Convert tree to numpy arrays, handling both numpy and torch data types."""
        # Try importing torch for tensor detection
        try:
            import torch
            HAS_TORCH = True
        except ImportError:
            HAS_TORCH = False

        # Check if it's a torch tensor
        if HAS_TORCH and isinstance(tree, torch.Tensor):
            # Convert tensor to numpy and handle potential torch.nan
            tree_np = tree.detach().cpu().numpy()
            return tree_np
    
        numpy_tree = []
        for node in tree:
        
            # Convert list to numpy array and pad with NaN
            node_array = np.full(len(node), np.nan)
            for i, val in enumerate(node):
                # Handle None case
                if val is None:
                    continue
                
                # Handle standard numpy/python nan case
                if not (isinstance(val, float) and np.isnan(val)):
                    node_array[i] = val
                
            numpy_tree.append(node_array)
        return numpy_tree

    
    # 比较两个 branch 的structural similarity
    def compare_branches(branch1, branch2, not_match_label=-np.inf):
        """比较两个 branch 的structural similarity，注意这里branch不是SimpleBranch，是df的行
        branch1: pd.Series
        branch2: pd.Series
        not_match_label: float, 是一个小值，表示两个branch不匹配，本来的返回值相似度应该是大值
        """
        # 比较两个 branch 的structural similarity
        dist_similarity = 0
        if comp_dist:
            # Compare the distribution of the class labels
            dist_similarity = jensenshannon(branch1["probas"], branch2["probas"])
        else:
            # step 1: compare the class label
            class_label_1 = np.argmax(branch1["probas"])
            class_label_2 = np.argmax(branch2["probas"])
            if class_label_1 != class_label_2:
                return not_match_label
    
        overlap = 0
        overall_range_1 = 0
        overall_range_2 = 0
        # step 2: compare the overlap regions and the overall range
        for feature in range(len(feature_names)):
            lower_1 = branch1[f"{feature}_lower"]
            upper_1 = branch1[f"{feature}_upper"] # region: lower_1 <= x <= upper_1
            lower_2 = branch2[f"{feature}_lower"]
            upper_2 = branch2[f"{feature}_upper"] # region: lower_2 <= x <= upper_2
            # calculate the overlap
            overlap += np.max([0, np.min([upper_1, upper_2]) - np.max([lower_1, lower_2])])
            # calculate the overall range
            overall_range_1 += upper_1 - lower_1
            overall_range_2 += upper_2 - lower_2

        branch_similarity = 2 * overlap / (overall_range_1 + overall_range_2)

        if comp_dist:
            branch_similarity = dist_similarity * dist_weight + branch_similarity * (1 - dist_weight)

        return branch_similarity

    # 定义一个函数来替换无穷值
    def replace_inf_with_bounds(series):
        if 'prob' in series.name: # 概率值不替换
            return series
        column_idx, bound_type = series.name.split("_")
        column_idx = int(column_idx)
        if bound_type == "upper":
            max_value = bounds[column_idx][1]
            return series.replace([np.inf], max_value)
        elif bound_type == "lower":
            min_value = bounds[column_idx][0]
            return series.replace([-np.inf], min_value)
        else:
            raise ValueError(f"Invalid bound type: {bound_type}")

    # Convert trees to numpy form for consistent processing
    np_tree1 = convert_to_numpy(tree1)
    np_tree2 = convert_to_numpy(tree2)

    branches1 = extract_df_rules_from_tree(np_tree1, feature_names, classes_)
    branches2 = extract_df_rules_from_tree(np_tree2, feature_names, classes_)

    # Handle case where there are no branches
    if len(branches1) == 0 or len(branches2) == 0:
        print("No branches found in one or both trees")
        return 0.0

    # 对每个特征应用替换函数
    branches1 = branches1.apply(replace_inf_with_bounds)
    branches2 = branches2.apply(replace_inf_with_bounds)
    print(branches1)
    print(branches2)
    print()

    similarity_matrix = np.zeros((len(branches1), len(branches2)))
    for i in range(len(branches1)):
        for j in range(len(branches2)):
            similarity_matrix[i, j] = compare_branches(branches1.iloc[i], branches2.iloc[j])
    
    # 使用匈牙利算法进行匹配
    row_ind, col_ind = [], []
    try:
        row_ind, col_ind = linear_sum_assignment(-1 * similarity_matrix) # minimize the cost matrix
    except:
        print("匈牙利算法匹配失败，尝试使用惩罚矩阵")
        M, N = similarity_matrix.shape
        unmatched_cost = 0
        penalty_matrix = np.full((M, M), -np.inf)
        np.fill_diagonal(penalty_matrix, unmatched_cost)
        augmented_matrix = np.hstack((similarity_matrix, penalty_matrix)) # 将 penalty_matrix 添加到 similarity_matrix 的右侧
        row_ind, col_ind = linear_sum_assignment(-1 * augmented_matrix) # 使用匈牙利算法进行匹配
    # 计算 unmapped branches
    unmapped_branches1_indices = set()
    unmapped_branches2_indices = set()
    # 去掉 penalty_matrix 的影响
    new_row_ind, new_col_ind = [], []
    for r,c in zip(row_ind, col_ind):
        if c < len(branches2):
            new_row_ind.append(r)
            new_col_ind.append(c)
        else:
            unmapped_branches1_indices.add(r)
    row_ind, col_ind = new_row_ind, new_col_ind
    unmapped_branches1_indices.update(set(range(len(branches1))) - set(row_ind))  
    unmapped_branches2_indices.update(set(range(len(branches2))) - set(col_ind))
    print(similarity_matrix)
    print(row_ind, col_ind)
    print(unmapped_branches1_indices, unmapped_branches2_indices)
    # 用最大可能相似度，近似计算 unmapped branches 的损失
    penalty = 0
    for i in unmapped_branches1_indices:
        penalty += np.max(similarity_matrix[i])
        print(np.max(similarity_matrix[i]))
    for j in unmapped_branches2_indices:
        penalty += np.max(similarity_matrix[:, j])
        print(np.max(similarity_matrix[:, j]))

    # 计算匹配的平均相似度
    total_similarity = 0
    for i, j in zip(row_ind, col_ind):
        total_similarity += similarity_matrix[i, j]
    average_similarity = 2 * (total_similarity - penalty) / (len(branches1) + len(branches2))
    
    return average_similarity



if __name__ == "__main__":
    # bound 貌似是(0, 1)，你暂且就按照全是(0,1)来处理但也要能处理nan
    class_str = ['c1','c2','c3']
    feature_names = ['x1','x2','x3','x4','x5'] #(最后三个features是分类的probability)
    tree1 = [[0, 3, 0.3, -1, 0, np.nan, np.nan, np.nan],  # 最好也支持tensor和含有torch.nan的tensor
            [1, -1, -1, -1, 0, 0.1, 0.2, 0.7], 
            [0, 3, 0.7, -1, 0, np.nan, np.nan, np.nan],  
            [np.nan, np.nan, np.nan, np.nan, np.nan], 
            [np.nan, np.nan, np.nan, np.nan, np.nan], 
            [1, -1, -1, -1, 0, 0.2, 0.3, 0.5],
            [1, -1, -1, -1, 0, 0.4, 0.5, 0.1],
            [np.nan, np.nan, np.nan, np.nan, np.nan], 
            [np.nan, np.nan, np.nan, np.nan, np.nan], 
            [0, np.nan, np.nan, np.nan, np.nan]] 
    tree2 = [[0, 3, 0.3, -1, 0, np.nan, np.nan, np.nan], 
            [1, -1, -1, -1, 0, 0.6, 0.2, 0.2], 
            [1, -1, -1, -1, 0, 0.3, 0.3, 0.4],
            [np.nan, np.nan, np.nan, np.nan, np.nan], 
            [np.nan, np.nan, np.nan, np.nan, np.nan], 
            [np.nan, np.nan, np.nan, np.nan, np.nan], 
            [np.nan, np.nan, np.nan, np.nan, np.nan], 
            [np.nan, np.nan, np.nan, np.nan, np.nan], 
            [np.nan, np.nan, np.nan, np.nan, np.nan], 
            [0, np.nan, np.nan, np.nan, np.nan]]
    print(compare_trees(tree1, tree2, feature_names, class_str, [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1)], comp_dist=True, dist_weight=0.5))