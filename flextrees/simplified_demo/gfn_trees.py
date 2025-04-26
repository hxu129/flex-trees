import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import jensenshannon
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
    def __init__(self, feature_names=None, amount_of_branches_threshold=100):
        self.feature_names = feature_names
        self.amount_of_branches_threshold = amount_of_branches_threshold
        self.branches_lists = []
        self.conjunctionSet = []
        self.classes_ = None
    
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
            
            # 如果规则太多，进行过滤
            if len(conjunctionSet) > self.amount_of_branches_threshold:
                conjunctionSet = self._filter_conjunction_set(conjunctionSet)
                print(f"规则数量太多，过滤后剩余: {len(conjunctionSet)}")
        
        # 移除重复规则
        # TODO：应该先去重复，然后再过滤
        self.conjunctionSet = self._remove_duplicate_branches(conjunctionSet)
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
    """
    Extract rules from a tree organized in BFS format
    
    Parameters:
    -----------
    bfs_tree : list of lists
        Each inner list is a node in format [a, b, c, d, e, f]
        where b indicates if node is leaf (1) or internal (0),
        c is the feature index, and d is the threshold
    feature_names : list
        Names of features
    classes_ : list
        Class labels
    
    Returns:
    --------
    list of SimpleBranch objects
    """
    # Find all leaf nodes
    leaf_indices = [
        i for i, node in enumerate(bfs_tree)
        if node[1] == 1  # Check if it's a leaf node
        # TODO: modify the index for the correct flag
    ]
    
    branches = []
    
    # For each leaf node
    for leaf_idx in leaf_indices:
        # For simplicity, we're setting uniform class probabilities
        # You may need to adjust this based on your actual data

        # TODO: the label probas should be the probability of the leaf node
        probas = [1.0 / len(classes_)] * len(classes_) # 叶节点样本的类别概率 (distribution of the leaf node)
        # TODO: the number of samples should be the number of samples of the leaf node
        # n_samples = bfs_tree[leaf_idx][4]
        n_samples = 1 # 叶节点样本数量
        
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
            # TODO: modify the index for the correct feature and threshold
            feature = bfs_tree[parent_id][2]
            threshold = bfs_tree[parent_id][3]
            
            # Add condition based on whether this is a left or right child
            if is_left_child:
                bound = "upper"  # Left child means <= threshold
            else:
                bound = "lower"  # Right child means > threshold
            
            # Add condition to branch
            branch.add_condition(feature, threshold, bound)
            
            # Move to parent
            node_id = parent_id
        
        branches.append(branch)
    
    return branches
    

def extract_df_rules_from_tree(tree, feature_names, classes_):
    """从决策树提取规则"""
    branches = extract_rules_from_bfs_tree(tree, feature_names, classes_)
    cs = SimpleConjunctionSet(feature_names=feature_names, amount_of_branches_threshold=len(branches))
    cs.aggregate_branches([branches], classes_)
    cs.buildConjunctionSet()
    return cs.get_conjunction_set_df()

def compare_trees(tree1, tree2, feature_names, classes_, bounds, comp_dist=False, dist_weight=0.5):
    """比较两棵树的structural similarity
    tree1: list[list[int]]
    tree2: list[list[int]]
    feature_names: list[str]
    classes_: list[str]
    bounds: list[(float, float)]
    comp_dist: bool, 是否计算标签分布差异
    dist_weight: float, 标签分布差异的权重
    """
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

    branches1 = extract_df_rules_from_tree(tree1, feature_names, classes_)
    branches2 = extract_df_rules_from_tree(tree2, feature_names, classes_)

    # 对每个特征应用替换函数
    branches1 = branches1.apply(replace_inf_with_bounds)
    branches2 = branches2.apply(replace_inf_with_bounds)

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


# def convert_trees(tree: List[List[int]]) -> 
# """This module extract rules from trees in BFS format"""