import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from scipy.optimize import linear_sum_assignment

def print_matrix(matrix):
    for row in matrix:
        print(" ".join(f"{cell:.2f}" for cell in row))

# 1. 简化版的Branch类 - 用于存储和管理单个规则
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

# 2. 简化版的ConjunctionSet类 - 用于管理规则集合
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

# 3. TreeBranch类 - 用于从规则构建决策树
class TreeBranch:
    def __init__(self, mask, classes=None, depth=0):
        self.mask = mask  # 标识哪些规则属于该节点的掩码
        self.classes_ = classes  # 类别
        self.left = None  # 左子节点
        self.right = None  # 右子节点
        self.split_feature = None  # 分裂特征
        self.split_value = None  # 分裂值
        self.depth = depth  # 节点深度

    # TODO: This is still greedy, and is not the original way of building the tree
    def split(self, df):
        """分裂当前节点，构建子树"""
        # 如果只有一个规则或无法继续分裂，则为叶节点
        if np.sum(self.mask) == 1:
            self.left = None
            self.right = None
            return
        
        # 确定可用特征
        self.features = [int(i.split("_")[0]) for i in df.keys() if ("_upper" in i or "_lower" in i)]
        
        # 选择最佳分裂特征和值
        self.split_feature, self.split_value = self.select_split_feature(df)
        print(self.split_feature, self.split_value)
        
        # 创建子节点的掩码
        self.create_mask(df)
        
        # 检查是否可分裂
        if not self.is_splitable():
            self.left = None
            self.right = None
            return
        
        # 创建左右子节点
        self.left = TreeBranch(
            list(np.logical_and(self.mask, np.logical_or(self.left_mask, self.both_mask))),
            self.classes_,
            depth=self.depth + 1,
        )
        self.right = TreeBranch(
            list(np.logical_and(self.mask, np.logical_or(self.right_mask, self.both_mask))),
            self.classes_,
            depth=self.depth + 1,
        )
        
        # 递归构建子树
        self.left.split(df)
        self.right.split(df)

    def is_splitable(self):
        """检查节点是否可分裂"""
        left_count = np.sum(np.logical_and(self.mask, np.logical_or(self.left_mask, self.both_mask)))
        right_count = np.sum(np.logical_and(self.mask, np.logical_or(self.right_mask, self.both_mask)))
        
        # 如果任一子节点为空或者节点不变，则不可分裂
        if left_count == 0 or right_count == 0:
            return False
        if left_count == np.sum(self.mask) or right_count == np.sum(self.mask):
            return False
            
        return True

    def create_mask(self, df):
        """为子节点创建掩码"""
        self.left_mask = df[f"{self.split_feature}_upper"] <= self.split_value
        self.right_mask = df[f"{self.split_feature}_lower"] >= self.split_value
        self.both_mask = (df[f"{self.split_feature}_lower"] < self.split_value) & (
            df[f"{self.split_feature}_upper"] > self.split_value
        )

    def select_split_feature(self, df):
        """选择最佳分裂特征和值"""
        feature_to_value = {}
        feature_to_metric = {}
        
        for feature in self.features:
            value, metric = self.check_feature_split_value(df, feature)
            feature_to_value[feature] = value
            feature_to_metric[feature] = metric
        
        # 选择信息增益最大（熵减少最多）的特征
        feature = min(feature_to_metric, key=feature_to_metric.get)
        return feature, feature_to_value[feature]

    def check_feature_split_value(self, df, feature):
        """计算特征的最佳分裂值"""
        value_to_metric = {}
        
        # 获取特征的所有可能取值
        values = list(set(
            list(df[f"{feature}_upper"][self.mask]) + 
            list(df[f"{feature}_lower"][self.mask])
        ))
        np.random.shuffle(values)
        
        # 评估每个可能的分裂值
        for value in values:
            left_mask = [True if upper <= value else False for upper in df[f"{feature}_upper"]]
            right_mask = [True if lower >= value else False for lower in df[f"{feature}_lower"]]
            both_mask = [
                True if value < upper and value > lower else False
                for lower, upper in zip(df[f"{feature}_lower"], df[f"{feature}_upper"])
            ]
            
            value_to_metric[value] = self.get_value_metric(df, left_mask, right_mask, both_mask)
        
        # 选择熵减少最多的值
        val = min(value_to_metric, key=value_to_metric.get)
        return val, value_to_metric[val]

    def get_value_metric(self, df, left_mask, right_mask, both_mask):
        """计算分裂值的度量（加权熵）"""
        l_df_mask = np.logical_and(np.logical_or(left_mask, both_mask), self.mask)
        r_df_mask = np.logical_and(np.logical_or(right_mask, both_mask), self.mask)
        
        # 如果任一子节点为空，返回无穷大
        if np.sum(l_df_mask) == 0 or np.sum(r_df_mask) == 0:
            return np.inf
        
        # 计算左右子节点的熵
        l_entropy = self.calculate_entropy(df, l_df_mask)
        r_entropy = self.calculate_entropy(df, r_df_mask)
        
        # 计算子节点的权重
        l_prop = np.sum(l_df_mask) / len(l_df_mask)
        r_prop = np.sum(r_df_mask) / len(l_df_mask)
        
        # 返回加权熵
        return l_entropy * l_prop + r_entropy * r_prop

    def predict_probas_and_depth(self, inst, training_df, explanation=None):
        """预测实例的概率和深度，并生成解释"""
        if explanation is None:
            explanation = {}
            
        # 如果是叶节点，返回节点概率
        if self.left is None and self.right is None:
            return self.node_probas(training_df), 1, {}
            
        # 根据分裂条件递归预测
        if inst[self.split_feature] <= self.split_value:
            prediction, depth, aux_explanation = self.left.predict_probas_and_depth(inst, training_df)
            explanation.update(aux_explanation)
            explanation[f"x{self.split_feature}"] = "<=" + str(self.split_value)
            return prediction, depth + 1, explanation
        else:
            prediction, depth, aux_explanation = self.right.predict_probas_and_depth(inst, training_df)
            explanation.update(aux_explanation)
            explanation[f"x{self.split_feature}"] = ">" + str(self.split_value)
            return prediction, depth + 1, explanation

    def node_probas(self, df):
        """获取节点的概率分布"""
        # 获取所有匹配节点的probas
        probas_list = df["probas"][self.mask].tolist()
        
        # 手动计算均值
        if not probas_list:
            return np.zeros(len(self.classes_))
        
        # 计算所有概率向量的平均值
        avg_probas = np.mean(np.vstack(probas_list), axis=0)
        
        # 确保概率和为1
        return avg_probas / np.sum(avg_probas) if np.sum(avg_probas) > 0 else avg_probas

    def calculate_entropy(self, test_df, test_df_mask):
        """计算节点的熵"""
        # 获取所有匹配节点的probas
        probas_list = test_df["probas"][test_df_mask].tolist()
        
        if not probas_list:
            return np.inf
        
        # 计算所有概率向量的平均值
        avg_probas = np.mean(np.vstack(probas_list), axis=0)
        
        # 确保概率和为1并计算熵
        normalized_probas = avg_probas / np.sum(avg_probas) if np.sum(avg_probas) > 0 else avg_probas
        return entropy(normalized_probas)

    def predict(self, X, classes_, branches_df):
        """预测多个实例的类别和解释"""
        probas, explanations = [], []
        
        # 遍历所有实例
        for inst in X:
            prob, _, explanation = self.predict_probas_and_depth(inst, branches_df)
            probas.append(prob)
            explanations.append(explanation)
        
        # 获取预测的类别
        predictions = [classes_[np.argmax(prob)] for prob in probas]
        
        # 生成解释
        explanations = [
            self.generate_explanation(pred, expl)
            for pred, expl in zip(predictions, explanations)
        ]
        
        return predictions, explanations
    
    def generate_explanation(self, target, explanation):
        """生成预测的解释文本"""
        ret = f"The instance was classified as {target}. Because:"
        ret += "".join([f" {feature}{value}," for feature, value in explanation.items()])
        return f"{ret[:-1]}." if explanation else f"The instance was classified as {target}."

# 4. 规则聚合函数
def generate_cs_dt_branches_from_list(client_cs, classes_, tree_model, threshold=100):
    """聚合客户端规则并构建全局模型
    client_cs: list[list[SimpleBranch]]
    classes_: list[str]
    tree_model: DecisionTreeClassifier
    threshold: int
    """
    # 创建规则集合对象
    cs = SimpleConjunctionSet(amount_of_branches_threshold=threshold)
    
    # 聚合客户端规则
    cs.aggregate_branches(client_cs, classes_)
    
    # 构建规则集
    cs.buildConjunctionSet()
    print(f"规则集大小: {len(cs.conjunctionSet)}")
    
    # 获取规则的DataFrame表示
    branches_df = cs.get_conjunction_set_df()
    
    # 规范化规则概率
    probabilities = branches_df["branch_probability"].tolist()
    total_probas = sum(probabilities)
    branches_df["branch_probability"] = branches_df["branch_probability"].map(lambda x: x / total_probas)
    
    # 检查NaN值
    if pd.isna(branches_df).any().any():
        print("警告: DataFrame中存在NaN值，将使用无穷大替换")
        branches_df = branches_df.fillna(np.inf)
    
    # 转换为字典格式，供TreeBranch使用
    branches_dict = {col: branches_df[col].values for col in branches_df.columns}
    
    # 创建全局决策树模型
    global_tree = tree_model([True] * len(branches_df), classes_)
    global_tree.split(branches_dict)
    
    return [cs, global_tree, branches_df]

# 比较两棵树的structural similarity
def compare_trees(tree1, tree2, feature_names, classes_, bounds, comp_dist=False, dist_weight=0.5):
    """比较两棵树的structural similarity
    tree1: DecisionTreeClassifier
    tree2: DecisionTreeClassifier
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

    branches1 = extract_df_rules_from_tree(tree1, feature_names, classes_)
    branches2 = extract_df_rules_from_tree(tree2, feature_names, classes_)

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

    # 对每个特征应用替换函数
    branches1 = branches1.apply(replace_inf_with_bounds)
    branches2 = branches2.apply(replace_inf_with_bounds)

    similarity_matrix = np.zeros((len(branches1), len(branches2)))
    for i in range(len(branches1)):
        for j in range(len(branches2)):
            similarity_matrix[i, j] = compare_branches(branches1.iloc[i], branches2.iloc[j])
    
    # 使用匈牙利算法进行匹配
    print_matrix(similarity_matrix) # shape of similarity_matrix: (len(branches1), len(branches2))
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


# 从单独一颗树当中提取df形式的规则
def extract_df_rules_from_tree(tree, feature_names, classes_):
    """从决策树提取规则"""
    branches = extract_rules_from_tree(tree, feature_names, classes_)
    cs = SimpleConjunctionSet(feature_names=feature_names, amount_of_branches_threshold=len(branches))
    cs.aggregate_branches([branches], classes_)
    cs.buildConjunctionSet()
    return cs.get_conjunction_set_df()

# 5. 用于从决策树提取规则的函数
def extract_rules_from_tree(tree, feature_names, classes_):
    """从决策树提取规则"""
    tree_ = tree.tree_
    
    # 获取所有叶节点
    leaf_indices = [
        i for i in range(tree_.node_count)
        if tree_.children_left[i] == -1 and tree_.children_right[i] == -1
    ]
    
    branches = []
    
    # 处理每个叶节点
    for leaf_idx in leaf_indices:
        # 获取叶节点样本的类别概率
        n_samples = tree_.n_node_samples[leaf_idx]
        value = tree_.value[leaf_idx][0]
        probas = value / value.sum()
        
        # 创建分支
        branch = SimpleBranch(
            feature_names=feature_names,
            classes_=classes_,
            label_probas=list(probas),
            number_of_samples=n_samples
        )
        
        # 从叶节点到根节点的路径
        node_id = leaf_idx
        while node_id > 0:  # 0是根节点
            # 找父节点
            parent_indices = np.where(tree_.children_left == node_id)[0]
            bound = "upper"  # 如果是左子节点，则条件是 <= threshold
            
            if len(parent_indices) == 0:
                parent_indices = np.where(tree_.children_right == node_id)[0]
                bound = "lower"  # 如果是右子节点，则条件是 > threshold
            
            parent_idx = parent_indices[0]
            feature = tree_.feature[parent_idx]
            threshold = tree_.threshold[parent_idx]
            
            # 添加条件
            branch.add_condition(feature, threshold, bound)
            
            # 移动到父节点
            node_id = parent_idx
        
        branches.append(branch)
    
    return branches

# 6. 演示
def run_demo():
    # 生成示例数据
    print("生成示例数据...")
    X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=42)
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    # 得到每一个feature的数值上下界
    bounds = []
    for i in range(X.shape[1]):
        bounds.append((X[:, i].min(), X[:, i].max()))
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 模拟3个客户端的数据
    num_clients = 2 
    print(f"\n模拟{num_clients}个客户端的数据...")
    client_data = []
    for i in range(num_clients):
        idx = np.random.choice(len(X_train), size=len(X_train)//num_clients, replace=False)
        client_data.append((X_train[idx], y_train[idx]))
    
    # 在每个客户端训练决策树
    print("\n在各客户端训练决策树...")
    client_trees = []
    tree_depth = 3
    for i, (X_c, y_c) in enumerate(client_data):
        tree = DecisionTreeClassifier(max_depth=tree_depth, random_state=42+i)
        tree.fit(X_c, y_c)
        client_trees.append(tree)
        print(f"客户端 {i+1} 训练完成，正确率: {tree.score(X_c, y_c):.4f}")


    # 计算每个客户端的树的structural similarity
    print("\n计算每个客户端的树的structural similarity...")
    client_similarity = []
    for i in range(num_clients):
        for j in range(i+1, num_clients):
            similarity = compare_trees(client_trees[i], client_trees[j], feature_names, np.unique(y), bounds, comp_dist=True, dist_weight=0.5)
            client_similarity.append((i, j, similarity))
            print(f"客户端 {i+1} 和 客户端 {j+1} 的structural similarity: {similarity:.4f}")
    
    # 从树中提取规则
    print("\n从决策树中提取规则...")
    client_branches = []
    for i, tree in enumerate(client_trees):
        branches = extract_rules_from_tree(tree, feature_names, np.unique(y))
        client_branches.append(branches) # list[list[SimpleBranch]]
        print(f"客户端 {i+1} 提取了 {len(branches)} 条规则")
    
    # 显示部分规则
    print("\n部分规则示例:")
    for i, branches in enumerate(client_branches):
        print(f"\n客户端 {i+1} 的规则示例:")
        for j, branch in enumerate(branches[:]):  # 只显示前两条
            print(f"  规则 {j+1}: {branch}")
    
    # 聚合规则构建全局模型
    print("\n聚合规则构建全局模型...")
    global_model = generate_cs_dt_branches_from_list(
        client_branches, np.unique(y), TreeBranch, threshold=50
    )
    
    # 使用全局模型进行预测
    print("\n使用全局模型进行预测...")
    global_tree = global_model[1]  # TreeBranch对象
    branches_df = global_model[2]  # 规则DataFrame
    
    # 测试全局树的效果
    predictions_train, explanations_train = global_tree.predict(X_train, np.unique(y), branches_df)
    predictions_test, explanations_test = global_tree.predict(X_test, np.unique(y), branches_df)
    
    # 显示预测结果
    print("\n预测结果示例:")
    print(f"训练集预测结果: {(predictions_train == y_train).mean()}")
    for i, (pred, exp, true) in enumerate(zip(predictions_train[:5], explanations_train[:5], y_train[:5])):
        print(f"\n样本 {i+1}:")
        print(f"真实类别: {true}")
        print(f"预测类别: {pred}")
        print(f"解释: {exp}")
    print()
    print(f"测试集预测结果: {(predictions_test == y_test).mean()}")
    for i, (pred, exp, true) in enumerate(zip(predictions_test[:5], explanations_test[:5], y_test[:5])):
        print(f"\n样本 {i+1}:")
        print(f"真实类别: {true}")
        print(f"预测类别: {pred}")
        print(f"解释: {exp}")

    # # show all rules
    # print("\n所有规则:")
    # # print the branches_df
    # print(branches_df.head(12))

if __name__ == "__main__":
    run_demo()
