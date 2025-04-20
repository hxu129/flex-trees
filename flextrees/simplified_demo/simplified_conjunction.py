import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
from statsmodels.distributions.empirical_distribution import ECDF

# 简化版Branch类
class SimpleBranch:
    def __init__(self, feature_names, feature_types, classes, label_probas=None, number_of_samples=0):
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.classes = classes
        self.label_probas = label_probas
        self.number_of_samples = number_of_samples
        # 存储条件：(特征索引, 阈值, 比较类型)
        self.conditions = []
        
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
            self.feature_types,
            self.classes, 
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
        
        # 更新标签概率
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
            feat_name = self.feature_names[feat]
            if bound == "upper":
                conditions_str.append(f"{feat_name} <= {val:.3f}")
            else:
                conditions_str.append(f"{feat_name} > {val:.3f}")
        return " AND ".join(conditions_str)
    
    def __str__(self):
        return self.str_branch() + f" -> {self.label_probas}"

# 简化版ConjunctionSet类
class SimpleConjunctionSet:
    def __init__(self, feature_names, feature_types, trees, max_branches=100):
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.trees = trees
        self.max_branches = max_branches
        self.branches_lists = []
        self.conjunctionSet = None
        self.classes_ = None
        
        # 从所有树中提取类别信息
        all_classes = []
        for tree in trees:
            all_classes.extend(list(tree.classes_))
        self.classes_ = list(set(all_classes))
        
        # 生成规则
        self._generate_branches()
        self._build_conjunction_set()
    
    def _generate_branches(self):
        """从所有树中提取分支规则"""
        for tree_idx, tree in enumerate(self.trees):
            tree_branches = self._get_tree_branches(tree, tree_idx)
            self.branches_lists.append(tree_branches)
    
    def _get_tree_branches(self, tree, tree_idx):
        """从单棵树提取分支规则"""
        tree_ = tree.tree_
        leaf_indices = [
            i for i in range(tree_.node_count)
            if tree_.children_left[i] == -1 and tree_.children_right[i] == -1
        ]
        
        branches = []
        for leaf_idx in leaf_indices:
            branch = self._get_branch_from_leaf_index(tree_, leaf_idx)
            branches.append(branch)
        
        return branches
    
    def _get_branch_from_leaf_index(self, tree_, leaf_idx):
        """从叶子节点索引提取分支规则"""
        # 计算样本的标签概率
        sum_of_probas = np.sum(tree_.value[leaf_idx][0]) # shape: (n_classes,)
        label_probas = [i / sum_of_probas for i in tree_.value[leaf_idx][0]]
        
        # 创建新分支
        new_branch = SimpleBranch(
            self.feature_names,
            self.feature_types,
            self.classes_,
            label_probas=label_probas,
            number_of_samples=tree_.n_node_samples[leaf_idx]
        )
        
        # 从叶子节点回溯到根节点，提取条件
        node_id = leaf_idx
        while node_id > 0:  # 0是根节点，所以我们停在根节点前
            # 查找当前节点的父节点
            parent_indices = np.where(tree_.children_left == node_id)[0]
            bound = "upper"
            if len(parent_indices) == 0:
                bound = "lower"
                parent_indices = np.where(tree_.children_right == node_id)[0]
            
            parent_idx = parent_indices[0]
            feature = tree_.feature[parent_idx]
            threshold = tree_.threshold[parent_idx]
            
            # 添加条件
            new_branch.add_condition(feature, threshold, bound)
            
            # 移动到父节点
            node_id = parent_idx
        
        return new_branch
    
    def _build_conjunction_set(self):
        """构建和合并规则集"""
        if not self.branches_lists:
            return
        
        # 从第一棵树开始
        conjunction_set = self.branches_lists[0]
        
        # 依次合并其他树的规则
        for i, branch_list in enumerate(self.branches_lists[1:]):
            print(f"合并树 {i+2} 的规则，当前规则数: {len(conjunction_set)}")
            
            # 合并规则
            conjunction_set = self._merge_branch_with_conjunction_set(branch_list, conjunction_set)
            
            # 如果规则太多，进行过滤
            if len(conjunction_set) > self.max_branches:
                conjunction_set = self._filter_conjunction_set(conjunction_set)
                print(f"规则数量太多，过滤后剩余: {len(conjunction_set)}")
        
        # 移除重复规则
        # TODO: 应该在合并规则的时候，就移除重复的规则，这样过滤更高效
        self.conjunctionSet = self._remove_duplicate_branches(conjunction_set)
        print(f"最终规则数: {len(self.conjunctionSet)}")
    
    def _merge_branch_with_conjunction_set(self, branch_list, conjunction_set):
        """合并新的分支列表到已有的规则集"""
        new_conjunction_set = []
        
        # 尝试合并每对分支
        for b1 in conjunction_set:
            for b2 in branch_list:
                # 只合并不矛盾的分支
                if not b1.contradict_branch(b2):
                    new_branch = b1.merge_branch(b2)
                    new_conjunction_set.append(new_branch)
        
        return new_conjunction_set
    
    def _filter_conjunction_set(self, cs):
        """过滤规则集，只保留最重要的规则"""
        if len(cs) <= self.max_branches:
            return cs
        
        # 使用样本数量作为重要性度量
        branch_metrics = [b.number_of_samples for b in cs]
        threshold = sorted(branch_metrics, reverse=True)[self.max_branches - 1]
        
        return [b for b, metric in zip(cs, branch_metrics) if metric >= threshold][:self.max_branches]
    
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
    
    def get_rules(self):
        """获取所有规则的可读表示"""
        rules = []
        for i, branch in enumerate(self.conjunctionSet):
            rule_str = f"规则 {i+1}: 如果 {branch.str_branch()} 则 分类概率为 {branch.label_probas}"
            rules.append(rule_str)
        return rules


# 演示代码
def demo_conjunction_set():
    # 1. 创建样本数据集
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    feature_names = [f'特征{i+1}' for i in range(X.shape[1])]
    feature_types = ['numeric'] * len(feature_names)
    
    # 2. 划分数据集并训练多棵树
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 创建3个不同的决策树
    trees = []
    for i in range(3):
        # 使用不同的参数创建树
        tree = DecisionTreeClassifier(
            max_depth=3, 
            random_state=42+i,  # 不同的随机种子
            min_samples_split=20
        )
        tree.fit(X_train, y_train)
        trees.append(tree)
        print(f"树 {i+1} 训练完成，准确率: {tree.score(X_test, y_test):.4f}")
    
    # 3. 使用ConjunctionSet提取和合并规则
    print("\n开始提取和合并规则...")
    cs = SimpleConjunctionSet(feature_names, feature_types, trees, max_branches=50)
    
    # 4. 输出规则
    print("\n提取的规则:")
    for rule in cs.get_rules():
        print(rule)

# 运行演示
if __name__ == "__main__":
    demo_conjunction_set()