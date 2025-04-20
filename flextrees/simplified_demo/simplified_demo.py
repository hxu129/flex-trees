#!/usr/bin/env python
# coding: utf-8

"""
简化版Demo: 可解释的客户端决策树聚合过程 (ICDTA4FL) 

这个脚本演示了ICDTA4FL流程的核心部分，主要关注于：
1. 本地决策树训练
2. 规则提取
3. 规则筛选
4. 规则聚合
5. 全局模型构建和评估
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# 为了简化，我们将从原始代码中抽取核心函数

# 添加这个新函数来打印决策树规则
def print_tree_rules(tree, feature_names=None, prefix="", branches_df=None):
    """递归打印决策树的规则"""
    if tree.left is None and tree.right is None:
        # 叶节点，打印类别分布
        if branches_df is not None and hasattr(tree, 'node_probas'):
            probas = tree.node_probas(branches_df)
        else:
            # 如果没有分支数据框或没有node_probas方法，尝试访问tree.label_probas
            probas = getattr(tree, 'label_probas', [0, 0])
            
        predicted_class = np.argmax(probas)
        print(f"{prefix}→ 预测类别: {predicted_class} (概率: {probas})")
        return
    
    # 获取特征名称
    feature = f"特征{tree.split_feature}" if feature_names is None else feature_names[tree.split_feature]
    
    # 打印左子树规则
    print(f"{prefix}如果 {feature} <= {tree.split_value}")
    print_tree_rules(tree.left, feature_names, prefix + "  ", branches_df)
    
    # 打印右子树规则
    print(f"{prefix}如果 {feature} > {tree.split_value}")
    print_tree_rules(tree.right, feature_names, prefix + "  ", branches_df)

# 添加这个函数来提取并打印ConjunctionSet规则
def print_conjunction_rules(cs, feature_names=None):
    """打印ConjunctionSet中的规则"""
    print("规则集中包含以下规则:")
    for i, branch in enumerate(cs.conjunctionSet):
        rule_str = f"规则 {i+1}: "
        
        # 添加上限条件 (features_upper)
        for feature, threshold in enumerate(branch.features_upper):
            if threshold != np.inf:  # 如果不是无穷大，表示有上限条件
                feature_name = f"特征{feature}" if feature_names is None else feature_names[feature]
                rule_str += f"{feature_name} <= {threshold:.4f} AND "
        
        # 添加下限条件 (features_lower)
        for feature, threshold in enumerate(branch.features_lower):
            if threshold != -np.inf:  # 如果不是负无穷大，表示有下限条件
                feature_name = f"特征{feature}" if feature_names is None else feature_names[feature]
                rule_str += f"{feature_name} > {threshold:.4f} AND "
        
        # 移除最后的 " AND "
        if rule_str.endswith(" AND "):
            rule_str = rule_str[:-5]
            
        # 添加预测类别
        predicted_class = np.argmax(branch.label_probas)
        rule_str += f" → 类别: {predicted_class} (概率: {branch.label_probas[predicted_class]:.4f})"
        
        print(rule_str)

# 1. 数据相关函数
def load_dataset(dataset_name='adult', categorical=False):
    """加载指定的数据集"""
    try:
        # 这里简化为手动加载adult数据集
        from flextrees.datasets import adult
        return adult(ret_feature_names=True, categorical=categorical)
    except ImportError:
        # 如果找不到特定的数据集，使用模拟数据
        print("未找到指定数据集，使用模拟数据")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
        feature_names = [f'x{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_df = pd.Series(y)
        
        # 创建数据集对象
        class Dataset:
            def __init__(self, X, y):
                self.X_data = X
                self.y_data = y
            def to_numpy(self):
                return self.X_data.to_numpy(), self.y_data.to_numpy()
                
        train_data = Dataset(X_df[:800], y_df[:800])
        test_data = Dataset(X_df[800:], y_df[800:])
        return train_data, test_data, feature_names

def split_data_to_clients(data, n_clients=5, iid=True):
    """将数据分割给多个客户端"""
    X, y = data.X_data.to_numpy(), data.y_data.to_numpy()
    client_data = []
    
    if iid:
        # IID分割: 随机均匀分割
        indices = np.random.permutation(len(X))
        chunk_size = len(indices) // n_clients
        
        for i in range(n_clients):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < n_clients - 1 else len(indices)
            client_indices = indices[start_idx:end_idx]
            
            # 创建数据集对象
            class ClientDataset:
                def __init__(self, X, y):
                    self.X_data = pd.DataFrame(X)
                    self.y_data = pd.Series(y)
            
            client_data.append(ClientDataset(X[client_indices], y[client_indices]))
    else:
        # Non-IID分割: 按类别偏向分配
        classes = np.unique(y)
        client_indices = [[] for _ in range(n_clients)]
        
        # 按类别分配
        for c in classes:
            idx = np.where(y == c)[0]
            np.random.shuffle(idx)
            
            # 偏向分配
            if len(classes) >= n_clients:
                # 如果类别数多于客户端数，每个客户端主要分配一种类型
                client_id = int(c % n_clients)
                client_indices[client_id].extend(idx[:int(len(idx)*0.6)])
                
                # 其余的随机分配
                remaining_idx = idx[int(len(idx)*0.6):]
                np.random.shuffle(remaining_idx)
                chunk_size = len(remaining_idx) // n_clients
                for i in range(n_clients):
                    start_idx = i * chunk_size
                    end_idx = (i + 1) * chunk_size if i < n_clients - 1 else len(remaining_idx)
                    client_indices[i].extend(remaining_idx[start_idx:end_idx])
            else:
                # 如果类别数少于客户端数，将每个类别平均分配
                chunk_size = len(idx) // n_clients
                for i in range(n_clients):
                    start_idx = i * chunk_size
                    end_idx = (i + 1) * chunk_size if i < n_clients - 1 else len(idx)
                    client_indices[i].extend(idx[start_idx:end_idx])
        
        # 创建数据集
        for indices in client_indices:
            class ClientDataset:
                def __init__(self, X, y):
                    self.X_data = pd.DataFrame(X)
                    self.y_data = pd.Series(y)
            
            client_data.append(ClientDataset(X[indices], y[indices]))
    
    return client_data

# 2. 树训练和规则提取函数
def train_local_model(client_data, model_params, random_state=42):
    """在本地训练决策树并提取规则"""
    # 根据模型类型创建分类器
    model_type = model_params.get('model_type', 'cart')
    
    # 创建分类器，这里简化只使用CART决策树
    clf = DecisionTreeClassifier(
        random_state=random_state,
        min_samples_split=max(1.0, int(0.02 * len(client_data.X_data))),
        max_depth=model_params.get('max_depth', 5),
        criterion=model_params.get('criterion', 'gini'),
        splitter=model_params.get('splitter', 'best')
    )
    
    # 准备训练数据
    X_data, y_data = client_data.X_data.to_numpy(), client_data.y_data.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    
    # 训练模型
    clf.fit(X_train, y_train)
    
    # 模型评估
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"客户端本地模型 - 准确率: {acc:.4f}, F1分数: {f1:.4f}, 训练样本数: {len(X_train)}, 测试样本数: {len(X_test)}")
    
    # 从决策树提取规则
    # 为了简化，这里我们直接使用原始代码中的ConjunctionSet类
    from flextrees.utils.ConjunctionSet import ConjunctionSet
    feature_names = [f'x{i}' for i in range(X_data.shape[1])]
    feature_types = ['int'] * len(feature_names)
    
    # 创建规则集合
    local_cs = ConjunctionSet(
        feature_names=feature_names, 
        original_data=X_train, 
        pruning_x=X_train, 
        pruning_y=y_train,
        model=[clf],  # 模型列表
        feature_types=feature_types,  # 特征类型
        amount_of_branches_threshold=3000,  # 分支数量阈值
        minimal_forest_size=1,  # 最小森林大小
        estimators=clf,  # 估计器
        filter_approach='probability',  # 过滤方法
        personalized=False  # 是否个性化
    )
    
    # 创建返回结果
    result = {
        'local_tree': clf,
        'local_cs': local_cs,
        'local_branches': local_cs.get_branches_list(),
        'local_branches_df': local_cs.get_conjunction_set_df().round(decimals=5),
        'local_classes': clf.classes_,
        'X_test': X_test,
        'y_test': y_test,
        'local_acc': acc,
        'local_f1': f1
    }
    
    return result

# 3. 规则评估和筛选函数
def evaluate_trees_on_client(client_model, all_models):
    """评估所有模型在当前客户端数据上的表现"""
    X_test, y_test = client_model['X_test'], client_model['y_test']
    eval_results = []
    
    for model in all_models:
        tree = model['local_tree']
        y_pred = tree.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        eval_results.append((acc, f1))
    
    return eval_results

def filter_trees(evaluation_results, filter_params):
    """基于评估结果筛选树"""
    # 计算平均性能
    avg_results = np.mean(evaluation_results, axis=0)
    
    # 根据筛选方法确定阈值
    filter_method = filter_params.get('filter_method', 'mean')
    if filter_method == 'mean':
        acc_threshold = np.mean(avg_results[:, 0])
        f1_threshold = np.mean(avg_results[:, 1])
    elif filter_method == 'percentile':
        percentile_value = filter_params.get('filter_value', 75)
        acc_threshold = np.percentile(avg_results[:, 0], percentile_value)
        f1_threshold = np.percentile(avg_results[:, 1], percentile_value)
    # else:
    #     # 默认使用固定阈值
    acc_threshold = filter_params.get('acc_threshold', 0.6)
    f1_threshold = filter_params.get('f1_threshold', 0.5)
    
    # 筛选满足条件的树索引
    selected_indices = []
    for i in range(len(avg_results)):
        if avg_results[i][0] >= acc_threshold and avg_results[i][1] >= f1_threshold:
            selected_indices.append(i)
    
    # 如果没有树被选中，选择表现最好的一棵
    if not selected_indices:
        best_idx = np.argmax(avg_results[0])  # 使用准确率选择
        selected_indices = [best_idx]
    
    print(f"筛选后选择了 {len(selected_indices)} 棵树，索引: {selected_indices}")
    return selected_indices

# 4. 规则聚合函数
def aggregate_rules(client_models, selected_indices):
    """聚合所选客户端的规则"""
    # 仅保留所选树的规则
    selected_models = [client_models[i] for i in selected_indices]
    
    # 提取规则和类别
    client_branches = [model['local_branches'] for model in selected_models]
    client_classes = [model['local_classes'] for model in selected_models]
    client_branches_df = [model['local_branches_df'] for model in selected_models]
    model_types = ['cart'] * len(selected_models)  # 简化为只使用CART
    
    # 使用utils_function_aggregator来聚合规则
    from flextrees.utils.utils_function_aggregator import generate_cs_dt_branches_from_list
    from flextrees.utils.branch_tree import TreeBranch
    
    # 准备输入格式
    list_of_weights = [(branches, classes, branches_df, model_type) 
                       for branches, classes, branches_df, model_type in 
                       zip(client_branches, client_classes, client_branches_df, model_types)]
    
    # 提取所有类别和特征
    classes_ = set()
    for client_class in client_classes:
        classes_ |= set(client_class)
    classes_ = list(classes_)
    
    # 提取分支列表
    client_cs = [cs for cs in client_branches]
    
    # 聚合为全局模型
    global_model = generate_cs_dt_branches_from_list(client_cs, classes_, TreeBranch)
    
    return global_model

# 5. 全局模型评估函数
def evaluate_global_model(global_model, test_data):
    """评估全局聚合模型的性能"""
    X_test, y_test = test_data.to_numpy()
    
    # 从全局模型中获取分支类别
    branches_df = global_model[2]
    classes_tree = get_classes_branches(branches_df)
    
    # 使用全局模型进行预测
    y_pred, _ = global_model[1].predict(X_test, classes_tree, branches_df)
    
    # 计算性能指标
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred)
    
    print("\n全局模型在测试集上的性能:")
    print(f"准确率: {acc:.4f}")
    print(f"宏平均F1: {f1:.4f}")
    print(f"分类报告: \n{report}")
    
    return acc, f1

def get_classes_branches(branches):
    """从分支DataFrame中获取类别"""
    assert branches is not None
    return list(range(len(branches['probas'].iloc[0])))

# 主函数
def main():
    # 配置参数
    N_CLIENTS = 2  # 客户端数量
    DATA_DISTRIBUTION = 'iid'  # 'iid' 或 'non-iid'
    MODEL_TYPE = 'cart'  # 模型类型，简化版只实现'cart'
    MAX_DEPTH = 2  # 决策树最大深度
    
    # 筛选参数
    FILTERING_METHOD = 'mean'  # 筛选方法
    ACC_THRESHOLD = 0.6  # 准确率阈值
    F1_THRESHOLD = 0.5  # F1分数阈值
    
    # 1. 加载数据
    print(f"\n加载数据集...")
    train_data, test_data, feature_names = load_dataset(categorical=False)
    print(f"数据集加载完成，特征数量: {len(feature_names)}")
    
    # 2. 将数据分发到客户端
    print(f"\n将数据分发到 {N_CLIENTS} 个客户端 ({DATA_DISTRIBUTION} 分布)...")
    client_data = split_data_to_clients(train_data, N_CLIENTS, DATA_DISTRIBUTION == 'iid')
    
    # 3. 配置本地模型参数
    local_model_params = {
        'max_depth': MAX_DEPTH,
        'criterion': 'gini',
        'splitter': 'best',
        'model_type': MODEL_TYPE,
    }
    
    # 4. 在每个客户端训练本地模型并提取规则
    print("\n第1步: 训练本地模型并提取规则...")
    client_models = []
    for i, data in enumerate(client_data):
        print(f"\n客户端 {i+1} 训练中...")
        model = train_local_model(data, local_model_params, random_state=i)
        client_models.append(model)
    print("第1步: 本地模型训练完成，规则已提取")
    
    # 打印每个客户端的决策树规则
    print("\n== 打印每个客户端的决策树规则 ==")
    for i, client_model in enumerate(client_models):
        print(f"\n客户端 {i+1} 的决策树规则:")
        local_tree = client_model['local_tree']
        feature_names = [f'特征{i}' for i in range(len(client_data[i].X_data.columns))]
        
        # 打印CART决策树规则（使用sklearn的文本表示）
        print("CART决策树规则:")
        try:
            from sklearn import tree as sk_tree
            print(sk_tree.export_text(local_tree, feature_names=feature_names))
        except Exception as e:
            print(f"无法打印决策树文本表示: {e}")
        
        # 打印规则集
        print(f"\n客户端 {i+1} 的规则集:")
        print_conjunction_rules(client_model['local_cs'], feature_names=feature_names)
    
    # 5. 评估所有客户端上的所有树
    print("\n第2-5步: 筛选弱决策树...")
    all_evaluations = []
    for i, client_model in enumerate(client_models):
        print(f"客户端 {i+1} 正在评估所有树...")
        eval_results = evaluate_trees_on_client(client_model, client_models)
        all_evaluations.append(eval_results)
    
    # 6. 筛选表现好的树
    filter_params = {
        'filter_method': FILTERING_METHOD,
        'acc_threshold': ACC_THRESHOLD, 
        'f1_threshold': F1_THRESHOLD
    }
    
    selected_indices = filter_trees(all_evaluations, filter_params)
    
    # 7. 聚合规则并构建全局模型
    print("\n第6-9步: 聚合规则并构建全局模型...")
    global_model = aggregate_rules(client_models, selected_indices)
    print("全局模型构建完成")
    
    # 打印全局树规则
    print("\n== 打印全局树规则 ==")
    # 全局树是global_model[1]，ConjunctionSet是global_model[0]，branches_df是global_model[2]
    global_tree = global_model[1]
    global_cs = global_model[0]
    branches_df = global_model[2]
    feature_names = [f'特征{i}' for i in range(len(client_data[0].X_data.columns))]
    
    print("全局决策树规则:")
    print_tree_rules(global_tree, feature_names=feature_names, branches_df=branches_df)
    
    print("\n全局规则集:")
    print_conjunction_rules(global_cs, feature_names=feature_names)
    
    # 8. 评估全局模型
    print("\n第10步: 评估全局模型...")
    eval_results = evaluate_global_model(global_model, test_data)
    
    print("\n--- ICDTA4FL 简化Demo完成 ---")

if __name__ == "__main__":
    main() 