#!/usr/bin/env python
# coding: utf-8

"""
简化版ICDTA4FL示例脚本，展示核心功能的使用和可视化
"""

import os
import numpy as np
from simplified_demo import (
    load_dataset, 
    split_data_to_clients, 
    train_local_model, 
    evaluate_trees_on_client, 
    filter_trees, 
    aggregate_rules, 
    evaluate_global_model
)
from utils import (
    visualize_decision_tree,
    visualize_rules,
    compare_models,
    print_tree_paths
)

# 创建输出目录
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

def main():
    # 配置参数
    n_clients = 3  # 客户端数量，使用较小的值方便演示
    model_params = {
        'max_depth': 3,  # 使用较小的深度便于可视化
        'criterion': 'gini',
        'splitter': 'best',
        'model_type': 'cart',
    }
    filter_params = {
        'filter_method': 'mean', 
        'acc_threshold': 0.6, 
        'f1_threshold': 0.5
    }
    
    print("=" * 80)
    print("ICDTA4FL 简化演示示例")
    print("=" * 80)
    
    # 1. 数据加载和分割
    print("\n步骤1: 加载数据")
    train_data, test_data, feature_names = load_dataset(categorical=False)
    print(f"数据集加载完成，特征数量: {len(feature_names)}")
    
    print("\n步骤2: 数据分发")
    client_data = split_data_to_clients(train_data, n_clients, iid=True)
    print(f"数据已分发给 {n_clients} 个客户端")
    
    # 2. 本地训练
    print("\n步骤3: 本地模型训练")
    client_models = []
    for i, data in enumerate(client_data):
        print(f"\n客户端 {i+1} 训练中...")
        model = train_local_model(data, model_params)
        client_models.append(model)
        
        # 可视化客户端决策树
        tree = model['local_tree']
        visualize_decision_tree(
            tree, 
            feature_names=feature_names, 
            save_path=os.path.join(output_dir, f"client_{i+1}_tree.png"),
            title=f"客户端 {i+1} 决策树"
        )
        
        # 打印决策路径
        print(f"\n客户端 {i+1} 决策树路径:")
        print_tree_paths(tree, feature_names=feature_names, max_paths=5)
        
        # 可视化规则
        branches_df = model['local_branches_df']
        visualize_rules(
            branches_df, 
            top_n=5, 
            save_path=os.path.join(output_dir, f"client_{i+1}_rules.png"),
            title=f"客户端 {i+1} 规则"
        )
    
    # 3. 规则评估和筛选
    print("\n步骤4: 规则评估和筛选")
    all_evaluations = []
    for i, client_model in enumerate(client_models):
        print(f"客户端 {i+1} 正在评估所有树...")
        eval_results = evaluate_trees_on_client(client_model, client_models)
        all_evaluations.append(eval_results)
    
    # 打印评估结果矩阵
    print("\n评估结果矩阵 (行=客户端数据, 列=客户端模型):")
    for i, evals in enumerate(all_evaluations):
        print(f"客户端 {i+1} 数据:")
        for j, (acc, f1) in enumerate(evals):
            print(f"  模型 {j+1}: 准确率={acc:.4f}, F1={f1:.4f}")
    
    # 筛选树
    selected_indices = filter_trees(all_evaluations, filter_params)
    print(f"筛选后选择了客户端模型: {[i+1 for i in selected_indices]}")
    
    # 4. 规则聚合
    print("\n步骤5: 规则聚合")
    global_model = aggregate_rules(client_models, selected_indices)
    print("全局模型构建完成")
    
    # 可视化全局模型规则
    branches_df = global_model[2]  # 全局模型的分支DataFrame
    visualize_rules(
        branches_df, 
        top_n=10,
        save_path=os.path.join(output_dir, "global_model_rules.png"),
        title="全局模型规则"
    )
    
    # 5. 评估全局模型
    print("\n步骤6: 评估全局模型")
    acc, f1 = evaluate_global_model(global_model, test_data)
    
    # 6. 比较模型性能
    print("\n步骤7: 比较模型性能")
    compare_models(
        client_models, 
        global_model, 
        {'acc': acc, 'f1': f1},
        title="客户端模型与全局模型性能比较"
    )
    
    print("\n全部步骤完成，图表已保存到 'output' 目录")
    print("=" * 80)

if __name__ == "__main__":
    main() 