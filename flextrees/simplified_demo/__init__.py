"""
ICDTA4FL 简化版实现
(Interpretable Client Decision Tree Aggregation for Federated Learning)

此模块包含ICDTA4FL算法的简化实现，专注于规则提取和聚合的核心流程。
"""

from .simplified_demo import (
    load_dataset,
    split_data_to_clients,
    train_local_model,
    evaluate_trees_on_client,
    filter_trees,
    aggregate_rules,
    evaluate_global_model,
    get_classes_branches
)

from .utils import (
    visualize_decision_tree,
    visualize_rules,
    compare_models,
    print_tree_paths
)

__all__ = [
    # 核心算法函数
    'load_dataset',
    'split_data_to_clients',
    'train_local_model',
    'evaluate_trees_on_client',
    'filter_trees',
    'aggregate_rules',
    'evaluate_global_model',
    'get_classes_branches',
    
    # 可视化和工具函数
    'visualize_decision_tree',
    'visualize_rules',
    'compare_models',
    'print_tree_paths'
] 