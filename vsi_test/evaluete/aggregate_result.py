import pandas as pd
import logging
from collections import OrderedDict
import numpy as np
import json
# ==========================================
# 1. 配置区域
# ==========================================

# 图片中左侧：Numerical Answer (使用 MRA 指标)
NA_QUESTION_TYPES = [
    "object_counting",
    "object_abs_distance",
    "object_size_estimation",
    "room_size_estimation",
]

# 图片中右侧：Multiple-Choice Answer (使用 Accuracy 指标)
MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance", # 注意：根据图片，Rel.Dist 在右侧 MCA 区域
    "route_planning",
    "obj_appearance_order",
]

# 指标定义
METRICS_FOR_MCA = {"accuracy": "exact_match"}
METRICS_FOR_NA = {"MRA:.5:.95:.05": "mean_relative_accuracy"}

# 模拟 Logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
eval_logger = logging.getLogger("eval_logger")

# ==========================================
# 2. 严格对应图片的输出顺序配置
# ==========================================
# 格式: (任务名称, 使用的指标后缀)
OUTPUT_COLUMN_ORDER = [
    # --- Numerical Answer (左侧) ---
    ("object_counting", "MRA:.5:.95:.05"),        # Obj. Count
    ("object_abs_distance", "MRA:.5:.95:.05"),    # Abs. Dist.
    ("object_size_estimation", "MRA:.5:.95:.05"), # Obj. Size
    ("room_size_estimation", "MRA:.5:.95:.05"),   # Room Size
    
    # --- Multiple-Choice Answer (右侧) ---
    ("object_rel_distance", "accuracy"),          # Rel. Dist.
    ("object_rel_direction", "accuracy"),         # Rel. Dir. (聚合后的分数)
    ("route_planning", "accuracy"),               # Route Plan
    ("obj_appearance_order", "accuracy"),         # Appr. Order
]




# path = './3D_QA/Spatial-MLLM-master/eval_results/eval_vsibench/qwen3b_keyvt/results_qwen3b_keyvt.json'

# root_path = [path.split('/')[i] for i in range(len(path.split('/'))-1)]
# root_path = '/'.join(root_path) + '/'
# print(root_path)
# with open(path,'r') as f:
#     results_json = json.load(f)
def vsibench_aggregate_results(results,root_path):
    df = pd.DataFrame(results)
    temp_scores = {}

    # 1. 分组计算原始分数 (MRA 和 Accuracy)
    if not df.empty and 'question_type' in df.columns:
        for question_type, group in df.groupby('question_type'):
            if question_type in MCA_QUESTION_TYPES:
                if 'accuracy' in group.columns:
                    temp_scores[f"{question_type}_accuracy"] = group['accuracy'].mean()
            elif question_type in NA_QUESTION_TYPES:
                if 'MRA:.5:.95:.05' in group.columns:
                    metric_key = 'MRA:.5:.95:.05'
                    temp_scores[f"{question_type}_{metric_key}"] = group[metric_key].mean()

    # 2. 聚合 Direction 分数
    direction_subtasks = [
        'object_rel_direction_easy_accuracy',
        'object_rel_direction_medium_accuracy',
        'object_rel_direction_hard_accuracy'
    ]
    direction_scores = [temp_scores.get(task) for task in direction_subtasks if task in temp_scores]
    if direction_scores:
        temp_scores['object_rel_direction_accuracy'] = sum(direction_scores) / len(direction_scores)

    # 3. 准备最终列表中的具体任务分（用于计算 Overall）
    task_keys = []
    for task_name, metric_suffix in OUTPUT_COLUMN_ORDER:
        key = f"{task_name}_{metric_suffix}"
        task_keys.append(key)
    
    # 计算 Overall (仅计算 OUTPUT_COLUMN_ORDER 中存在的指标)
    valid_scores = [temp_scores[k] for k in task_keys if k in temp_scores]
    overall_val = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    # 4. 构造最终字典：Overall 放在第一位
    final_output = OrderedDict()
    final_output['overall'] = overall_val # 强制置顶
    
    for key in task_keys:
        final_output[key] = temp_scores.get(key, 0.0)

    # 5. 写入 JSON
    save_path = root_path + 'detail.json'
    with open(save_path, 'w') as f:
        json.dump(final_output, f, indent=4)
    
    # 6. 打印 LaTeX 格式：用 & 分割，乘以 100，保留一位小数
    formatted_values = [f"{val * 100:.1f}" for val in final_output.values()]
    print("\n" + "="*20 + " LaTeX Row " + "="*20)
    
    
    latex_row =" & ".join(formatted_values)
    print(latex_row)
    
    print("="*51 + "\n")
        
    return final_output,latex_row

def preprocess_data(json_data):
    """
    将嵌套的 JSON 列表转换为 vsibench_aggregate_results 所需的扁平列表。
    """
    flat_data = []
    for item in json_data['results']:
        # 提取 question_type，这是 groupby 的关键
        q_type = item['sample']['original_question_type']
        
        row = {
            "question_type": q_type,
            # 必须包含 METRICS_FOR_NA 中的 key
            "MRA:.5:.95:.05": float(item.get('reward', 0.0)),
            # 必须包含 METRICS_FOR_MCA 中的 key
            "accuracy": 1.0 if item.get('correct') else 0.0
        }
        flat_data.append(row)
    return flat_data


def agg_data(path):
    root_path = [path.split('/')[i] for i in range(len(path.split('/'))-1)]
    root_path = '/'.join(root_path) + '/'
    with open(path,'r') as f:
        results_json = json.load(f)
    test_data = preprocess_data(results_json)
    # 运行
    res,latex_row = vsibench_aggregate_results(test_data,root_path)
    
    return res,latex_row



