import sys
import os
import json
import numpy as np
from collections import defaultdict
from nltk.stem import WordNetLemmatizer

# 引入评估工具
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

class ScanQAEvaluator:
    def __init__(self, gt_path, use_spice=False):
        """
        初始化评估器：加载 GT 数据和评估模型
        :param gt_path: Ground Truth JSON 文件的路径
        :param use_spice: 是否使用 SPICE 指标 (速度较慢)
        """
        print(f"Initializing Evaluator with GT: {gt_path}...")
        
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"GT path not found: {gt_path}")
            
        with open(gt_path, 'r') as f:
            raw_gt = json.load(f)
            
        # 构建快速查找字典: {question_id: [answer1, answer2, ...]}
        # 注意：这里存储的是原始 ID (例如 "val-scene0011-0")
        self.gt_lookup = {item['question_id']: item['answers'] for item in raw_gt}
        
        # 初始化 Scorer
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]
        if use_spice:
            self.scorers.append((Spice(), "SPICE"))
            
        self.tokenizer = PTBTokenizer()
        self.lemmatizer = WordNetLemmatizer()
        print("Evaluator Ready.")

    def _get_lemma(self, ss):
        return [self.lemmatizer.lemmatize(token) for token in ss.split()]

    def _simple_ratio(self, numerator, denominator):
        num_numerator = sum([1 if token in numerator else 0 for token in denominator])
        num_denominator = len(denominator)
        return num_numerator / num_denominator if num_denominator > 0 else 0

    def _tokens_unigram_f_value(self, ref: str, pred: str) -> float:
        ref_lemma = self._get_lemma(ref)
        pred_lemma = self._get_lemma(pred)
        precision = self._simple_ratio(ref_lemma, pred_lemma)
        recall = self._simple_ratio(pred_lemma, ref_lemma)
        return 2 * (recall * precision) / (recall + precision) if recall + precision != 0. else 0

    def compute_metrics(self, inputs):
        """
        计算评估指标，智能识别输入格式。
        
        :param inputs: 
            格式 A (推荐): list of dicts, 例如:
                [{'sample_id': 'scanqa_val-scene...', 'pred_response': 'answer...'}, ...]
            格式 B: 单个 dict (单样本)
        :return: 包含各项指标的字典
        """
        
        # 1. 统一格式处理
        if isinstance(inputs, dict):
            inputs = [inputs] # 转为列表处理
        
        if not inputs or not isinstance(inputs, list):
            return {"Error": "Input must be a list of dictionaries or a single dictionary."}

        # 准备数据容器
        score_dict = defaultdict(list)
        gts_coco = {}
        res_coco = {}
        valid_count = 0

        # 2. 遍历预测数据
        for item in inputs:
            # --- 关键修改：解析 ID 和 Answer ---
            # 处理 ID 前缀 "scanqa_" 以匹配 GT
            raw_id = item.get('sample_id', '')
            qid = raw_id.replace("scanqa_", "") 
            
            pred_text = item.get('pred_response', '')

            # 检查 ID 是否存在于 GT 中
            if qid not in self.gt_lookup:
                # print(f"Warning: ID {qid} (raw: {raw_id}) not found in GT. Skipping.")
                continue
            
            valid_count += 1
            ref_answers = self.gt_lookup[qid]
            
            # --- A. 计算 EM 和 F-Value (规则匹配) ---
            # Top-1 EM
            # 注意：即使你的输入里已经做过 lower()，为了保险这里和 GT 比对时可以再处理一下，或者信任输入
            if pred_text in ref_answers:
                score_dict['Top1 (EM)'].append(1)
                score_dict['Top1 (F-value)'].append(1)
            else:
                f_scores = [self._tokens_unigram_f_value(pred_text, ref) for ref in ref_answers]
                score_dict['Top1 (EM)'].append(0)
                score_dict['Top1 (F-value)'].append(max(f_scores) if f_scores else 0)
            
            # --- B. 准备 COCO Metrics 数据 ---
            # 必须符合 {id: [{'caption': text}]} 格式
            gts_coco[qid] = [{'caption': ans} for ans in ref_answers]
            res_coco[qid] = [{'caption': pred_text}]

        if valid_count == 0:
            return {"Error": "No valid samples matched with Ground Truth."}

        # --- 3. 计算 Caption Metrics (BLEU, CIDEr等) ---
        # 使用 PTBTokenizer 进行标准分词 (这对长句子评分至关重要)
        try:
            gts_coco = self.tokenizer.tokenize(gts_coco)
            res_coco = self.tokenizer.tokenize(res_coco)
        except Exception as e:
            print(f"Tokenization Error: {e}")
            return {"Error": "Tokenization failed"}

        for scorer, method in self.scorers:
            # compute_score 返回 (平均分, 每个样本的分数列表)
            score, scores = scorer.compute_score(gts_coco, res_coco)
            
            if isinstance(method, list): # 处理 Bleu 返回列表的情况
                for m, s in zip(method, score):
                    score_dict[m].append(s * 100)
            else:
                score_dict[method].append(score * 100)

        # 4. 汇总结果
        final_metrics = {}
        
        # 对于 EM/F-Value，我们需要自己求平均
        final_metrics['Top1 (EM)'] = np.mean(score_dict['Top1 (EM)']) * 100
        final_metrics['Top1 (F-value)'] = np.mean(score_dict['Top1 (F-value)']) * 100
        
        # 对于 COCO Metrics，scorer 已经返回了加权平均值
        # 我们从 score_dict 中取出对应的值 (如果是 batch，取第一个元素即为平均值)
        for k, v in score_dict.items():
            if k not in ['Top1 (EM)', 'Top1 (F-value)']:
                final_metrics[k] = v[0] 

        return final_metrics

    # 配置你的 GT 路径
GT_PATH = './3D_QA/TStar/cdViews/data/qa/ScanQA/ScanQA_v1.0_val.json'
    
    # 1. 初始化 (只做一次)
evaluator = ScanQAEvaluator(gt_path=GT_PATH, use_spice=False)

    # # 2. 模拟你的数据格式 (List of Dicts)
    # # 假设这是模型跑出来的结果
    # my_pred_data = [
    #     {
    #         "sample_id": "scanqa_val-scene0011-0",  # 带有前缀
    #         "pred_response": "the chair in the kitchen is brown" # 长句子
    #     },
    #     {
    #         "sample_id": "scanqa_val-scene0011-1",
    #         "pred_response": "brown" # 短句子
    #     }
    # ]

    # print("\n--- Evaluating Pred Data ---")
    # # 直接传入你的列表
    # metrics = evaluator.compute_metrics(my_pred_data)
    
    # # 打印漂亮的 JSON 结果
    # print(json.dumps(metrics, indent=4))