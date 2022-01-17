import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve, plot_roc_curve
import seaborn as sns
import tqdm
from terminaltables import AsciiTable
import os


class Evaluate(object):
    """Computes Accuracy, F1 Score, AUC Score per classess"""
    
    def __init__(self, classes):
        self.gts = []
        self.img_ids = []
        self.pred_probs = [] # 각 샘플 별 gt 클래스에 맞는 모델의 logit 확률값들
        self.preds = [] # 각 샘플 별 모델이 max 확률 값을 내뿜는 클래스
        self.classes = classes
        self.n_classes = len(classes)
        self.reset()
    
    def reset(self):
        self.confusion_matrices = []
        self.per_class_acc = []
        self.per_class_precision = []
        self.per_class_recall = []
        self.per_class_f1 = []
        self.roc_results = {}
    
    def accuracy(self, tp, tn, fp, fn):
        return (tp + tn) / (tp + tn + fp + fn)
    
    def precision(self, tp, tn, fp, fn):
        return tp / (tp + fp)
    
    def recall(self, tp, tn, fp, fn):
        return tp / (tp + fn)
    
    def f1_score(self, precision, recall):
        return 2*precision*recall/(precision+recall)

    def confusion_matrix(self, cls):
        """
        - TP, TN, FP, FN 갯수를 주어진 cls에 대한 binary classification으로 간주하고 진행
        - 반환값: TP, TN, FP, FN 갯수
        """
        cls_gt = np.array(self.gts) == cls
        cls_pred = np.array(self.preds) == cls

        tp = sum(cls_gt*cls_pred)
        tn = sum((cls_gt+cls_pred) == 0)
        fp = sum(cls_pred > cls_gt)
        fn = sum(cls_gt > cls_pred)

        assert sum([tp, fp, fn, tn]) == len(cls_pred)
        return tp, tn, fp, fn
    
    def confusion_matrix_thresh(self, cls, thresh):
        """
        - TP, TN, FP, FN 갯수를 주어진 cls에 대한 binary classification으로 간주하고 진행
        - 반환값: TP, TN, FP, FN 갯수
        """
        cls_gt = np.array(self.gts) == cls
        cls_pred = np.array(self.pred_probs)[:, cls] > thresh # 해당 class에 대한 확률 값이 threshold 이상인 경우, 해당 class를 positive하게 예측한 것으로 처리

        tp = sum(cls_gt*cls_pred)
        tn = sum((cls_gt+cls_pred) == 0)
        fp = sum(cls_pred > cls_gt) # 원래 0인데 1로 예측한 경우
        fn = sum(cls_gt > cls_pred) # 원래 1인데 0으로 예측한 경우

        assert sum([tp, fp, fn, tn]) == len(cls_pred)
        return tp, tn, fp, fn

    def draw_save_roc_curve(self, tprs, fprs, cls):
        cls_name = self.classes[cls]
        plot = sns.lineplot(x=fprs, y=tprs).set_title(cls_name.capitalize())
        fig = plot.get_figure()

        os.makedirs("./roc_curves", exist_ok=True)
        fig.savefig(f"./roc_curves/ROC_{cls_name.capitalize()}.png")
        plot.fig.clf()

    def calc_roc_curve(self):
        roc_values = {}
        for cls in tqdm.tqdm(range(self.n_classes)):
            tprs, tnrs, fprs = [], [], []
            tps, tns, fps, fns = [], [], [], []
            _, _, thresholds = roc_curve(np.array(self.gts)==cls, np.array(self.pred_probs)[:, cls])
            for thresh in thresholds:
                tp, tn, fp, fn = self.confusion_matrix_thresh(cls, thresh)
                tps.append(tp)
                tns.append(tn)
                fps.append(fp)
                fns.append(fn)
                tprs.append(tp / (tp+fn))
                tnrs.append(tn / (fp + tn))
                fprs.append(fp / (fp + tn))
            
            auc_val = auc(x=fprs, y=tprs)
            roc_values[cls] = {
                'tps': tps,
                'tns': tns,
                'fps': fps,
                'fns': fns,
                'tpr': tprs,
                'fpr': fprs,
                'tnr': tnrs,
                'thresh': thresholds,
                'auc': auc_val,
            }
        return roc_values

    def summarize_result(self):
        self.confusion_matrices = []
        self.per_class_acc = []
        self.per_class_precision = []
        self.per_class_recall = []
        self.per_class_f1 = []

        print("Summarizing results")
        for cls in tqdm.tqdm(range(self.n_classes)):
            tp, tn, fp, fn = self.confusion_matrix(cls)
            self.confusion_matrices.append([tp, tn, fp, fn])
            self.per_class_acc.append(self.accuracy(tp, tn, fp, fn))
            precision = self.precision(tp, tn, fp, fn)
            recall = self.recall(tp, tn, fp, fn)
            self.per_class_precision.append(precision)
            self.per_class_recall.append(recall)
            self.per_class_f1.append(self.f1_score(precision, recall))
        
        print("Summarizing roc curve results")
        self.roc_results = self.calc_roc_curve()
        for cls, roc_value in self.roc_results.items():
            self.draw_save_roc_curve(roc_value['tpr'], roc_value['fpr'], cls)

    def update(self, output, batch):
        with torch.no_grad():
            self.gts.extend(batch['gt'].cpu().numpy())
            self.img_ids.extend(batch['img_name'])
            self.pred_probs.extend(output.cpu().detach().numpy())
            self.preds.extend(torch.argmax(output, dim=1).cpu().detach().numpy())

    def print_acc_metric(self):
        print("\n\n =========  Accuracy Metric  =========")
        accuracy_table_data = [["Class ID", "TP", "TN", "FP", "FN", "Accuracy"]]
        # print("|   Class ID   |   TP   |   TN   |   FP   |   FN   |   Accuracy   |")
        for i, acc in enumerate(self.per_class_acc):
            tp, tn, fp, fn = self.confusion_matrices[i]
            accuracy_table_data.append([self.classes[i].capitalize(), tp, tn, fp, fn, round(acc, 2)])
        
        accuracy_table = AsciiTable(accuracy_table_data)
        print(accuracy_table.table)

        print("\nSummarize Accuracy: ")
        accuracy_summarize_data = [["Class ID", "Accuracy"]]
        # print("|   Class ID   |   Accuracy   |")
        for i, acc in enumerate(self.per_class_acc):
            accuracy_summarize_data.append([self.classes[i].capitalize(), round(acc, 2)])
            # print(f"|   {self.classes[i].capitalize()}   |   {acc}   |")
        accuracy_summarize_table = AsciiTable(accuracy_summarize_data)
        print(AsciiTable(accuracy_summarize_table.table))

    def print_f1_metric(self):
        print("\n\n =========  F1 Score Metric  =========")
        f1_table_data = [[ "Class ID",  "TP", "FP", "FN", "Precision", "Recall", "F1-Score" ]]
        # print("|   Class ID   |   TP   |   FP   |   FN   |   Precision   |   Recall   |   F1-Score   |")
        for i, (precision, recall, f1) in enumerate(zip(self.per_class_precision, self.per_class_recall, self.per_class_f1)):
            tp, _, fp, fn = self.confusion_matrices[i]
            # print(f"|   {self.classes[i].capitalize()}   |   {tp}   |   {fp}   |   {fn}   |   {precision}   |   {recall}   |   {f1}   |")
            f1_table_data.append([self.classes[i].capitalize(), tp, fp, fn, round(precision, 2), round(recall, 2), round(f1, 2)])

        f1_table = AsciiTable(f1_table_data)
        print(f1_table.table)

        f1_summary_data = [["Class ID", "F1 Score"]]
        # print("|   Class ID   |   F1 Score   |")
        for i, f1 in enumerate(self.per_class_f1):
            f1_summary_data.append([self.classes[i].capitalize(), round(f1, 2)])
            # print(f"|   {self.classes[i].capitalize()}   |   {f1}   |")
        
        f1_summary_table = AsciiTable(f1_summary_data)
        print("\nSummarize F1: ")
        print(f1_summary_table.table)


    def print_auc_metric(self):
        print("\n\n ===========  AUC Metric  ============")
        auc_data = [[ "Class ID", "TP", "TN", "FP", "FN", "P", "N", "Threshold", "TNR", "FPR", "TPR" ]]
        # print("|   Class ID   |   TP   |   TN   |   FP   |   FN   |   P   |   N   |   Threshold   |   TNR   |   FPR   |   TPR   |")
        for cls, roc_value in self.roc_results.items():
            for i, thresh in enumerate(roc_value['thresh']):
                tp = roc_value['tps'][i]
                tn = roc_value['tns'][i]
                fp = roc_value['fps'][i]
                fn = roc_value['fns'][i]
                tpr = roc_value['tpr'][i]
                fpr = roc_value['fpr'][i]
                tnr = roc_value['tnr'][i]
                auc_data.append([self.classes[cls].capitalize(), tp, tn, fp, fn, tp+fn, fp+tn, thresh, round(tnr, 2), round(fpr, 2), round(tpr, 2)])
                # print(f"|   {self.classes[cls].capitalize()}   |   {tp}   |   {tn}   |   {fp}   |   {fn}   |   {tp+fn}   |   {fp+tn}   |   {thresh}   |   {tnr}   |   {fpr}   |   {tpr}   |")

        auc_table = AsciiTable(auc_data)
        print(auc_table.table)

        print("\n Summarize AUC: ")
        auc_summary_data = [[ "Class ID", "AUC" ]]
        # print("|   Class ID   |   AUC   |")
        for cls, roc_value in self.roc_results.items():
            auc_summary_data.append([ self.classes[cls].capitalize(), round(roc_value['auc'],2)])
            # print(f"|   {self.classes[cls].capitalize()}   |   {roc_value['auc']}   |")
        auc_summary_table = AsciiTable(auc_summary_data)
        print(auc_summary_table.table)

    def summarize(self):
        self.reset()
        self.summarize_result()

        self.print_acc_metric()
        self.print_f1_metric()
        self.print_auc_metric()
    
