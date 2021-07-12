from pathlib import Path
from functools import wraps
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.utils import to_categorical
import pycm
import wandb

"""
Metrics
=======
"""

CLASS_METRICS = [
    "TP", "FP", "TN", "FN",
    "PPV", # Precision
    "TPR", # Recall
    "ACC", # Accuracy
    "F1"   # F1 score
]

OVERALL_METRICS = [
    "ACC Macro", # - Accuracy Macro
    "ERR Macro", # - Error Macro
    "PPV Macro", # - Precision Macro
    "TNR Macro", # - Specificity Macro
    "TPR Macro", # - Recall Macro
    "F1 Macro",  # - F1-Score Macro
    "Kappa"      # - Kappa
]

def _filter_and_validate_stats(stats, metrics):
    filtered_stats = {}
    for m in metrics:
        if isinstance(stats[m], dict):
            filtered_stats[m] = { 
                k:v if v != "None" else 0.0 for k,v in stats[m].items()
            }
        else:
            filtered_stats[m] = stats[m] if stats[m] != "None" else 0.0
    return filtered_stats


def classification_metrics_by_class(y_true, y_pred, labels=None, flatten=True, metrics=CLASS_METRICS):
    """get classification metrics by class for binary and multi-class.
    metrics:
        "TP", "FP", "TN", "FN",
        "PPV", # Precision
        "TPR", # Recall
        "ACC", # Accuracy
        "F1"   # F1 score
    """
    cm = pycm.ConfusionMatrix(actual_vector=y_true, predict_vector=y_pred)
    if labels is not None:
        #if isinstance(labels, list) or isinstance(labels, np.ndarray):
        if not isinstance(labels, dict):
            labels = {i:str(l) for i, l in enumerate(labels)}
        cm.relabel(mapping=labels)

    stats = _filter_and_validate_stats(cm.class_stat, metrics)
    if flatten:
        flatten_stats = {}
        for metric, by_class in stats.items(): 
            flatten_stats.update({f"{metric} - {c}": m for c, m in by_class.items()})
        stats = flatten_stats
    return stats

def overall_classification_metrics(y_true, y_pred, labels=None, metrics=OVERALL_METRICS):
    """get overall classification metrics for multi-class.
    metrics:
        "ACC Macro", # - Accuracy Macro
        "ERR Macro", # - Error Macro
        "PPV Macro", # - Precision Macro
        "TNR Macro", # - Specificity Macro
        "TPR Macro", # - Recall Macro
        "F1 Macro",  # - F1-Score Macro
        "Kappa"      # - Kappa
    """
    cm = pycm.ConfusionMatrix(actual_vector=y_true, predict_vector=y_pred)
    if labels is not None:
        #if isinstance(labels, list):
        if not isinstance(labels, dict):
            labels = {i:str(l) for i, l in enumerate(labels)}
        cm.relabel(mapping=labels)
    stats = cm.overall_stat
    #add error rate
    stats["ERR Macro"] = sum(cm.ERR.values())/len(cm.ERR)
    stats = _filter_and_validate_stats(stats, metrics)
    return stats

def get_metrics(y_true, y_prob, metrics, labels=None, num_classes=None, run_id=None):
    num_classes = num_classes or y_prob.shape[-1]
    y_pred = y_prob.argmax(axis=-1) if num_classes > 2 else np.round(y_prob).astype(int)
    all_metrics = {}
    tables_metrics = {}
    base_columns = [] if run_id is None else ['Fold']
    base_data = [] if run_id is None else [run_id]
    if "overall_classification_metrics" in metrics:
        overall_metrics = overall_classification_metrics(y_true, y_pred, labels=labels)
        all_metrics.update(overall_metrics)
        tables_metrics["overall_metrics_table"] = wandb.Table(
            columns=base_columns + list(overall_metrics.keys()),
            data=[base_data + list(overall_metrics.values())]
        )
    if "classification_metrics_by_class" in metrics:
        metrics_by_class = classification_metrics_by_class(y_true, y_pred, labels=labels)
        all_metrics.update(metrics_by_class)
        tables_metrics["by_class_metrics_table"] = wandb.Table(
            columns=base_columns + list(metrics_by_class.keys()),
            data=[base_data + list(metrics_by_class.values())]
        )
    return all_metrics, tables_metrics

"""
Plots
=====
"""

def save_or_show_plot(plot_name):
    """
    Add save/show 
    """
    def real_decorator(plot_func):
        @wraps(plot_func)
        def wrapper(*args, **kwargs):
            save_to = kwargs.get("save_to", None)
            model_name=kwargs.get("model_name", None)
            fig = plot_func(*args, **kwargs)
            if save_to is not None:
                save_to = Path(save_to)
                fig.savefig(save_to / f'{model_name}_{plot_name}.png')
            show = kwargs.get("show", False)
            if show:
                plt.show()
                plt.close()
                return None
            else:
                return fig
        return wrapper
    return real_decorator


def confusion_matrix(y_true, y_pred, labels=None):
    cm = pycm.ConfusionMatrix(actual_vector=y_true, predict_vector=y_pred)
    if labels:
        if isinstance(labels, list):
            labels = {i:str(l) for i, l in enumerate(labels)}
        cm.relabel(mapping=labels)
    return cm

@save_or_show_plot("Confussion Matrix")
def plot_confusion_matrix(cm, model_name=None, labels=None, cmap=None, normalize=True, *args, **kwargs):
    """
    given a pycm confusion matrix (cm), make a nice plot
    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix
    labels: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']
    title:        the text to display at the top of the matrix
    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    Return
    ------
        figure or None
    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by pycm
                          normalize    = True,                # show proportions
                          labels = y_labels_vals,             # list of names of the classes
                          title        = best_estimator_name) # title of graph
    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    accuracy = cm.ACC_Macro 
    kappa = cm.Kappa
    labels = labels or cm.classes
    cm = cm.to_array()
    if normalize:
        group_counts = [f"{value}"
                        for value in cm.flatten()]
        group_percentages = [f"{100*value:.1f}"
                             for value in cm.flatten()/np.sum(cm)]
        annot = [f"{v1}\n{v2}%"
                 for v1, v2 in zip(group_counts, group_percentages)]
        annot = np.array(annot).reshape(cm.shape)
        fmt =''
    else:
        fmt ='d'
        annot=True
    # Heatmap
    fig = plt.figure(figsize=(5, 4))
    cmap = cmap or plt.get_cmap('Blues')  
    sns.heatmap(cm, annot=annot, cmap=cmap, fmt=fmt, 
                xticklabels=labels, yticklabels=labels)
    # Style
    if model_name is not None:
        title = f"{model_name}\nConfusion Matrix"
    else:
        title = f"Confusion Matrix"
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel(f'Predicted label\n\naccuracy={accuracy:0.4f}; Kappa={kappa:0.4f}')
    plt.grid(False)
    plt.tight_layout()    
    return fig

def roc_curve_by_class(y_true, y_prob, num_classes=None):
    #m_samples, n_classes = y_prob.shape
    num_classes = num_classes or (y_prob.shape[-1] if y_prob.ndim > 1 else 2)
    y_true = to_categorical(y_true, num_classes=num_classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        if num_classes == 2:
            fpr[i], tpr[i], _ = roc_curve((1 - y_true[:]), (1 - y_prob[:])) if i == 0 else  roc_curve(y_true[:], y_prob[:])
        else:
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc

@save_or_show_plot("ROC Curve")
def plot_roc_curve_by_class(fpr, tpr, roc_auc, model_name=None, labels=None):
    """
    given a sklearn roc curves, make a nice plot
    Arguments
    ---------
    fpr, tpr, roc_auc : 
    labels: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']
    title:        the text to display at the top of the matrix
    Return
    ------
        figure or None
    Usage
    -----
    display_multiclass_roc_curve(...)
    Citiation
    ---------
    https://stackoverflow.com/questions/50941223/plotting-roc-curve-with-multiple-classes
    """
    n_classes = len(fpr)
    labels = labels or range(n_classes)
    # ROC Curves
    fig = plt.figure(figsize=(5, 4))
    for i in range(n_classes):
        plt.plot( fpr[i], tpr[i], 
            label=f'ROC curve of class {labels[i]} (AUC = {roc_auc[i]:0.2f})')
    # Style
    if model_name is not None:
        title = f"{model_name}\nReceiver Operating Characteristic"
    else:
        title = f"Receiver Operating Characteristic"
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(title)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.tight_layout()
    return fig

def get_plots(X_true, y_true, y_prob, plots, labels=None, num_classes=None, wandb_plot=False):
    y_pred = y_prob.argmax(axis=-1) if num_classes > 2 else np.round(y_prob).astype(int)
    all_plots = {}
    if 'cm' in plots:
        if wandb_plot:
            raise NotImplementedError
        else:
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            fig_cm = plot_confusion_matrix(cm, labels=labels)
            all_plots['cm'] = wandb.Image(fig_cm)
            plt.close(fig_cm)
    if 'pr' in plots:
        if wandb_plot:
            #all_plots['pr'] = wandb.plots.precision_recall(y_true, y_prob, labels)
            raise NotImplementedError
        else:
            fpr, tpr, roc_auc = roc_curve_by_class(y_true, y_prob, num_classes=num_classes)
            fig_pr = plot_roc_curve_by_class(fpr, tpr, roc_auc, labels=labels)
            all_plots['pr'] = wandb.Image(fig_pr)
            plt.close(fig_pr)
    return all_plots

"""
Highlights
==========
"""

def save_or_show_sample(plot_name):
    """
    Add save/show 
    """
    def real_decorator(plot_func):
        @wraps(plot_func)
        def wrapper(*args, **kwargs):
            save_to = kwargs.get("save_to", None)
            model_name=kwargs.get("model_name", None)
            index=kwargs.get("i", None)
            fig = plot_func(*args, **kwargs)
            if save_to is not None:
                save_to = Path(save_to)
                fig.savefig(save_to / f'{model_name}_{plot_name}_{index}.png')
            show = kwargs.get("show", False)
            if show:
                plt.show()
                plt.close()
                return None
            else:
                return fig
        return wrapper
    return real_decorator


@save_or_show_sample("Prediction")
def plot_prediction(Xi, *args, **kwargs):
    Xi = Xi.squeeze()
    ndim = Xi.ndim
    if ndim == 1:
        # Signal 1D
        fig = plt.figure(figsize=(5, 4))
        plt.plot(Xi)
        #plt.ylim([-1.0, 1.0])
        plt.xlabel('$\Delta t$', fontsize=16)
        plt.ylabel('Amplitude', fontsize=16)
        plt.tight_layout()
        plt.grid(True)
        return fig
    else:
        raise NotImplementedError("Xi ndim > 1")
    

def get_highlights(X, y_true, y_prob, y_absolute_index, labels, top_k=10):
    # Sort by performance
    cross_entropy = -1*np.log(y_prob[np.arange(len(y_true)), y_true] + np.finfo(np.float32).eps)
    sorted_index = np.argsort(cross_entropy)
    # Select highlights
    if y_absolute_index is None:
        y_absolute_index = np.arange(len(y_true))
    # Table data
    ## id
    id_top_best, id_top_worst = list(y_absolute_index[sorted_index[:top_k]]), list(y_absolute_index[sorted_index[-top_k:]])
    ## target
    y_true_top_best, y_true_top_worst = y_true[sorted_index[:top_k]], y_true[sorted_index[-top_k:]]
    target_top_best, target_top_worst = [labels[yi] for yi in y_true_top_best], [labels[yi] for yi in y_true_top_worst]
    ## prediction
    y_pred_top_best, y_pred_top_worst = y_prob.argmax(axis=-1)[sorted_index[:top_k]], y_prob.argmax(axis=-1)[sorted_index[-top_k:]]
    prediction_top_best, prediction_top_worst = [labels[yi] for yi in y_pred_top_best], [labels[yi] for yi in y_pred_top_worst]
    ## X
    X_top_best, X_top_worst = X[sorted_index[:top_k]], X[sorted_index[-top_k:]]
    X_plot_top_best = []
    for Xi in X_top_best:
        fig = plot_prediction(Xi=Xi) 
        X_plot_top_best.append(wandb.Image(fig))
        plt.close(fig)
    X_plot_top_worst = []
    for Xi in X_top_worst:
        fig = plot_prediction(Xi=Xi) 
        X_plot_top_worst.append(wandb.Image(fig))
        plt.close(fig)
    ## Cross-Entropy
    cross_entropy_top_best, cross_entropy_top_worst = list(cross_entropy[sorted_index[:top_k]]), list(cross_entropy[sorted_index[-top_k:]])
    # Buid Tables
    columns=["id", "X", "target", "prediction", "cross_entropy"]
    transpose_data_best = [
        id_top_best, X_plot_top_best, target_top_best, prediction_top_best, cross_entropy_top_best
    ]
    data_best = [list(row) for row in zip(*transpose_data_best)] #transpose
    transpose_data_worst = [
        id_top_worst, X_plot_top_worst, target_top_worst, prediction_top_worst, cross_entropy_top_worst
    ]
    data_worst = [list(row) for row in zip(*transpose_data_worst)] #transpose
    return {
        "top_best": wandb.Table(columns=columns, data=data_best),
        "top_worst": wandb.Table(columns=columns, data=data_worst)
    }
