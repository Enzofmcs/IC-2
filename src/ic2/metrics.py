"""Métricas binárias globais, com melanoma explicitamente definido como classe 1."""
import math


def _validate(labels, probabilities, threshold):
    labels, probabilities = list(labels), list(probabilities)
    if not labels or len(labels) != len(probabilities):
        raise ValueError('Labels e probabilidades devem ter o mesmo tamanho não nulo.')
    if any(y not in (0, 1) for y in labels):
        raise ValueError('Labels devem ser 0 (não melanoma) ou 1 (melanoma).')
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Limiar fora de [0, 1].')
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError('Probabilidades devem ser finitas e pertencer a [0, 1].')
    return labels, probabilities


def binary_metrics(labels, probabilities, threshold=0.5):
    labels, probabilities = _validate(labels, probabilities, threshold)
    predictions = [int(p >= threshold) for p in probabilities]
    tn = sum(y==0 and p==0 for y,p in zip(labels,predictions))
    fp = sum(y==0 and p==1 for y,p in zip(labels,predictions))
    fn = sum(y==1 and p==0 for y,p in zip(labels,predictions))
    tp = sum(y==1 and p==1 for y,p in zip(labels,predictions))
    sensitivity = tp/(tp+fn) if tp+fn else None
    specificity = tn/(tn+fp) if tn+fp else None
    precision = tp/(tp+fp) if tp+fp else 0.0
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
    # AUROC por ranks, com empates recebendo rank médio.
    order = sorted(zip(probabilities, labels))
    positive_rank_sum, i = 0.0, 0
    while i < len(order):
        j = i+1
        while j < len(order) and order[j][0] == order[i][0]:
            j += 1
        positive_rank_sum += ((i+1+j)/2)*sum(label for _,label in order[i:j])
        i = j
    npos, nneg = sum(labels), len(labels)-sum(labels)
    auc = (positive_rank_sum - npos*(npos+1)/2)/(npos*nneg) if npos and nneg else None
    return {'n':len(labels), 'threshold':threshold, 'positive_class':'melanoma',
            'confusion_matrix':[[tn,fp],[fn,tp]], 'accuracy':(tn+tp)/len(labels),
            'precision':precision, 'sensitivity':sensitivity, 'specificity':specificity,
            'balanced_accuracy':(sensitivity+specificity)/2 if npos and nneg else None,
            'f1':f1, 'roc_auc':auc,
            'brier_score':sum((p-y)**2 for y,p in zip(labels,probabilities))/len(labels)}


def select_threshold(validation_labels, validation_probabilities):
    """Maximiza F1 na validação; em empate, prefere limiar mais próximo de 0,5."""
    labels, probabilities = _validate(validation_labels, validation_probabilities, 0.5)
    if set(labels) != {0, 1}:
        raise ValueError('A validação deve conter ambas as classes.')
    candidates = sorted({0.0, 0.5, 1.0, *probabilities})
    return max(candidates, key=lambda t:(binary_metrics(labels,probabilities,t)['f1'], -abs(t-0.5), -t))
