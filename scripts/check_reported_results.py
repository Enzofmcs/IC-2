"""Confere contagens e percentuais do relatório; não recalcula AUC nem executa modelos."""
import json
from pathlib import Path


def calculate_from_counts(tn,fp,fn,tp):
    if any(not isinstance(v,int) or v<0 for v in (tn,fp,fn,tp)):
        raise ValueError('Contagens devem ser inteiros não negativos.')
    if not (tp+fn and tn+fp and tp+fp and tn+fn):
        raise ValueError('Matriz insuficiente para todas as métricas publicadas.')
    return {'n':tn+fp+fn+tp, 'negative_support':tn+fp, 'positive_support':fn+tp,
            'accuracy_percent':100*(tn+tp)/(tn+fp+fn+tp),
            'sensitivity_percent':100*tp/(tp+fn), 'specificity_percent':100*tn/(tn+fp),
            'precision_percent':100*tp/(tp+fp), 'npv_percent':100*tn/(tn+fn),
            'f1':2*tp/(2*tp+fp+fn)}


def verify(path):
    source=json.loads(Path(path).read_text(encoding='utf-8'))
    result=[]
    for model in source['models']:
        metrics=calculate_from_counts(**model['confusion_counts'])
        if metrics['n'] != 1788 or metrics['positive_support'] != 889 or metrics['negative_support'] != 899:
            raise ValueError('Contagens não conferem com a Tabela 4.1 do relatório.')
        for field,value in model['reported_percentages'].items():
            if round(metrics[field],2) != value:
                raise ValueError(f'{model["name"]}: {field} não confere com o relatório.')
        result.append({'model':model['name'],'computed_from_reported_counts':metrics,
                       'roc_auc_status':'reported_only; individual probabilities unavailable'})
    return result


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    print(json.dumps(verify(root/'results/final_report_results.json'),indent=2))
