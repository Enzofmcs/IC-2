"""Nova execução comparável entre arquiteturas; não reproduz métricas históricas por si só."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random

from .data import read_manifest
from .metrics import binary_metrics, select_threshold


def build_model(architecture, pretrained=True):
    import torch.nn as nn
    from torchvision import models
    choices = {
        'efficientnet_b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
        'mobilenet_v3_large': (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V1),
        'vgg16': (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1),
    }
    constructor, weights = choices[architecture]
    model = constructor(weights=weights if pretrained else None)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    last = model.classifier[-1]
    model.classifier[-1] = nn.Linear(last.in_features, 1)
    return model, weights.transforms()


def run(args):
    import numpy as np
    import torch
    from PIL import Image
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    import torchvision

    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError('epochs, batch-size e patience precisam ser positivos.')
    if args.output.exists():
        raise ValueError('Diretório de saída já existe; use um novo nome por experimento.')
    rows = read_manifest(args.manifest)
    root = args.image_root.resolve()
    for row in rows:
        path = (root/row['relative_path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Imagem ausente ou fora da raiz: {row["image_id"]}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, evaluation_transform = build_model(args.architecture, not args.no_pretrained)
    model.to(device)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), evaluation_transform,
    ])

    class Images(Dataset):
        def __init__(self, split):
            self.rows = [r for r in rows if r['split']==split]
            self.transform = train_transform if split=='train' else evaluation_transform

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            with Image.open(root/row['relative_path']) as image:
                tensor = self.transform(image.convert('RGB'))
            return tensor, int(row['label'])

    generator = torch.Generator().manual_seed(args.seed)
    loaders = {s:DataLoader(Images(s), batch_size=args.batch_size, shuffle=s=='train',
                           num_workers=0, generator=generator if s=='train' else None)
               for s in ('train','val','test')}
    train_labels = [int(r['label']) for r in rows if r['split']=='train']
    pos_weight = (len(train_labels)-sum(train_labels))/sum(train_labels)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight],device=device))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=1e-4, weight_decay=1e-4)

    def epoch(loader, training):
        model.train(training)
        if training:
            model.features.eval()  # Backbone congelado: não altera estatísticas de BatchNorm.
        labels, probabilities = [], []
        loss_sum, count = 0.0, 0
        with torch.set_grad_enabled(training):
            for images, targets in loader:
                images, targets = images.to(device), targets.to(device)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                logits = model(images).reshape(-1)
                loss = loss_fn(logits, targets.float())
                if not torch.isfinite(loss):
                    raise ValueError('Perda não finita; interrompido sem publicar resultado.')
                if training:
                    loss.backward()
                    optimizer.step()
                loss_sum += loss.item()*len(targets)
                count += len(targets)
                labels.extend(targets.detach().cpu().tolist())
                probabilities.extend(torch.sigmoid(logits).detach().cpu().tolist())
        return loss_sum/count, labels, probabilities

    args.output.mkdir(parents=True)
    checkpoint = args.output/'best.pt'
    history, best_f1, stale = [], -1.0, 0
    configuration = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    configuration.update(positive_label='mel', negative_label='non_melanoma',
                         output_logits=1, backbone='frozen', pos_weight=pos_weight,
                         manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                         torch_version=str(torch.__version__), torchvision_version=str(torchvision.__version__))
    for index in range(args.epochs):
        train_loss, train_y, train_p = epoch(loaders['train'],True)
        val_loss, val_y, val_p = epoch(loaders['val'],False)
        train_metrics, val_metrics = binary_metrics(train_y,train_p), binary_metrics(val_y,val_p)
        record = {'epoch':index+1, 'train_loss':train_loss, 'val_loss':val_loss,
                  'train':train_metrics, 'val':val_metrics}
        history.append(record)
        print(json.dumps(record,allow_nan=False))
        if val_metrics['f1'] > best_f1+1e-6:
            best_f1, stale = val_metrics['f1'], 0
            torch.save({'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        'configuration':configuration, 'epoch':index+1},checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break
    saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(saved['state_dict'])
    model.to(device)
    _, val_y, val_p = epoch(loaders['val'],False)
    threshold = select_threshold(val_y,val_p)
    # O teste só é consultado após fixar checkpoint e limiar na validação.
    test_loss, test_y, test_p = epoch(loaders['test'],False)
    report = {'configuration':configuration, 'best_epoch':saved['epoch'],
              'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'threshold_selected_on':'validation', 'validation':binary_metrics(val_y,val_p,threshold),
              'test_loss':test_loss, 'test':binary_metrics(test_y,test_p,threshold),
              'status':'new_run_not_historical_replication'}
    for name,value in [('metrics.json',report),('history.json',history)]:
        (args.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (args.output/'test_predictions.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['image_id','label','probability_melanoma'])
        writer.writerows((r['image_id'],y,p) for r,y,p in zip(loaders['test'].dataset.rows,test_y,test_p))
    print(json.dumps(report,indent=2,allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--image-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--architecture',choices=['efficientnet_b0','mobilenet_v3_large','vgg16'],default='efficientnet_b0')
    parser.add_argument('--epochs',type=int,default=25)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--patience',type=int,default=5)
    parser.add_argument('--seed',type=int,default=91432)
    parser.add_argument('--no-pretrained',action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
