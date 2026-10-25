"""Manifestos reproduzíveis: separação por lesão antes de qualquer aumento de dados."""
import argparse
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

DIAGNOSES = {'mel', 'nv', 'bkl', 'bcc', 'akiec', 'vasc', 'df'}
SPLITS = ('train', 'val', 'test')


def split_metadata(rows, seed=91432, val_fraction=0.15, test_fraction=0.15):
    """Estratifica por alvo no nível de lesion_id; proporções aproximadas em grupos."""
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction+test_fraction < 1):
        raise ValueError('Proporções inválidas.')
    groups = defaultdict(list)
    image_ids = set()
    for row in rows:
        if not all(row.get(k) for k in ('image_id', 'lesion_id', 'dx')):
            raise ValueError('Metadados precisam de image_id, lesion_id e dx preenchidos.')
        if row['dx'] not in DIAGNOSES:
            raise ValueError(f'Diagnóstico desconhecido: {row["dx"]}')
        if row['image_id'] in image_ids:
            raise ValueError('image_id duplicado.')
        image_ids.add(row['image_id'])
        groups[row['lesion_id']].append(dict(row, label=int(row['dx']=='mel')))
    by_label = {0: [], 1: []}
    for lesion_id, items in groups.items():
        labels = {r['label'] for r in items}
        if len(labels) != 1:
            raise ValueError(f'Rótulos conflitantes para a lesão {lesion_id}.')
        by_label[items[0]['label']].append(lesion_id)
    rng = random.Random(seed)
    result = []
    for label in (0, 1):
        group_ids = sorted(by_label[label])
        if len(group_ids) < 3:
            raise ValueError('Cada classe precisa de pelo menos três lesões distintas.')
        rng.shuffle(group_ids)
        nval = max(1, round(len(group_ids)*val_fraction))
        ntest = max(1, round(len(group_ids)*test_fraction))
        if nval+ntest >= len(group_ids):
            raise ValueError('Grupos insuficientes para as proporções solicitadas.')
        for i, lesion_id in enumerate(group_ids):
            split = 'val' if i < nval else 'test' if i < nval+ntest else 'train'
            result.extend(dict(r, split=split) for r in groups[lesion_id])
    result.sort(key=lambda r: (r['split'], r['image_id']))
    validate_manifest(result)
    return result


def validate_manifest(rows):
    """Recusa vazamento por lesão, imagem duplicada e divisões sem ambas as classes."""
    seen, groups = set(), {}
    labels = {s:set() for s in SPLITS}
    for row in rows:
        if row['split'] not in SPLITS or int(row['label']) not in (0, 1):
            raise ValueError('Split ou label inválido.')
        if not row['image_id'] or not row['lesion_id'] or row['image_id'] in seen:
            raise ValueError('Identificador vazio ou imagem duplicada.')
        seen.add(row['image_id'])
        group = row['lesion_id']
        assignment = (row['split'], int(row['label']))
        if group in groups and groups[group] != assignment:
            raise ValueError('A mesma lesão aparece em splits ou classes diferentes.')
        groups[group] = assignment
        labels[row['split']].add(int(row['label']))
        if 'relative_path' in row:
            path = Path(row['relative_path'])
            if path.is_absolute() or '..' in path.parts or not row['relative_path']:
                raise ValueError('Caminho de imagem deve ser relativo à raiz dos dados.')
    if any(labels[s] != {0, 1} for s in SPLITS):
        raise ValueError('Cada split precisa conter ambas as classes.')


def attach_paths(rows, image_root):
    """Localiza imagens sem copiá-las, movê-las ou alterar suas pastas."""
    root = Path(image_root).resolve()
    index = {}
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            if path.stem in index:
                raise ValueError(f'Identificador de imagem ambíguo: {path.stem}')
            index[path.stem] = path.relative_to(root).as_posix()
    result = []
    for row in rows:
        if row['image_id'] not in index:
            raise ValueError(f'Imagem ausente: {row["image_id"]}')
        result.append(dict(row, relative_path=index[row['image_id']]))
    return result


def read_manifest(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    validate_manifest(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description='Cria manifesto HAM10000 agrupado por lesion_id.')
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=91432)
    args = parser.parse_args()
    provenance = args.output.with_suffix('.provenance.json')
    if args.output.exists() or provenance.exists():
        parser.error('Saída já existe; use um novo nome para preservar o manifesto anterior.')
    with args.metadata.open(encoding='utf-8-sig', newline='') as stream:
        rows = split_metadata(list(csv.DictReader(stream)), seed=args.seed)
    rows = attach_paths(rows, args.image_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ['image_id', 'lesion_id', 'dx', 'label', 'split', 'relative_path']
    with args.output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    summary = {'seed':args.seed, 'group_key':'lesion_id', 'positive_label':'mel',
               'metadata_sha256':hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
               'manifest_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),
               'images':{s:sum(r['split']==s for r in rows) for s in SPLITS},
               'lesions':{s:len({r['lesion_id'] for r in rows if r['split']==s}) for s in SPLITS},
               'note':'Proporções 70/15/15 aproximadas por lesão; não garantem separação por paciente.'}
    provenance.write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
