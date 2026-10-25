"""Valida estrutura e compilação das células sem executar código de pesquisa."""
import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
count = 0
for path in sorted((root/'notebooks').rglob('*.ipynb')):
    nb = json.loads(path.read_text(encoding='utf-8'))
    assert nb['nbformat'] == 4, path
    ids = set()
    for cell in nb['cells']:
        assert cell['cell_type'] in ('code', 'markdown', 'raw'), path
        assert cell['id'] not in ids, path
        ids.add(cell['id'])
        if cell['cell_type'] != 'code':
            continue
        assert cell.get('outputs') == [] and cell.get('execution_count') is None, path
        text = cell['source']
        text = ''.join(text) if isinstance(text, list) else text
        if not any(line.lstrip().startswith(('!', '%')) for line in text.splitlines()):
            ast.parse(text, filename=str(path))
    count += 1
print(f'{count} notebooks válidos estruturalmente; células não executadas.')
