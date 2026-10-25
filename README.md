# IC-2 · Visão computacional para classificação de lesões de pele

**Python · PyTorch · transfer learning · EfficientNet · VGG16 · MobileNet · avaliação de modelos**

Pesquisa de **Enzo Souza**, na **PUC-Campinas**, sob orientação de **Prof. Dr. Ademar Takeo Akabane**, com período informado de **setembro/2024 a agosto/2025**. O relatório final compara **VGG16 e MobileNetV3-Large** para classificação binária de melanoma versus não melanoma. Os experimentos com EfficientNet foram preservados como exploração adicional e não são apresentados como o resultado central do relatório.

**Estado:** organização retrospectiva em outubro de 2026, complementada pelo relatório final. As matrizes de confusão e os percentuais reportados conferem com as saídas salvas de C51. Imagens, checkpoints e históricos JSON externos continuam ausentes; não houve execução dos modelos nem validação clínica.

## Explore o projeto

- [Mapa dos 14 notebooks e suas diferenças](docs/curation.md).
- [Experimentos, métricas e inconsistências encontradas](docs/experiments.md).
- [Demonstração de separação por lesão e avaliação](notebooks/01_data_and_metrics_demo.ipynb), sem baixar imagens.
- [Guia de nova execução](notebooks/02_training_guide.ipynb).
- [Proveniência e hashes dos arquivos de origem](docs/provenance.json).

## O que os experimentos exploram

- Transfer learning com VGG16, MobileNetV3-Large e EfficientNet-B0.
- Desbalanceamento de classes, ponderação da perda, oversampling e Focal Loss.
- Seleção de checkpoints por validação e avaliação com F1, AUC, sensibilidade e especificidade.
- Tentativa de exportação ONNX e rascunho de apresentação DermAI, com coautores preservados.

O rascunho DermAI também menciona metas de tamanho e latência. Os arquivos não demonstram que essas metas foram atingidas.

## Primeiros passos

Use Python 3.11 ou superior, a partir da raiz do repositório:

```sh
python -m venv .venv
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/validate_notebooks.py
```

Ative `.venv` antes da instalação (`.venv\Scripts\Activate.ps1` no PowerShell ou `source .venv/bin/activate` no Linux/macOS). O núcleo de manifestos e métricas usa somente a biblioteca padrão do Python.

Para uma nova execução com imagens obtidas de forma autorizada:

```sh
python -m pip install -e ".[train]"
python -m ic2.data --metadata data/raw/HAM10000_metadata.csv --image-root data/raw/images --output data/processed/split.csv
python -m ic2.train --manifest data/processed/split.csv --image-root data/raw/images --architecture efficientnet_b0 --epochs 25 --output runs/efficientnet_001
```

A preparação faz busca recursiva das imagens, exige `lesion_id` e mantém cada lesão em um único split. Rejeita imagens ausentes ou ambíguas, sem mover os arquivos. A estratificação é por alvo no nível de lesão, em proporções aproximadas de 70/15/15; não se promete separação por paciente.

O treino usa um logit binário, alvo positivo explícito `mel`, peso calculado somente no treino, transformações determinísticas na avaliação e seleção de limiar apenas na validação. Salva checkpoint, manifesto por hash, versões do ambiente, histórico, métricas globais e probabilidades individuais. Cada execução exige uma nova pasta. `--no-pretrained` evita baixar pesos ImageNet, mas mantém o backbone congelado; serve a verificações técnicas, não a uma comparação de desempenho.

**O fluxo novo não é réplica dos experimentos antigos:** padroniza as três arquiteturas com backbone congelado e um classificador binário. Resultados novos precisam ser identificados como novos. Não foi realizado treinamento nesta curadoria.

## Estrutura

```text
src/ic2/                   Manifesto, métricas e treinamento padronizado
notebooks/                 Demonstrações e guia de execução
notebooks/historical/      14 cadernos recuperados, sem imagens ou saídas embutidas
docs/                      Catálogo, análise e rastreabilidade
results/                   Evidências das saídas históricas
tests/                     Independência dos splits e correção das métricas
```

## Leitura dos resultados

O relatório final e as saídas de avaliação C51 concordam nos resultados abaixo. Foram conferidos os percentuais derivados das contagens; AUC e AP continuam como valores reportados, sem recálculo a partir de probabilidades individuais. Os resultados de outros checkpoints e de EfficientNet não foram misturados nesta comparação. [Conferência e limitações](docs/report_review.md).

| Modelo | Acurácia | Sensibilidade para melanoma | Especificidade | ROC-AUC reportada |
|---|---:|---:|---:|---:|
| VGG16 | 90,21% | 91,00% | 89,43% | 0,962 |
| MobileNetV3-Large | 88,87% | 86,28% | 91,43% | 0,960 |

Teste reportado: **1.788 imagens**, sendo 899 não melanoma e 889 melanoma. Não se comprovou separação por lesão/paciente nem desempenho em dados externos.

Os notebooks históricos podem conter operações sobre pastas e dependências externas. Leia-os como registros; use o fluxo novo para preparar dados. Imagens, pesos, credenciais e conteúdo de sessões Colab não foram publicados. [Autoria e direitos](RIGHTS.md).

## Conferir os resultados do relatório

```sh
python scripts/check_reported_results.py
```

O comando confere as matrizes de confusão e os percentuais do relatório sem carregar modelos. A apresentação em pôster em outubro de 2024 é informada na seção de atividades realizadas; o documento não comprova publicação de artigo. [Fontes e páginas](docs/report_sources.json).
