# Catálogo dos notebooks históricos

A sequência abaixo é uma **ordem de leitura proposta**, inferida pelo conteúdo. Não é uma cronologia comprovada dos experimentos. Os códigos A/B/C distinguem as três exportações fornecidas.

| ID | Arquivo organizado | Conteúdo e diferenças | Limitações |
|---|---|---|---|
| A12 | [01_ham10000_exploration](../notebooks/historical/01_ham10000_exploration.ipynb) | Exploração das classes, idade, sexo e localização no HAM10000. | Metadados externos necessários; várias visualizações alternativas. |
| A08 | [02_preparation_attempt](../notebooks/historical/02_preparation_attempt.ipynb) | Seleção de imagens de melanoma e tentativa de preparação. | Erro salvo: customDataset não disponível; apesar do nome, não implementa aumento de dados. |
| C02 | [03_mobilenet_training](../notebooks/historical/03_mobilenet_training.ipynb) | MobileNetV3-Large com duas saídas e fine-tuning parcial. | BCELoss incompatível com logits de duas saídas e alvos escalares; não há treino completo salvo. Não usar como reprodução do checkpoint avaliado em Summary. |
| C00 | [04_mobilenet_onnx](../notebooks/historical/04_mobilenet_onnx.ipynb) | Tentativa de exportação de checkpoint MobileNet para ONNX. | Checkpoint ausente; configuração de exportação precisa ser validada. Nenhum ONNX foi incluído no material. |
| C67 | [05_efficientnet_baseline](../notebooks/historical/05_efficientnet_baseline.ipynb) | EfficientNet-B0 com uma saída, BCEWithLogitsLoss e 10 épocas. | Registra avaliação e erro na previsão de amostras; média de acurácia por lote difere da agregação global. |
| C65 | [06_efficientnet_f1](../notebooks/historical/06_efficientnet_f1.ipynb) | EfficientNet-B0 com peso positivo e seleção por F1 de melanoma. | Caminhos conflitantes no mesmo notebook; resumo diz 50 épocas, log para antes. Resultado histórico não reproduzido. |
| C57 | [07_vgg16_training](../notebooks/historical/07_vgg16_training.ipynb) | VGG16 com duas saídas, label smoothing, sampler e early stopping. | Congelamento indexa parâmetros, não blocos; métricas por lote e relatório global divergem. |
| C13 | [08_weighted_focal](../notebooks/historical/08_weighted_focal.ipynb) | EfficientNet com sampler ponderado e Focal Loss. | Variação metodológica; sem resultado final salvo que permita afirmar superioridade. |
| C07 | [09_focal_mixup](../notebooks/historical/09_focal_mixup.ipynb) | EfficientNet com Focal Loss, MixUp, OneCycleLR e busca de limiar. | Split estratificado por imagem; independência por lesão não demonstrada; logs de falha de multiprocessamento. |
| C52 | [10_partial_finetuning](../notebooks/historical/10_partial_finetuning.ipynb) | EfficientNet com descongelamento parcial e até 150 épocas. | Treino interrompido, mas há saída de teste; pos_weight e mapeamento precisam de revisão. |
| C53 | [11_oversampling_initial](../notebooks/historical/11_oversampling_initial.ipynb) | Tentativa de oversampling com separação por metadados. | Erro salvo de conjunto vazio; procura imagens na raiz, enquanto a estrutura baixada usa subpastas. |
| C29 | [12_oversampling_revision](../notebooks/historical/12_oversampling_revision.ipynb) | Revisão com busca recursiva de imagens e uma saída binária. | Mantém outras inconsistências; a busca recursiva diferencia esta versão de C53. Não declarar resultado final sem reexecutar. |
| C51 | [13_checkpoint_comparison](../notebooks/historical/13_checkpoint_comparison.ipynb) | Avaliação de checkpoints com AUC, sensibilidade, especificidade e limiar. | Resumo narrativo contradiz saída MobileNet; células dependem de variáveis anteriores e checkpoints ausentes. |
| C56 | [14_dermai_group_draft](../notebooks/historical/14_dermai_group_draft.ipynb) | Rascunho relacionado DermAI com contexto e funções de avaliação. | Notebook identifica Enzo Souza, Vinicius Hardy e Angelo Pereira. Metas de AUC/latência/tamanho são objetivos, não resultados demonstrados. |

## Critérios de seleção

Foram inspecionadas 104 entradas dos três ZIPs. Arquivos vazios, exercícios de disciplinas e projetos sem relação demonstrada com estas pesquisas ficaram fora dos repositórios. Arquivos de nomes iguais em exportações diferentes não foram tratados automaticamente como duplicatas. Hashes e origem permitem rastrear a seleção.

As saídas embutidas, imagens e metadados de sessão foram removidos das cópias públicas. Nenhuma execução histórica foi simulada ou reconstituída. Trechos de métricas efetivamente salvos foram transcritos em um arquivo de evidência separado.

## Materiais relacionados que ficaram fora

- `A06` (`003-dataaugmentation.ipynb`): variante que move arquivos de volta para pastas de origem antes de copiar; não realiza augmentation. O contexto útil está documentado, sem oferecer esse fluxo como preparação recomendada.
- `C25` (`Untitled(1)`, sem extensão): move imagens entre treino, validação e teste, envia imagens para uma pasta trash e registra contagens inconsistentes. Foi substituído pelo manifesto por lesão; não há como confirmar sobreposição histórica sem as imagens e a linhagem dos aumentos.
- `C66` (`binary_classification.ipynb`): protótipo VGG com exclusão/reestruturação de pastas e vários erros de dependência. O experimento VGG mais completo é C57.
- `C03`: apenas imports, sem treinamento, apesar do nome EfficientNet.
- `C42`: template parcialmente preenchido, com falha de leitura do HAM10000 e sobreposição com variantes posteriores.
- `C17`, `C68`, `A02`, `C46`, `C55`: estudos de redes, MNIST/FashionMNIST e/ou quantização, sem demonstração de experimento de lesões de pele. Não apresentados como resultado da IC.
