# Experimentos e avaliação das evidências

## Famílias de experimentos

| Família | Fontes | Diferença principal |
|---|---|---|
| Exploração e preparação | A12, A08 | Metadados HAM10000 e seleção da classe mel; A08 falha por customDataset ausente |
| MobileNet | C02, C00 | Treinamento com duas saídas; tentativa posterior de exportação ONNX |
| EfficientNet com pastas pré-divididas | C67, C65 | Baseline de 10 épocas versus ponderação e seleção por F1 |
| VGG16 | C57 | Duas saídas, CrossEntropyLoss, sampler, label smoothing e early stopping |
| EfficientNet com metadados | C13, C07, C52, C53, C29 | Focal Loss, MixUp, fine-tuning parcial e oversampling em combinações diferentes |
| Avaliação de checkpoints | C51 | Métricas por arquitetura, curvas e ajuste de limiar |
| Apresentação relacionada | C56 | Rascunho DermAI em grupo; metas e metodologia, sem comprovação de todas as metas |

A ordem é de leitura, não cronológica. O nome do arquivo ou a ordem de exportação não identifica por si só a versão final.

## Resultados históricos recuperados

`results/historical_evidence.json` contém trechos de **outputs salvos**, com ID e número da célula no original. Os notebooks publicados receberam uma célula inicial de curadoria; a numeração de evidência refere-se à fonte.

| Fonte | Saída registrada | Ressalva |
|---|---|---|
| C67, EfficientNet baseline | Test Accuracy 91,3664%; resumo 91,37%; teste com 1.788 imagens | Acurácia média por lote; não equivale necessariamente à acurácia global do relatório |
| C65, EfficientNet por F1 | Test Accuracy 91,29%; F1 de validação 0,9052; teste com 1.788 imagens | Mistura caminhos; resumo de 50 épocas contradiz parada no log; não é repetição de C67 |
| C57, VGG16 | Test Acc 88,84%; classification report 0,8881 | Média por lote versus contagem global; não substituir um valor pelo outro sem explicação |
| C51, MobileNet, célula 22 | Accuracy 0,8887; ROC-AUC 0,9604 | Resumo narrativo em markdown cita 0,8682 e 0,9392, divergindo do output |
| C51, VGG16, célula 28 | Accuracy 0,9021; ROC-AUC 0,9615 | Checkpoint/limiar de avaliação; não corresponde automaticamente ao treino salvo em C57 |
| C52, EfficientNet parcial | Test Accuracy 91,89%; teste com 1.503 imagens | Há KeyboardInterrupt no notebook; dataset e processo de seleção diferem das séries com 1.788 imagens |

Não calcular uma média nem escolher o maior valor dessa tabela como resultado final da IC. Sem checkpoints, manifesto e dados, a reprodução e comparabilidade não estão estabelecidas.

## Inconsistências técnicas encontradas

1. **Função de perda e saída:** C02 produz dois logits, mas usa BCELoss e alvos escalares. C00 tenta carregar checkpoint de duas saídas; não se pode afirmar que ele veio do treino exibido em C02. O fluxo novo usa sempre um logit + BCEWithLogitsLoss + sigmoid.
2. **Métricas por lote:** C57, C65 e C67 somam percentuais de acurácia de lotes e dividem pelo número de lotes. O lote final menor recebe o mesmo peso. A versão nova calcula métricas sobre todas as amostras e perda ponderada pelo tamanho do lote.
3. **Dimensões na inferência:** C67 aplica softmax/squeeze em uma saída binária; há IndexError salvo. O fluxo novo mantém probabilidades de classe positiva com `reshape(-1)` e sigmoid, inclusive com lote unitário.
4. **Classe positiva:** C65 procura a string `melanoma`, embora a estrutura use `mel`, e recorre ao índice 1. O fluxo novo define explicitamente `mel=1` e não depende da ordem alfabética de pastas.
5. **Separação dos dados:** variantes com metadados usam train_test_split estratificado por imagem, sem agrupar lesion_id. As pastas antigas foram modificadas em C25, inclusive usando imagens de teste para compor validação; esse procedimento não prova por si só sobreposição, mas impede reconstruir uma avaliação independente sem a linhagem dos arquivos.
6. **Augmentation e balanceamento:** imagens aumentadas já presentes em pastas não têm rastreabilidade por original nos ZIPs. Faça a divisão por lesão antes de augmentation e aplique mudanças aleatórias somente no treino. Não combine indiscriminadamente oversampling, sampler e peso da perda.
7. **Congelamento:** C57 enumera parâmetros para congelar, embora o comentário se refira a blocos de camadas; a implementação não garante o limite de bloco descrito. O fluxo novo congela explicitamente todo `features` e mantém seu BatchNorm em modo de avaliação.
8. **Reprodutibilidade:** caminhos de Drive sobrescritos, helper_functions baixado de branch mutável, variáveis de células anteriores e checkpoints ausentes dificultam a reprodução. O fluxo novo não importa código auxiliar remoto e registra configuração e hashes.

## Nova execução

Obtenha os metadados e imagens pelos canais oficiais do dataset de acordo com suas condições de uso. Os notebooks de origem referenciam HAM10000, o espelho Kaggle `kmader/skin-cancer-mnist-ham10000` e uma versão aumentada `utkarshps/skin-cancer-mnist10000-ham-augmented-dataset`; não se assume que essas fontes sejam intercambiáveis. Os arquivos externos não foram baixados nesta curadoria.

Use as imagens originais para gerar o manifesto. O novo split agrupa `lesion_id`, preserva ambas as classes e registra proporções efetivas; sem patient_id, não garante independência por paciente. Verifique também duplicatas e múltiplas imagens derivadas de uma mesma fonte antes de uma avaliação final.

Para comparar arquiteturas, reutilize **o mesmo manifesto**, orçamento de treino, política de pré-processamento e critério de seleção. Fixe checkpoint e limiar na validação antes de consultar o teste. O novo treinamento seleciona checkpoint por F1 a limiar 0,5 e depois otimiza o limiar na validação; essa política é diferente do Youden usado em partes de C51 e deve ser relatada.

O pacote usa seed e configuração determinística do cuDNN, mas não promete igualdade bit a bit entre equipamentos. Os pesos ImageNet podem exigir download. `metrics.json` registra versões e hashes. Não usa pesos históricos sem correspondência explícita de arquitetura, pré-processamento, rótulos e partição.

## Limites desta entrega

Validaram-se a estrutura dos notebooks, as funções de manifesto/métricas e as demonstrações sintéticas. O treinamento foi inspecionado e compilado, mas não executado: este ambiente não contém PyTorch, as imagens ou os checkpoints. Não se afirma desempenho de generalização, utilidade diagnóstica, latência ou tamanho de modelo implantado.
