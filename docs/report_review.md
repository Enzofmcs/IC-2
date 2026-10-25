# Conferência do relatório final da IC-2

## Identificação e foco

O relatório “Estudo e Implementação de um Classificador de Lesão de Pele Utilizando a Arquitetura de Redes Neurais Convolucionais” identifica Enzo Souza, Ademar Takeo Akabane como supervisor, PUC-Campinas e o período setembro/2024–agosto/2025. A capa é datada de agosto/2025. As páginas citadas abaixo são as páginas físicas do PDF; a numeração impressa está entre parênteses.

O objetivo, na página 4 (2), e os resultados finais comparam VGG16 e MobileNetV3-Large. EfficientNet aparece nos notebooks adicionais, mas não é uma das duas arquiteturas centrais do relatório. O texto informa apresentação em pôster em 22 e 23 de outubro de 2024, na página 6 (4). A escrita de artigo figura como atividade prevista; não foi encontrada comprovação de artigo publicado.

## Partições e resultados

A Tabela 4.1, página 7 (5), registra 7.772 imagens de treino, 2.511 de validação e 1.788 de teste. O teste tem 899 não melanoma e 889 melanoma; a proporção real é a da tabela, não o novo split aproximado 70/15/15 do pacote.

O nome `ben` é um rótulo de pasta dos experimentos. O alvo foi documentado como melanoma versus não melanoma; esse agrupamento não permite apresentar o classificador como um teste geral de câncer versus ausência de câncer ou de benignidade clínica.

| Modelo | TN | FP | FN | TP | Acurácia conferida | Sensibilidade | Especificidade |
|---|---:|---:|---:|---:|---:|---:|---:|
| VGG16 | 804 | 95 | 80 | 809 | 90,21% | 91,00% | 89,43% |
| MobileNetV3-Large | 822 | 77 | 122 | 767 | 88,87% | 86,28% | 91,43% |

As matrizes e os percentuais aparecem nas páginas 10 (8) e 11 (9) e foram conferidos visualmente. Coincidem com os outputs salvos de `Summary.ipynb` (C51), células 28 e 22. O código `scripts/check_reported_results.py` recalcula acurácia, sensibilidade, especificidade, precisão, NPV e F1 a partir dessas contagens. Ele não executa o classificador.

ROC-AUC de VGG16 é reportada como 0,962 no PDF e 0,9615 no notebook; MobileNet como 0,960 e 0,9604. São compatíveis com o arredondamento. A métrica chamada PR-AUC é calculada por `average_precision_score` em C51 e as figuras a identificam como AP; o repositório preserva essa definição, sem reinterpretá-la como integração trapezoidal.

## Divergências documentadas

1. O parágrafo de VGG16 da página 10 diz “899 casos de melanoma”, mas 809+80 = **889**; a Tabela 4.1 e a matriz concordam em 889. Há 899 casos não melanoma. A conferência usa as contagens da matriz sem alterar o PDF original.
2. O resumo em markdown de C51 cita MobileNet com acurácia 0,8682 e AUC 0,9392. O relatório e os outputs concordam em **0,8887** e **0,9604**. O resultado principal do portfólio usa essa correspondência, preservando o histórico da divergência.
3. O relatório descreve CrossEntropyLoss para MobileNet, mas C02 contém BCELoss incompatível com seus dois logits. Sem o checkpoint e a versão de treino efetivamente usada, não se afirma que o C02 disponibilizado reproduza o modelo final.
4. O relatório afirma independência dos conjuntos, mas não fornece o manifesto por imagem/lesão nem a linhagem de augmentation. As contagens e a existência de pastas separadas não comprovam independência por lesão ou paciente. O novo fluxo agrupado é uma melhoria para execuções futuras, não uma descrição retroativa do protocolo original.
5. A capa data o relatório de agosto/2025, enquanto C51 contém timestamps de checkpoints de setembro/2025. Esses timestamps podem refletir cópia/modificação; não demonstram data de treino. A cronologia exata das versões não foi reconstruída.
6. As diferenças de latência, tamanho, consumo e custo de inferência são discutidas qualitativamente no relatório, mas não vieram benchmarks que permitam quantificá-las. Não foram adicionados números de desempenho operacional.

## Como apresentar o trabalho

A contribuição comprovada pelo documento é implementação e comparação de classificadores com transfer learning, análise do conjunto de dados, organização dos experimentos e avaliação de métricas. O resultado deve ser descrito como desempenho experimental reportado no teste utilizado. Não se infere utilidade diagnóstica ou superioridade absoluta de uma arquitetura.

Os dados completos de imagem, checkpoints e probabilidades individuais não foram fornecidos. As fontes e os hashes constam de `report_sources.json`; o PDF integral, com identificadores e contatos na capa, não foi redistribuído.
