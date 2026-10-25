# Validação da curadoria

Verificação local em 25/10/2026, Python 3.12.14.

- 11 testes automatizados passaram: split reprodutível e agrupado, detecção de sobreposição de lesões, imagens duplicadas/ausentes, rótulos inconsistentes, métricas conhecidas, AUROC com empates, lote unitário, agregação global, seleção de limiar na validação e duas verificações dos resultados do relatório.
- Os 16 notebooks passaram pela validação de estrutura; as células Python sem comandos mágicos passaram pela análise sintática.
- A demonstração sintética e a interface de criação do manifesto foram executadas e verificadas. A preparação não moveu as imagens de exemplo.
- As células dos 14 notebooks históricos foram comparadas com os originais: código e texto preservados, saídas e metadados de execução removidos.
- Links locais da documentação verificados; não foram encontrados padrões comuns de credenciais no conteúdo publicado.

O relatório final foi recebido na complementação. As matrizes de confusão e as Tabelas/figuras relevantes foram conferidas; seus percentuais foram recalculados e comparados ao PDF e aos outputs de C51. O script de conferência passou. AUC e AP não foram recalculadas, pois faltam probabilidades individuais.

O novo módulo de treinamento foi inspecionado e compilado, mas não executado. PyTorch, imagens, checkpoints e JSONs externos de histórico não estão disponíveis neste ambiente. Nenhuma métrica de treino novo foi inventada. O GitHub Actions testa o núcleo sem instalar a extensão de treinamento; não valida GPU, treino ou exportação ONNX. Consulte o PR para o status da execução remota.
