# Atividade K-NN — Dupla 08

Atividade da disciplina de **Aprendizado de Máquina** — FATEC Jundiaí.

Classificação binária (`nivel_alto` × `nivel_baixo`) com K-NN a partir de 9 atributos numéricos anônimos (`indice_01` a `indice_09`). O projeto investiga, passo a passo, como escalonamento, valor de k, métrica de distância, tipo de votação e atributos irrelevantes afetam o desempenho, e termina em um modelo final salvo em disco.

## Resultado

| | Acurácia | F1 macro |
|---|---|---|
| Baseline (k = 5, sem escalonamento) — teste | 0,613 | 0,613 |
| **Modelo final — teste** | **0,819** | **0,819** |
| Modelo final — treino | 0,824 | 0,823 |
| Modelo final — validação cruzada (5 dobras, treino) | — | 0,782 ± 0,023 |

**Configuração final:** `MinMaxScaler` (ajustado só no treino) + K-NN com k = 15, distância euclidiana (Minkowski, p = 2) e votação uniforme.

**Como ler esses números:** o teste tem apenas 155 observações e algumas escolhas (escalonador, k, distância, votação) consideraram o desempenho nele. A estimativa de ~0,82 deve ser lida como ordem de grandeza. A melhora sobre o baseline é grande; as diferenças entre as configurações finalistas (k = 15 uniforme, k = 15 ponderada, k = 7 Manhattan) são pequenas e ficam dentro do desvio da validação cruzada.

## Estrutura

```
Atividade_KNN_dupla_08/
├── data/
│   ├── raw/dupla_08.csv          # dados originais (620 linhas, 9 atributos + classe)
│   └── processed/                # arquivos gerados pelos notebooks
│       ├── train.csv, test.csv           # divisão 75/25 estratificada (465 / 155)
│       ├── baseline.csv                  # métricas do baseline
│       ├── escalonamento.csv             # A (original), B (MinMax), C (Standard), k = 5
│       ├── k_teste.csv, k_cv.csv         # resultados por k (teste e validação cruzada)
│       ├── votacao.csv, votacao_cv.csv   # votação uniforme × ponderada
│       └── modelo_final.csv              # métricas do modelo final
├── models/
│   └── modelo_final.joblib       # pipeline treinado (MinMaxScaler + K-NN)
├── notebooks/                    # análise, na ordem de execução
└── src/
    └── knn_utils.py              # funções de apoio (métricas, escalonamento, validação cruzada)
```

## Notebooks (rodar na ordem)

Cada notebook lê arquivos gerados pelos anteriores, então a ordem importa.

| # | Notebook | O que faz |
|---|---|---|
| 01 | `01_conhecimento_dos_dados` | Estrutura, alvo, escala dos atributos, correlações e PCA. `indice_03` responde por ~99,8% da distância quadrática. |
| 02 | `02_preparacao_e_baseline` | Divisão estratificada em treino/teste e baseline (k = 5, sem escala): F1 0,613. |
| 03 | `03_escalonamento` | Compara sem escala, MinMax e Standard; demonstra o vazamento de dados ao ajustar o escalonador antes de dividir. |
| 04 | `04_hiperparametro_k` | Varia k de 1 a 31, no teste e em validação cruzada. |
| 05 | `05_distancia_e_votacao` | Manhattan × Euclidiana e votação uniforme × ponderada. |
| 06 | `06_generalizacao_e_dimensionalidade` | Compara três configurações candidatas e mede o efeito de 10 atributos aleatórios. |
| 07 | `07_modelo_final` | Treina o modelo final, avalia, compara com o baseline, conclui e salva o modelo. |

## Principais conclusões

- **Escalonamento é decisivo.** Sem ele, `indice_03` domina a distância. Com k = 5, o F1 de teste vai de 0,613 (sem escala) para 0,800 (MinMax) e 0,768 (Standard).
- **Vazamento de dados.** Ajustar o escalonador com todos os dados antes de dividir usa informação do teste. Aqui o efeito foi pequeno (MinMax: 0,800 → 0,819; Standard: sem mudança), mas o procedimento correto é ajustar só no treino, e a validação cruzada usa `Pipeline` por isso.
- **k.** k = 1 memoriza o treino (F1 de treino 1,0, gap 0,24); a partir de k = 31 o desempenho cai. Entre k = 3 e k = 21 a validação cruzada é praticamente igual.
- **Votação ponderada** chega a F1 de treino 1,0 (cada ponto é vizinho de si mesmo) e não melhora na validação cruzada.
- **Atributos irrelevantes prejudicam.** Com 10 atributos aleatórios, o F1 de teste cai de 0,819 para 0,626, e a razão entre a distância ao vizinho mais próximo e a distância média sobe de 0,31 para 0,54.
- **Sobreposição das classes** limita o desempenho; não se espera acurácia próxima de 100%.

## Como executar

Requisitos: Python 3.12 e as bibliotecas `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` e `joblib`.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
cd notebooks
jupyter notebook
```

Execute os notebooks de 01 a 07. Os caminhos são relativos (`../data/...`), então abra-os a partir da pasta `notebooks`. O notebook 02 cria `data/processed/` e o 07 cria `models/`.

## Usando o modelo salvo

O arquivo contém o pipeline completo; o escalonamento já está incluído, então os dados novos entram sem transformação prévia.

```python
import joblib
import pandas as pd

modelo = joblib.load("models/modelo_final.joblib")
novos = pd.read_csv("novos_dados.csv")[[f"indice_0{i}" for i in range(1, 10)]]
modelo.predict(novos)
```

O modelo deve ser carregado com a mesma versão do `scikit-learn` usada para salvá-lo. Os atributos precisam estar na mesma ordem e na mesma unidade dos dados de treino; valores fora da faixa vista no treino podem passar de 0–1 depois do `MinMaxScaler` e devem ser tratados com cuidado.

## Reprodutibilidade

- Divisão treino/teste: `random_state = 42`, `stratify = y`.
- Validação cruzada: `StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)`, sempre só no conjunto de treino.
- Atributos aleatórios do experimento de dimensionalidade: `np.random.default_rng(42)`.
