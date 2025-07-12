# Análise, pré-processamento e escolha de classificadores em uma base de dados misteriosa

### **Abstract:** Este trabalho aborda a classificação em uma base de dados de alta dimensionalidade e desbalanceada[cite: 9]. Foi aplicado um pipeline de pré-processamento com remoção de outliers, balanceamento de classes (SMOTE) e redução de dimensionalidade (comparando PCA e Correlação de Pearson) para otimizar os classificadores K-Nearest Neighbors (KNN), Decision Tree (DT) e Support Vector Machine (SVM)[cite: 10].
---

## I. INTRODUÇÃO

Este projeto aborda o desafio de construir classificadores a partir de uma base de dados de fonte misteriosa[cite: 13]. Primeiramente, foi realizada uma análise exploratória para compreender as características da base[cite: 14]. Em seguida, um conjunto de técnicas de pré-processamento foi aplicado para preparar os dados para os algoritmos de classificação[cite: 14].

As técnicas de pré-processamento incluíram:
* Remoção de outliers[cite: 15].
* Balanceamento de classes[cite: 15].
* Redução de dimensionalidade, comparando a Análise de Componentes Principais (PCA) com a remoção de atributos baseada na Correlação de Pearson[cite: 15].

Utilizou-se uma pipeline de aprendizado para comparar o desempenho de três algoritmos (K-Nearest Neighbors - KNN, Decision Tree - DT e Support Vector Machine - SVM) com diferentes hiperparâmetros e técnicas de pré-processamento[cite: 16]. Todo o desenvolvimento foi realizado em Python, utilizando a biblioteca `scikit-learn`[cite: 17].

---

## II. METODOLOGIA

### A. Análise Exploratória dos Dados

O foco do projeto foi no arquivo `17.csv`[cite: 20]. Este dataset possui **1625 atributos e 69 amostras** [cite: 21], o que indica um alto risco de *overfitting*, tornando a redução de dimensionalidade uma etapa fundamental[cite: 23, 24, 25].

A base de dados apresenta três classes (1, 2 e 3), com a seguinte distribuição:
* **Classe 1:** 11 amostras [cite: 26]
* **Classe 2:** 39 amostras [cite: 26]
* **Classe 3:** 19 amostras [cite: 26]

O desbalanceamento é significativo, já que a classe majoritária (Classe 2) possui mais que o triplo de amostras da classe minoritária (Classe 1)[cite: 27]. Por isso, a métrica de acurácia seria enganosa[cite: 28]. A principal métrica escolhida foi a **F1 Score weighted**, que considera tanto a precisão quanto o recall[cite: 29]. Adicionalmente, foram aplicadas as técnicas `SMOTE` para balanceamento e `StratifiedKFold` para a validação cruzada, garantindo a proporção das classes[cite: 30].

A análise de outliers foi realizada observando a distância entre os valores mínimos/máximos e os quartis, uma vez que a visualização de boxplots para todos os atributos era inviável[cite: 33, 34].

### B. Pré-processamento

* **Remoção de Outliers:** Foi utilizado o algoritmo `Isolation Forest` da biblioteca `scikit-learn`, com um parâmetro de `contamination=0.1`[cite: 39, 40].
* **Balanceamento de Classes:** Foi aplicada a técnica `SMOTE` utilizando a biblioteca `imblearn` para criar amostras sintéticas das classes minoritárias[cite: 41, 42].
* **Redução de Dimensionalidade:**
    * **Correlação de Pearson:** Atributos com correlação superior a 0.9 foram excluídos[cite: 43, 45].
    * **PCA (Principal Component Analysis):** Os componentes foram selecionados de modo a reter 95% da variância total explicada dos dados[cite: 46].

### C. Classificadores

Foram avaliados três classificadores, utilizando `GridSearchCV` para encontrar a melhor combinação de hiperparâmetros[cite: 49]. A métrica de avaliação foi o `F1 Score weighted`, e a validação foi feita com `StratifiedKFold`[cite: 51].

Os modelos e seus respectivos grids de hiperparâmetros foram:
* **KNN:** `n_neighbors`: [3, 5, 7], `weights`: ['uniform', 'distance'] [cite: 53]
* **Decision Tree:** `max_depth`: [3, 5, None], `criterion`: ['gini', 'entropy'] [cite: 54]
* **SVM:** `C`: [0.1, 1, 10], `kernel`: ['linear', 'rbf'], `gamma`: ['scale', 0.1] [cite: 54]

---

## III. RESULTADOS E DISCUSSÃO

Como linha de base, o `DummyClassifier` (que sempre prevê a classe majoritária) obteve um F1-score de $0,410 \pm 0,043$[cite: 75, 76].

A tabela abaixo apresenta os F1-scores ponderados para cada classificador em diferentes cenários[cite: 74]:

| Conjunto                             | KNN                 | Decision Tree       | SVM                 |
| :----------------------------------- | :------------------ | :------------------ | :------------------ |
| **S/ Pré-processamento** | $0.602 \pm 0.103$   | $0.766 \pm 0.123$   | $0.789 \pm 0.084$   |
| **C/ Pré-processamento** | $0.871 \pm 0.037$   | $0.900 \pm 0.035$   | $1.000 \pm 0.000$   |
| **C/ Pré-processamento e PCA** | $0.870 \pm 0.038$   | $0.814 \pm 0.084$   | $0.901 \pm 0.045$   |
| **C/ Pré-processamento e Correlação**| $0.973 \pm 0.036$   | $0.982 \pm 0.035$   | $1.000 \pm 0.000$   |

*(Fonte: Dados adaptados da Tabela I, páginas 2-3)* [cite: 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73]

**Observações:**
* Mesmo sem pré-processamento, todos os modelos superaram a baseline[cite: 77].
* Com o pipeline de pré-processamento (remoção de outliers, SMOTE e normalização), todos os classificadores tiveram ganhos expressivos[cite: 78]. O SVM se destacou, atingindo um F1-score perfeito[cite: 78].
* Entre as técnicas de redução de dimensionalidade, o PCA teve um desempenho melhor para o KNN[cite: 79]. Para a Decision Tree e o SVM, a remoção por correlação de Pearson se mostrou superior[cite: 80].
* Os melhores hiperparâmetros encontrados foram:
    * **KNN:** `n_neighbors=5` e `weights='distance'`[cite: 83].
    * **Decision Tree:** `criterion='gini'` e `max_depth=5`[cite: 84].
    * **SVM:** `kernel='linear'`, `C=0.1` e `gamma='scale'`[cite: 85].

Os resultados reforçam a importância do pré-processamento e da otimização de hiperparâmetros, destacando a robustez do SVM para esta base de dados[cite: 86].

---

## REFERENCES

[1] Månsson, Robert et al. Pearson Correlation Analysis of Microarray Data Allows for the Identification of Genetic Targets for Early B-cell Factor [boxs]. *Journal of Biological Chemistry*, Volume 279, Issue 17. 17905-17913. [cite: 88, 89]
