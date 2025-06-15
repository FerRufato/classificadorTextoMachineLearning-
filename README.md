# Classificador de Texto com Machine Learning

Este projeto em Python treina um classificador de textos curtos para identificar a qual das seis categorias um texto pertence: **tecnologia**, **esportes**, **política**, **saúde**, **entretenimento** e **negócios**.

## 💡 Funcionalidades

- Classificação de textos em categorias pré-definidas.
- Aprendizado supervisionado usando Multinomial Naive Bayes.
- Métricas de desempenho: precisão, recall, f1-score e acurácia no conjunto de teste.

## 📁 Estrutura do Projeto

```
classificador-texto/
│
├── python_classificador.py    # Script principal
└── README.md                  # Documentação (este arquivo)
```

### `python_classificador.py`

1. **Pré-processamento**

   - Remove stopwords em português.

2. **Vetorização**

   - Converte textos em vetores TF–IDF uni- e bi-gramas.

3. **Divisão treino/teste**

   - 50% dos dados para treino, 50% para teste, com estratificação por categoria.

4. **Treinamento**

   - Ajusta um `MultinomialNB` nos dados de treino.

5. **Avaliação**

   - Gera relatório com `classification_report` e imprime a acurácia geral.

## ⚙️ Fluxo de Execução

1. **Coleta de dados**: listas de frases rotuladas em cada categoria.
2. **Limpeza**: remoção de stopwords e normalização (minúsculas, remoção de pontuação, etc.).
3. **Extração de features**: TF–IDF uni- e bi-gramas.
4. **Divisão dos dados**: `train_test_split(..., stratify=...)`.
5. **Treinamento**: `clf = MultinomialNB(); clf.fit(X_train, y_train)`.
6. **Predição e métricas**: `y_pred = clf.predict(X_test)` e `classification_report`.

## 🧠 Técnicas e Bibliotecas

- **Aprendizado Supervisionado**
- `scikit-learn`:

  - `TfidfVectorizer`
  - `train_test_split`
  - `MultinomialNB`
  - `classification_report`, `accuracy_score`

- `warnings.filterwarnings("ignore")` para suprimir alertas de depuração.

## 📊 Exemplo de Saída no Teste

```
✅ Acurácia geral no teste: 0.44
▶️ Relatório detalhado no teste:
                precision    recall  f1-score   support

entretenimento       0.50      0.33      0.40        15
      esportes       0.35      0.64      0.45        14
      negócios       0.43      0.20      0.27        15
      política       0.44      0.27      0.33        15
         saúde       0.38      0.43      0.40        14
    tecnologia       0.60      0.80      0.69        15

      accuracy                           0.44        88
     macro avg       0.45      0.45      0.42        88
  weighted avg       0.45      0.44      0.42        88
```

## 🚀 Possíveis Aplicações

- Classificação automática de tickets de suporte por assunto.
- Organização de documentos internos ou e-mails corporativos.
- Filtragem de mensagens em redes sociais ou chats de atendimento.

## ✅ Como Executar

1. Clone este repositório.

2. Crie e ative um ambiente virtual (recomendado).

3. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

4. Execute o script:

   ```bash
   python python_classificador.py
   ```

## 📌 Melhorias Futuras

- Aumentar e diversificar o dataset (mais exemplos reais por categoria).
- Incluir bi-gramas e pré-processamento avançado (stemming, lematização).
- Testar outros classificadores (SVM, Random Forest, redes neurais).
- Implementar validação cruzada (`StratifiedKFold`) e tuning de hiperparâmetros (`GridSearchCV`).
- Adicionar interface web ou API REST para uso em produção.
