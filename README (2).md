
# Classificador de Texto com Machine Learning

Este projeto tem como objetivo construir um classificador de textos curtos em Python, capaz de identificar a qual categoria (tecnologia, esportes ou política) um texto pertence. Ele utiliza machine learning supervisionado com Scikit-learn.

## 💡 O que este projeto faz

- Classifica textos curtos em categorias pré-definidas.
- Usa aprendizado supervisionado com o algoritmo Naive Bayes.
- Apresenta relatório de desempenho com métricas de avaliação.

## 📁 Estrutura do Projeto

- `python_classificador.py`: Script principal que realiza:
  - Vetorização dos textos com TF-IDF
  - Treinamento com MultinomialNB
  - Avaliação com classification_report

## ⚙️ Como funciona

1. **Entrada dos dados**: Lista de frases e suas categorias corretas.
2. **Vetorização**: Os textos são transformados em vetores numéricos usando TF-IDF.
3. **Treinamento**: O modelo aprende os padrões com base nos dados.
4. **Teste e Avaliação**: O modelo é avaliado com dados novos e gera um relatório de desempenho.

## 🧠 Técnicas usadas

- Aprendizado supervisionado
- Vetorização com TfidfVectorizer
- Classificação com Multinomial Naive Bayes
- Avaliação com classification_report (precisão, recall, f1-score, acurácia)
- Remoção de stopwords em português

## 📊 Exemplo de relatório gerado

```
              precision    recall  f1-score   support

    esportes       1.00      1.00      1.00         1
    política       0.50      0.50      0.50         2
  tecnologia       0.50      0.50      0.50         2

    accuracy                           0.60         5
   macro avg       0.67      0.67      0.67         5
weighted avg       0.60      0.60      0.60         5
```

## 🚀 Possíveis Aplicações na Empresa

- Classificação de mensagens de atendimento por tema
- Análise de sentimentos ou opiniões em comentários
- Organização automática de documentos por assunto
- Filtragem de conteúdo por categoria

## ✅ Como executar

1. Certifique-se de ter o Python instalado (3.8+ recomendado).
2. Instale as dependências:

```
pip install scikit-learn
```

3. Execute o script:

```
python python_classificador.py
```

## 📌 Observações

Este projeto é uma base para entender como funciona um classificador supervisionado. Para aplicações reais, é recomendado trabalhar com datasets maiores e modelos mais avançados.
