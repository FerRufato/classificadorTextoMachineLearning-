from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import warnings

# Oculta avisos
warnings.filterwarnings("ignore")

# Lista de stopwords em português
stopwords_pt = [
    "a", "o", "os", "as", "de", "do", "da", "das", "dos",
    "em", "no", "na", "nos", "nas", "um", "uma", "e", "com", "por", "para", "é"
]

# Dados de entrada aumentados
textos = [
    # TECNOLOGIA
    "O novo iPhone chegou ao mercado",
    "Google anuncia atualização do Android",
    "Apple lança novo MacBook com chip M3",
    "Meta apresenta recursos avançados de IA",
    "Samsung apresenta celular dobrável",
    "Intel revela novos processadores",
    "Microsoft lança nova versão do Windows",
    "Tesla desenvolve nova tecnologia de bateria",

    # ESPORTES
    "Flamengo vence o clássico carioca",
    "Palmeiras contrata novo jogador",
    "Seleção brasileira vence nas eliminatórias",
    "Corinthians estreia com vitória no campeonato",
    "Brasil conquista medalha nas Olimpíadas",
    "Time vence partida nos acréscimos",
    "Atleta bate recorde mundial",
    "Final do campeonato será no Maracanã",

    # POLÍTICA
    "Congresso aprova nova lei",
    "Presidente discursa na ONU",
    "Ministro anuncia novo plano econômico",
    "Governo propõe reforma tributária",
    "Novo projeto de lei é debatido no Senado",
    "Câmara aprova orçamento federal",
    "Senador propõe mudanças na educação",
    "Eleições terão novas regras em vigor"
]

categorias = [
    "tecnologia", "tecnologia", "tecnologia", "tecnologia",
    "tecnologia", "tecnologia", "tecnologia", "tecnologia",
    "esportes", "esportes", "esportes", "esportes",
    "esportes", "esportes", "esportes", "esportes",
    "política", "política", "política", "política",
    "política", "política", "política", "política"
]

# Vetorização
vectorizer = TfidfVectorizer(stop_words=stopwords_pt)
X = vectorizer.fit_transform(textos)

# Separação treino/teste com stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, categorias, test_size=0.2, random_state=42, stratify=categorias
)

# Treinamento e predição
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Exibição do relatório
print("\n✅ Relatório de Classificação:\n")
print(classification_report(y_test, y_pred))
