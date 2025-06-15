from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import warnings

# 1) Suprime avisos
warnings.filterwarnings("ignore")

# 2) Stopwords em português
stopwords_pt = [
    "a", "o", "os", "as", "de", "do", "da", "das", "dos",
    "em", "no", "na", "nos", "nas", "um", "uma", "e",
    "com", "por", "para", "é"
]

# 3) Frases-base por categoria (cada lista deve ter 30 entradas únicas)
base_texts = {
    "tecnologia": [
        "O novo iPhone chegou ao mercado",
        "Google anuncia atualização do Android",
        "Apple lança novo MacBook com chip M3",
        "Meta apresenta recursos avançados de IA",
        "Samsung apresenta celular dobrável",
        "Intel revela novos processadores",
        "Microsoft lança nova versão do Windows",
        "Tesla desenvolve nova tecnologia de bateria",
        "NVIDIA anuncia arquitetura de GPU RTX 5000",
        "WhatsApp libera reações em mensagens de voz",
        "Linux 6.5 traz melhorias de desempenho",
        "Amazon Web Services amplia serviços de IA",
        "Facebook testa feed personalizado com IA",
        "Twitter implementa criptografia de ponta a ponta",
        "Uber lança serviço de táxi voador em testes",
        "IBM desenvolve supercomputador quântico",
        "Samsung Galaxy Z Fold4 ganha nova câmera",
        "Google Chrome exige menos memória RAM",
        "YouTube lança modo offline para vídeos",
        "TikTok expande limite de duração de vídeos",
        "Microsoft Teams ganha integração com Outlook",
        "Netflix testa streaming interativo em jogos",
        "WhatsApp Web permite chamadas de vídeo",
        "Adobe atualiza Photoshop com IA generativa",
        "Intel investe em chips de 2nm",
        "Sony anuncia sensor de imagem de 100MP",
        "Xiaomi lança celular 5G acessível",
        "Discord testa servidores de voz de alta fidelidade",
        "Spotify integra algoritmo de recomendação",
        "Reddit anuncia redesign da interface móvel"
    ],
    "esportes": [
        "Flamengo vence o clássico carioca",
        "Palmeiras contrata novo jogador",
        "Seleção brasileira vence nas eliminatórias",
        "Corinthians estreia com vitória no campeonato",
        "Brasil conquista medalha nas Olimpíadas",
        "Time vence partida nos acréscimos",
        "Atleta bate recorde mundial",
        "Final do campeonato será no Maracanã",
        "Neymar sofre lesão em partida internacional",
        "Pelé é homenageado em cerimônia histórica",
        "Lewis Hamilton conquista pole position",
        "Simona Halep avança à final do Grand Slam",
        "Fórmula 1 testa pneus em pista molhada",
        "NBA anuncia estrela para jogo das celebridades",
        "Ligue 1 contrata Ronaldinho como embaixador",
        "UFC confirma duelo de pesos pesados",
        "Maratona de Nova Iorque atrai milhares",
        "Tour de France tem etapa de montanha decisiva",
        "Vasco festeja retorno à Série A do Brasileirão",
        "Cruzeiro sobe para a liderança da Série B",
        "Santos FC faz treinamento com novos reforços",
        "Chelsea contrata técnico renomado",
        "Atlanta Hawks vence jogo em prorrogação",
        "Lionel Messi marca hat-trick em amistoso",
        "Rafael Nadal se aposenta após 20 anos de carreira",
        "Seleção feminina de vôlei ganha campeonato mundial",
        "Skateboarding é destaque nos Jogos Olímpicos",
        "Brasil sediará Copa do Mundo de Basquete 2027",
        "Juan Mata anuncia aposentadoria do futebol"
    ],
    "política": [
        "Congresso aprova nova lei",
        "Presidente discursa na ONU",
        "Ministro anuncia novo plano econômico",
        "Governo propõe reforma tributária",
        "Novo projeto de lei é debatido no Senado",
        "Câmara aprova orçamento federal",
        "Senador propõe mudanças na educação",
        "Eleições terão novas regras em vigor",
        "Prefeito inaugura obra de mobilidade urbana",
        "Governo estadual decreta estado de emergência",
        "Partido lança candidatura à presidência",
        "Ministério público abre inquérito sobre corrupção",
        "Assembleia vota mudanças na previdência",
        "Tribunal Superior Eleitoral define calendário eleitoral",
        "Governador anuncia investimento em saúde",
        "Deputados discutem flexibilização de armas",
        "Brasil firma acordo comercial com União Europeia",
        "ONU critica violações de direitos humanos",
        "Embaixada dos EUA celebra diplomacia cultural",
        "Ministro da Justiça apresenta balanço de segurança pública",
        "Governo lança programa de geração de empregos",
        "Senado aprova emergência climática",
        "Assembleia Geral discute crise migratória",
        "Ministro da Educação apresenta novo currículo escolar",
        "Partidos debatem coalizão para governo local",
        "Presidência recebe líderes mundiais para cúpula",
        "Governo Federal anuncia pacote de privatizações",
        "Defesa Nacional realiza exercício militar conjunto",
        "MPF investiga contratos do setor de energia"
    ],
    "saúde": [
        "Ministério da Saúde lança campanha de vacinação",
        "Hospital municipal inaugura novo centro cirúrgico",
        "Clínica especializada oferece exames gratuitos",
        "Pesquisa desenvolve tratamento contra COVID-19",
        "Consultório odontológico inicia atendimento 24h",
        "Plano de saúde amplia cobertura para idosos",
        "Nutricionista recomenda dieta balanceada",
        "Suspeita de surto de dengue em bairro periférico",
        "Estudo aponta aumento de casos de ansiedade",
        "Vacina contra HPV é oferecida gratuitamente",
        "Pesquisa relaciona sono à prevenção de doenças",
        "Médicos realizam cirurgia robótica inovadora",
        "Campanha de doação de sangue é um sucesso",
        "Laboratório testa novo medicamento contra câncer",
        "Programa de saúde mental chega a escolas",
        "Especialistas alertam sobre resistência a antibióticos",
        "Hospital universitário ganha prêmio internacional",
        "Pesquisa sobre Alzheimer avança em testes clínicos",
        "Estudo liga poluição a problemas respiratórios",
        "Clínica cardíaca inaugura unidade de emergência",
        "Governo patrocina projeto de saúde comunitária",
        "Farmácia popular reduz preços de remédios",
        "Estudo sobre obesidade é publicado em revista médica",
        "Psicólogos promovem terapia em grupo online",
        "Vacinação antirrábica ocorre em bairros rurais",
        "Ministério anuncia combate à malária",
        "Nutrólogos criam app para monitorar dietas",
        "Hospital de campanha é montado durante epidemia",
        "Pesquisa identifica genética de doenças raras"
    ],
    "entretenimento": [
        "Filme nacional ganha prêmio em festival de cinema",
        "Série de suspense estreia em plataforma de streaming",
        "Artista faz show beneficente para arrecadar fundos",
        "Festival de música atrai milhares de pessoas",
        "Peça de teatro infantil estreia neste fim de semana",
        "Cantor grava clipe em locação histórica",
        "Netflix anuncia nova temporada de documentário",
        "Reality show define finalistas em votação pública",
        "Quadrinista lança nova HQ sobre super-heróis",
        "Banda brasileira faz turnê internacional",
        "Museu de arte contemporânea abre exposição",
        "Humorista grava especial de stand-up comedy",
        "Game de realidade virtual é lançado mundialmente",
        "Cantora pop faz participação em novela",
        "Evento de cosplay reúne fãs no centro da cidade",
        "YouTuber bate recorde de inscritos em 24h",
        "Teatro reabre com espetáculo de dança clássica",
        "Produtora anuncia remake de clássico dos anos 80",
        "Show de fogos celebra inauguração de parque",
        "Podcast cultural estreia com entrevista exclusiva",
        "Coreógrafo apresenta nova peça de balé moderno",
        "Premiação de cinema atrai celebridades",
        "Festival de comédia tem ingressos esgotados",
        "Artista plástico expõe obras em galeria renomada",
        "Jingle de novela vira sucesso nas rádios",
        "Bailarinos participam de concurso internacional",
        "Game indie ganha prêmio de inovação",
        "Série documental discute história do rock brasileiro",
        "Parque temático comemora aniversário com festa"
    ],
    "negócios": [
        "Economia cresce 2% no último trimestre",
        "Bolsa de valores opera em alta histórica",
        "Startup recebe aporte de investidores internacionais",
        "Inflação desacelera e preços começam a cair",
        "Setor financeiro lança fintech de pagamentos",
        "Fusão entre duas grandes empresas é homologada",
        "Empresas brasileiras expandem mercado externo",
        "Análise aponta alta na taxa de juros para dezembro",
        "Indústria automobilística investe em carros elétricos",
        "Consultoria ressalta importância de ESG",
        "Marketplace alcança recorde de vendas online",
        "Relatório aponta queda no desemprego formal",
        "Empresa de logística adota drones para entregas",
        "Shoppings reportam aumento no fluxo de visitantes",
        "Pesquisa revela perfil do consumidor pós-pandemia",
        "Rede de franquias expande operações nacionais",
        "CEO anuncia plano de reestruturação corporativa",
        "Banco digital lança produto de investimento automatizado",
        "Feira de negócios reúne startups de tecnologia",
        "Segmento de saúde animal cresce 15% ao ano",
        "Imobiliárias registram alta nos aluguéis comerciais",
        "Relatório global destaca crescimento de e-commerce",
        "Empresa de energia anuncia capitalização no mercado",
        "Consultoria em RH vê aumento de contratações temporárias",
        "Mercado de criptomoedas oscila com notícias regulatórias",
        "Varejistas adotam realidade aumentada para vendas",
        "Companhia aérea anuncia nova rota internacional",
        "Sindicato negocia reajuste salarial com indústrias",
        "Fundo de investimentos realiza IPO de fintech"
    ]
}

# 4) Monta lista de textos e categorias sem repetições artificiais
textos = []
categorias = []
for cat, frases in base_texts.items():
    textos.extend(frases)
    categorias.extend([cat] * len(frases))

# 5) Vetorização TF–IDF
vectorizer = TfidfVectorizer(stop_words=stopwords_pt)
X = vectorizer.fit_transform(textos)

# 6) Separação treino/teste (50/50) com estratificação
X_train, X_test, y_train, y_test = train_test_split(
    X, categorias,
    test_size=0.5,
    random_state=42,
    stratify=categorias
)

# 7) Treinamento do modelo
clf = MultinomialNB()
clf.fit(X_train, y_train)

# 8) Avaliação no conjunto de teste
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Acurácia geral no teste: {acc:.2f}\n")
print("▶️ Relatório detalhado no teste:")
print(classification_report(y_test, y_pred))
