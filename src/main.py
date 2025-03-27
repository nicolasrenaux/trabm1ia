import numpy as np
import skfuzzy as fuzz
import pandas as pd
from skfuzzy import control as ctrl

# Carregar o dataset
movie_data = pd.read_csv('assets/movie_dataset.csv')

# Perguntar quem está assistindo
opcao = ""
while opcao not in {"1", "2", "3", "4"}:
    print("Quem está assistindo hoje?")
    print("1 - Arthur Carminati")
    print("2 - Nicolas Renaux")
    print("3 - Pedro Ruthes")
    print("4 - Henrique Schroeder")
    opcao = input("Escolha uma opção (1-4): ").strip()
    if opcao not in {"1", "2", "3", "4"}:
        print("Opção inválida. Tente novamente.\n")

# Preferências individuais
preferencias = {
    # Arthur Carminati
    "1": {
        "generos": {"suspense", "crime", "drama"},
        "keywords": {"real", "psychological", "dark"},
        "peso_revenue": 0.5,
        "peso_popularity": 0.7,
    },
    # Nicolas Renaux
    "2": {
        "generos": {"suspense", "comedy", "adventure", "crime", "action"},
        "keywords": {"thriller", "true story", "space", "sci-fi"},
        "peso_revenue": 0.8,
        "peso_popularity": 0.9,
    },
    # Pedro Ruthes
    "3": {
        "generos": {"comedy", "family", "romance"},
        "keywords": {"love", "friendship", "school"},
        "peso_revenue": 0.3,
        "peso_popularity": 0.6,
    },
    # Henrique Schroeder
    "4": {
        "generos": {"action", "adventure", "sci-fi", "suspense", "comedy"},
        "keywords": {"spy", "time travel", "tech", "dark", "history"},
        "peso_revenue": 0.5,
        "peso_popularity": 0.9,
    },
}

prefs = preferencias.get(opcao, preferencias["1"])

# Definição das variáveis de entrada
vote_average = ctrl.Antecedent(np.arange(0, 10, 0.1), 'nota')
duration = ctrl.Antecedent(np.arange(0, 300, 1), 'duracao')
genre = ctrl.Antecedent(np.arange(0, 3, 1), 'genero')
revenue = ctrl.Antecedent(np.arange(0, 3_000_000_000, 1_000_000), 'revenue')
keywords = ctrl.Antecedent(np.arange(0, 3, 1), 'keywords')
popularity = ctrl.Antecedent(np.arange(0, 3_000_000_000, 1_000_000), 'popularity')
score = ctrl.Consequent(np.arange(0, 100, 1), 'score')

# Funções de pertinência
vote_average['ruim'] = fuzz.trapmf(vote_average.universe, [0, 0, 2, 4])
vote_average['regular'] = fuzz.trimf(vote_average.universe, [3, 5, 7])
vote_average['boa'] = fuzz.trimf(vote_average.universe, [6, 7.5, 9])
vote_average['excelente'] = fuzz.trapmf(vote_average.universe, [8, 9, 10, 10])

duration['curta'] = fuzz.trapmf(duration.universe, [0, 0, 60, 90])
duration['media'] = fuzz.trimf(duration.universe, [90, 120, 150])
duration['longa'] = fuzz.trapmf(duration.universe, [120, 180, 300, 300])

genre['nenhum'] = fuzz.trimf(genre.universe, [0, 0, 1])
genre['presente'] = fuzz.trimf(genre.universe, [0, 1, 2])
genre['muito_presente'] = fuzz.trimf(genre.universe, [1, 2, 2])

revenue['baixo'] = fuzz.trapmf(revenue.universe, [0, 0, 500_000_000, 1_000_000_000])
revenue['medio'] = fuzz.trimf(revenue.universe, [500_000_000, 1_500_000_000, 2_500_000_000])
revenue['alto'] = fuzz.trapmf(revenue.universe, [1_500_000_000, 2_500_000_000, 3_000_000_000, 3_000_000_000])

keywords['nenhuma'] = fuzz.trimf(keywords.universe, [0, 0, 1])
keywords['presente'] = fuzz.trimf(keywords.universe, [0, 1, 2])
keywords['muito_presente'] = fuzz.trimf(keywords.universe, [1, 2, 2])

popularity['baixa'] = fuzz.trapmf(popularity.universe, [0, 0, 500_000_000, 1_000_000_000])
popularity['media'] = fuzz.trimf(popularity.universe, [500_000_000, 1_500_000_000, 2_500_000_000])
popularity['alta'] = fuzz.trapmf(popularity.universe, [1_500_000_000, 2_500_000_000, 3_000_000_000, 3_000_000_000])

score['muito_baixo'] = fuzz.trapmf(score.universe, [0, 0, 10, 30])
score['baixo'] = fuzz.trimf(score.universe, [20, 40, 50])
score['medio'] = fuzz.trimf(score.universe, [40, 60, 70])
score['alto'] = fuzz.trimf(score.universe, [60, 80, 90])
score['muito_alto'] = fuzz.trapmf(score.universe, [80, 90, 100, 100])

# Regras Fuzzy
rules = [
    ctrl.Rule(vote_average['excelente'] & duration['longa'], score['muito_alto']),
    ctrl.Rule(vote_average['excelente'] & duration['media'], score['alto']),
    ctrl.Rule(vote_average['excelente'] & duration['curta'], score['baixo']),
    ctrl.Rule(vote_average['boa'] & duration['longa'], score['alto']),
    ctrl.Rule(vote_average['boa'] & duration['media'], score['medio']),
    ctrl.Rule(vote_average['regular'] & duration['longa'], score['medio']),
    ctrl.Rule(vote_average['regular'] & duration['media'], score['baixo']),
    ctrl.Rule(vote_average['ruim'] & duration['curta'], score['muito_baixo']),
    ctrl.Rule(genre['presente'], score['medio']),
    ctrl.Rule(genre['muito_presente'], score['alto']),
    ctrl.Rule(revenue['alto'], score['muito_alto']),
    ctrl.Rule(revenue['medio'], score['alto']),
    ctrl.Rule(revenue['baixo'], score['medio']),
    ctrl.Rule(keywords['presente'], score['medio']),
    ctrl.Rule(keywords['muito_presente'], score['muito_alto']),
    ctrl.Rule(popularity['alta'], score['muito_alto']),
    ctrl.Rule(popularity['media'], score['alto']),
    ctrl.Rule(popularity['baixa'], score['medio']),
]

# Sistema Fuzzy
ranking_ctrl = ctrl.ControlSystem(rules)
ranking = ctrl.ControlSystemSimulation(ranking_ctrl)

# Aplicar o sistema
rankings = []
for _, row in movie_data.iterrows():
    try:
        ranking.input['nota'] = row['vote_average']
        ranking.input['duracao'] = row['runtime'] if pd.notna(row['runtime']) else 100

        revenue_val = row['revenue'] if pd.notna(row['revenue']) else 0
        popularity_val = row['popularity'] if pd.notna(row['popularity']) else 0

        ranking.input['revenue'] = revenue_val * prefs["peso_revenue"]
        ranking.input['popularity'] = popularity_val * prefs["peso_popularity"]

        genres = row['genres'].lower().split() if pd.notna(row['genres']) else []
        genre_score = sum(1 for g in prefs["generos"] if g in genres)
        ranking.input['genero'] = min(genre_score, 2)

        keywords_list = row['keywords'].lower().split() if pd.notna(row['keywords']) else []
        keyword_score = sum(1 for k in prefs["keywords"] if k in keywords_list)
        ranking.input['keywords'] = min(keyword_score, 2)

        ranking.compute()
        score_value = ranking.output.get('score', 0)
        rankings.append((row['title'], score_value))
    except Exception as e:
        print(f"Erro ao processar o filme {row.get('title', 'Desconhecido')}: {e}")

# Exibir os 10 melhores
rankings.sort(key=lambda x: x[1], reverse=True)
print('\nTop 10 Filmes Rankeados:')
for title, score in rankings[:10]:
    print(f'{title}: {score:.2f}')
