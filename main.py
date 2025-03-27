import numpy as np
import skfuzzy as fuzz
import pandas as pd
from skfuzzy import control as ctrl

# Carregar o dataset
movie_data = pd.read_csv('movie_dataset.csv')

# Definição das variáveis de entrada
vote_average = ctrl.Antecedent(np.arange(0, 10, 0.1), 'nota')
duration = ctrl.Antecedent(np.arange(0, 300, 1), 'duracao')
genre = ctrl.Antecedent(np.arange(0, 3, 1), 'genero')  # 0: Nenhum, 1: Presente, 2: Muito presente
revenue = ctrl.Antecedent(np.arange(0, 3_000_000_000, 1_000_000), 'revenue')  # Faturamento em bilhões
keywords = ctrl.Antecedent(np.arange(0, 3, 1), 'keywords')  # 0: Nenhuma, 1: Presente, 2: Muito presente
popularity = ctrl.Antecedent(np.arange(0, 3_000_000_000, 1_000_000), 'popularity')  # Popularidade em bilhões


# Variável de saída
score = ctrl.Consequent(np.arange(0, 100, 1), 'score')

# Funções de pertinência - Nota
vote_average['ruim'] = fuzz.trapmf(vote_average.universe, [0, 0, 2, 4])
vote_average['regular'] = fuzz.trimf(vote_average.universe, [3, 5, 7])
vote_average['boa'] = fuzz.trimf(vote_average.universe, [6, 7.5, 9])
vote_average['excelente'] = fuzz.trapmf(vote_average.universe, [8, 9, 10, 10])

# Funções de pertinência - Duração
duration['curta'] = fuzz.trapmf(duration.universe, [0, 0, 60, 90])
duration['media'] = fuzz.trimf(duration.universe, [90, 120, 150])
duration['longa'] = fuzz.trapmf(duration.universe, [120, 180, 300, 300])

# Funções de pertinência - Gênero
genre['nenhum'] = fuzz.trimf(genre.universe, [0, 0, 1])
genre['presente'] = fuzz.trimf(genre.universe, [0, 1, 2])
genre['muito_presente'] = fuzz.trimf(genre.universe, [1, 2, 2])

# Funções de pertinência - Receita
revenue['baixo'] = fuzz.trapmf(revenue.universe, [0, 0, 500_000_000, 1_000_000_000])
revenue['medio'] = fuzz.trimf(revenue.universe, [500_000_000, 1_500_000_000, 2_500_000_000])
revenue['alto'] = fuzz.trapmf(revenue.universe, [1_500_000_000, 2_500_000_000, 3_000_000_000, 3_000_000_000])

# Funções de pertinência - Keywords
keywords['nenhuma'] = fuzz.trimf(keywords.universe, [0, 0, 1])
keywords['presente'] = fuzz.trimf(keywords.universe, [0, 1, 2])
keywords['muito_presente'] = fuzz.trimf(keywords.universe, [1, 2, 2])

# Funções de pertinência - Popularidade
popularity['baixa'] = fuzz.trapmf(popularity.universe, [0, 0, 500_000_000, 1_000_000_000])
popularity['media'] = fuzz.trimf(popularity.universe, [500_000_000, 1_500_000_000, 2_500_000_000])
popularity['alta'] = fuzz.trapmf(popularity.universe, [1_500_000_000, 2_500_000_000, 3_000_000_000, 3_000_000_000])


# Funções de pertinência - Score
score['muito_baixo'] = fuzz.trapmf(score.universe, [0, 0, 10, 30])
score['baixo'] = fuzz.trimf(score.universe, [20, 40, 50])
score['medio'] = fuzz.trimf(score.universe, [40, 60, 70])
score['alto'] = fuzz.trimf(score.universe, [60, 80, 90])
score['muito_alto'] = fuzz.trapmf(score.universe, [80, 90, 100, 100])

# Regras Fuzzy
rule1 = ctrl.Rule(vote_average['excelente'] & duration['longa'], score['muito_alto'])
rule2 = ctrl.Rule(vote_average['excelente'] & duration['media'], score['alto'])
rule2 = ctrl.Rule(vote_average['excelente'] & duration['curta'], score['baixo'])
rule3 = ctrl.Rule(vote_average['boa'] & duration['longa'], score['alto'])
rule4 = ctrl.Rule(vote_average['boa'] & duration['media'], score['medio'])
rule5 = ctrl.Rule(vote_average['regular'] & duration['longa'], score['medio'])
rule6 = ctrl.Rule(vote_average['regular'] & duration['media'], score['baixo'])
rule7 = ctrl.Rule(vote_average['ruim'] & duration['curta'], score['muito_baixo'])
rule8 = ctrl.Rule(genre['presente'], score['medio'])
rule9 = ctrl.Rule(genre['muito_presente'], score['alto'])
rule10 = ctrl.Rule(revenue['alto'], score['muito_alto'])
rule11 = ctrl.Rule(revenue['medio'], score['alto'])
rule12 = ctrl.Rule(revenue['baixo'], score['medio'])
rule13 = ctrl.Rule(keywords['presente'], score['medio'])
rule14 = ctrl.Rule(keywords['muito_presente'], score['muito_alto'])
rule15 = ctrl.Rule(popularity['alta'], score['muito_alto'])
rule16 = ctrl.Rule(popularity['media'], score['alto'])
rule17 = ctrl.Rule(popularity['baixa'], score['medio'])


# Sistema de Controle
ranking_ctrl = ctrl.ControlSystem([
    rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10,
    rule11, rule12, rule13, rule14, rule15, rule16, rule17
])
ranking = ctrl.ControlSystemSimulation(ranking_ctrl)

# Aplicando o Sistema Fuzzy aos filmes
rankings = []
for _, row in movie_data.iterrows():
    try:
        ranking.input['nota'] = row['vote_average']
        ranking.input['duracao'] = row['runtime'] if pd.notna(row['runtime']) else 100
        ranking.input['revenue'] = row['revenue'] if pd.notna(row['revenue']) else 0
        ranking.input['popularity'] = row['popularity'] if pd.notna(row['popularity']) else 0

        
        # Verificar presença dos gêneros de interesse e excluir super-heróis
        genres = row['genres'].lower().split() if pd.notna(row['genres']) else []
        relevant_genres = {'suspense', 'comedy', 'adventure', 'crime', 'action'}
        excluded_genres = {'superhero'}
        genre_score = sum(1 for g in relevant_genres if g in genres and g not in excluded_genres)
        ranking.input['genero'] = min(genre_score, 2)  # Ajustando para a escala 0-2
        
        # Verificar palavras-chave relevantes
        keywords_list = row['keywords'].lower().split() if pd.notna(row['keywords']) else []
        relevant_keywords = {'thriller', 'true story', 'space', 'sci-fi'}
        keyword_score = sum(1 for k in relevant_keywords if k in keywords_list)
        ranking.input['keywords'] = min(keyword_score, 2)
        
        ranking.compute()
        score_value = ranking.output.get('score', 0)
        rankings.append((row['title'], score_value))
    except Exception as e:
        print(f"Erro ao processar o filme {row['title']}: {e}")

# Ordenando os resultados
rankings.sort(key=lambda x: x[1], reverse=True)

# Exibindo os 10 melhores filmes
print('Top 10 Filmes Rankeados:')
for title, score in rankings[:10]:
    print(f'{title}: {score:.2f}')
