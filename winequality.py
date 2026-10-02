import os
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error

# Importando bibliotecas de aprendizado profundo
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Input

def ajustar_outliers(dados):
    dados = np.copy(dados)
    for j in range(10):
        mean = np.mean(dados)
        str_dev = np.std(dados)
        
        for i in range(len(dados)):
            if dados[i] < mean - 3 * str_dev:
                dados[i] = mean - 3.0 * str_dev
            elif dados[i] > mean + 3 * str_dev:
                dados[i] = mean + 3.0 * str_dev

    return dados

# Carregando o dataset na mesma pasta do script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "winequality-red.csv")

wine_data = pd.read_csv(csv_path, delimiter=';')

# Removendo a coluna especificada
wine_data = wine_data.drop(columns=['residual sugar']) 

# Selecionando as colunas para normalização
colunas = ['fixed acidity', 'volatile acidity', 'citric acid',
           'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
           'pH', 'sulphates', 'alcohol']

# Criando uma instância do StandardScaler
scaler = StandardScaler()

# Aplicando a normalização em todas as colunas selecionadas
wine_data_normalizado = scaler.fit_transform(wine_data[colunas])

# Convertendo de volta para um DataFrame
wine_data_normalizado = pd.DataFrame(wine_data_normalizado, columns=colunas)

wine_data_normalizado['quality'] = wine_data['quality'].values

for coluna in colunas:
    wine_data_normalizado[coluna] = ajustar_outliers(wine_data_normalizado[coluna].values)

y = wine_data_normalizado['quality']  # Target
x = wine_data_normalizado.drop(columns=['quality'])  # Features

# Dividir o conjunto de dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


def treinar_lstm(dados_treino, saida_treino, n_lags=1, n_outputs=1, epochs=100, batch_size=64):

    # Ajustar os dados para o formato LSTM (samples, timesteps, features)
    dados_treino = np.array(dados_treino)
    saida_treino = np.array(saida_treino)

    dados_treino = dados_treino.reshape((dados_treino.shape[0], n_lags, dados_treino.shape[1] // n_lags))

    # Criando o modelo LSTM
    model = Sequential()
    model.add(Input(shape=(n_lags, dados_treino.shape[2])))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(n_outputs))  # Saída com o número de previsões
    
    model.compile(optimizer='adam', loss='mse')
    
    # Treinamento do modelo
    model.fit(dados_treino, saida_treino, epochs=epochs, batch_size=batch_size, verbose=1)
    
    return model

def testar_lstm(model, dados_teste, n_lags=1):
    # Ajustar os dados para o formato LSTM (samples, timesteps, features)
    dados_teste = np.array(dados_teste)
    dados_teste = dados_teste.reshape((dados_teste.shape[0], n_lags, dados_teste.shape[1] // n_lags))

    # Fazer a previsão
    previsao = model.predict(dados_teste)

    return previsao


# Executando a LSTM
lstm = treinar_lstm(x_train, y_train)
y_prev_lstm = testar_lstm(lstm, x_test)

# Avaliar o modelo LSTM
mae_lstm = mean_absolute_error(y_test, y_prev_lstm)

# Exibir resultados
print("\n===========\nLSTM")
print(f'\nErro médio absoluto : {mae_lstm}')


# Definir o regressor MLP
mlp = MLPRegressor(hidden_layer_sizes=(50,),
                   activation='relu',
                   random_state=42,
                   solver='adam',
                   batch_size=16,
                   max_iter=500)

# Treinar o modelo MLP
mlp.fit(x_train, y_train)

# Fazer previsões no conjunto de teste
y_prev_mlp = mlp.predict(x_test)

# Avaliar o modelo MLP
mae_mlp = mean_absolute_error(y_test, y_prev_mlp)

# Exibir resultados
print("\n===========\nMLP")
print(f'\nErro médio absoluto : {mae_mlp}\n')