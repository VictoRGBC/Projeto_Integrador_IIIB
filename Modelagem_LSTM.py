import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# -------------------- I. FUNÇÕES AUXILIARES --------------------
# Função para criar janelas deslizantes (X = janelas passadas, Y = previsões futuras)
def criar_dataset_janelas(data, N_observacao, K_previsao):
    X, Y = [], []
    for i in range(len(data) - N_observacao - K_previsao + 1):
        janela_x = data[i:(i + N_observacao)]
        X.append(janela_x)
        janela_y = data[(i + N_observacao):(i + N_observacao + K_previsao)]
        Y.append(janela_y)
    return np.array(X), np.array(Y)

# Função para calcular o MAPE (Erro Percentual Absoluto Médio)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Filtra valores para evitar divisão por zero
    return np.mean(np.abs((y_true - y_pred) / y_true[y_true != 0])) * 100

# Função para definir a arquitetura do modelo
def criar_modelo_lstm(N_observacao, K_previsao):
    model = Sequential()
    model.add(LSTM(
        units=50, 
        activation='relu', 
        input_shape=(N_observacao, 1),
        return_sequences=True
    ))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    
    # Camada de saída para 1 valor (embora o plano use 3, vamos manter 1 para consistência com o que rodamos)
    model.add(Dense(units=K_previsao)) 
    return model

# -------------------- II. PREPARAÇÃO E MODELAGEM --------------------

# Carregar o arquivo corrigido (gerado no Notebook)
# Nota: O plano de ação usa sep=';' e decimal=',', mas usamos ',' e '.' no tratamento.
# Vamos usar o separador correto para o arquivo final gerado:
df_inflacao = pd.read_csv(
    'dados_inflacao_corrigidos.csv', 
    index_col=0, 
    sep=',', 
    decimal='.', # Usamos ponto como decimal ao salvar
    parse_dates=True
)

# Converter a série para um array NumPy
serie_np = df_inflacao.iloc[:, 0].values.astype('float32')

# Parâmetros (Mantendo N_OBSERVACAO=12 e K_PREVISAO=1 para rodar o modelo já treinado)
N_OBSERVACAO = 12 
K_PREVISAO = 1     
TRAIN_SIZE = 0.8  

# 1. Separação Sequencial (Ajustamos para o mesmo tamanho do Notebook: 60 no teste)
tamanho_treino = len(serie_np) - 60 
serie_treino = serie_np[:tamanho_treino]
serie_teste = serie_np[tamanho_treino:]

# 2. Normalização
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(serie_treino.reshape(-1, 1))

serie_treino_escalada = scaler.transform(serie_treino.reshape(-1, 1)).flatten()
serie_teste_escalada = scaler.transform(serie_teste.reshape(-1, 1)).flatten()

# 3. Criação das Janelas
X_train, Y_train = criar_dataset_janelas(serie_treino_escalada, N_OBSERVACAO, K_PREVISAO)
X_test, Y_test = criar_dataset_janelas(serie_teste_escalada, N_OBSERVACAO, K_PREVISAO)

# 4. Formato 3D para LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# -------------------- III. TREINAMENTO E AVALIAÇÃO --------------------

# 1. Compilação
model = criar_modelo_lstm(N_OBSERVACAO, K_PREVISAO)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print(model.summary())

# 2. Treinamento
print("\n--- INICIANDO TREINAMENTO DO MODELO LSTM ---")
history = model.fit(
    X_train, 
    Y_train, 
    epochs=100,
    batch_size=32, 
    validation_data=(X_test, Y_test), 
    verbose=0,
    shuffle=False 
)

# -------------------- IV. RESULTADOS --------------------

# 1. Predição e Desescalonamento
y_predito_escalado = model.predict(X_test)
Y_test_real = scaler.inverse_transform(Y_test)
y_predito_real = scaler.inverse_transform(y_predito_escalado)

# 2. Cálculo e Impressão das Métricas
rmse = np.sqrt(mean_squared_error(Y_test_real, y_predito_real))
mae = mean_absolute_error(Y_test_real, y_predito_real)
mape = mean_absolute_percentage_error(Y_test_real, y_predito_real)

print("\n--- RESULTADOS DE DESEMPENHO ---")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"MAE (Mean Absolute Error): {mae:.4f}")
print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
plt.savefig(grafico_lstm_previsao.png)
# Comando para rodar no terminal: python Modelagem_LSTM.py