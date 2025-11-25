import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def carregar_dados_para_analise(nome_arquivo):
    try:
        df = pd.read_csv(
            nome_arquivo,
            sep=';',
            decimal=',',
            encoding='utf-8-sig'
        )
        df['Data'] = pd.to_datetime(df['Data'])
        df = df.set_index('Data').sort_index()
        
        df = df.asfreq('MS')
        
        return df
    except FileNotFoundError:
        print(f"Erro: O arquivo '{nome_arquivo}' não foi encontrado. Gere-o no passo anterior.")
        return None

def gerar_graficos_analise(df):
    # Verifica se há dados nulos gerados pelo 'asfreq' e preenche ou avisa
    if df.isnull().values.any():
        df = df.interpolate(method='linear')

    # 1. Plotagem da Série Histórica Completa
    plt.figure()
    plt.plot(df.index, df['Inflacao_Alim'], label='IPC-Fipe Alimentação', color='#1f77b4')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title('Histórico de Inflação de Alimentos (1997-2025)')
    plt.ylabel('Variação Mensal (%)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. Histograma de Distribuição
    plt.figure()
    sns.histplot(df['Inflacao_Alim'], kde=True, bins=30, color='green')
    plt.title('Distribuição das Variações Mensais (Volatilidade)')
    plt.xlabel('Variação (%)')
    plt.tight_layout()
    plt.show()

    # 3. Decomposição Sazonal
    print("Gerando decomposição sazonal...")
    decomposicao = seasonal_decompose(df['Inflacao_Alim'], model='additive', period=12)
    fig = decomposicao.plot()
    fig.set_size_inches(12, 10)
    plt.tight_layout()
    plt.show()

    # 4. Boxplot Sazonal (Inflação por Mês do Ano)
    df_sazonal = df.copy()
    df_sazonal['Mes'] = df_sazonal.index.month
    plt.figure()
    sns.boxplot(x='Mes', y='Inflacao_Alim', data=df_sazonal, palette="Blues", hue='Mes', legend=False)
    plt.title('Sazonalidade Mensal: Distribuição da Inflação por Mês')
    plt.xlabel('Mês (1=Jan, 12=Dez)')
    plt.tight_layout()
    plt.show()

    # 5. Autocorrelação (ACF) e Autocorrelação Parcial (PACF)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(df['Inflacao_Alim'], lags=24, ax=ax1, title='Autocorrelação (ACF)')
    plot_pacf(df['Inflacao_Alim'], lags=24, ax=ax2, title='Autocorrelação Parcial (PACF)', method='ywm')
    plt.tight_layout()
    plt.show()

arquivo_para_ler = 'dados_inflacao_corrigidos.csv'
df_ipc = carregar_dados_para_analise(arquivo_para_ler)

if df_ipc is not None:
    gerar_graficos_analise(df_ipc)