import os    
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def processar_e_salvar_dados(nome_arquivo_entrada, nome_arquivo_saida):
    if not os.path.exists(nome_arquivo_entrada):
        print(f"Erro: O arquivo '{nome_arquivo_entrada}' não foi encontrado.")
        return None

    try:
        df = pd.read_csv(
            nome_arquivo_entrada,
            sep=';',
            encoding='latin1',
            header=0,
            dtype=str 
        )

        if len(df.columns) >= 2:
            df = df.iloc[:, :2]
            df.columns = ['Data', 'Inflacao_Alim']

        df['Data'] = pd.to_datetime(df['Data'], format='%m/%Y', errors='coerce')
        df = df.dropna(subset=['Data'])

        df['Inflacao_Alim'] = df['Inflacao_Alim'].str.replace(',', '.')
        df['Inflacao_Alim'] = df['Inflacao_Alim'].astype(float)

        df = df.set_index('Data').sort_index()

        print(f"Processamento concluído: {df.shape[0]} registros válidos.")

        df.to_csv(nome_arquivo_saida, sep=';', decimal=',', encoding='utf-8-sig')
        print(f"Arquivo salvo com sucesso: {nome_arquivo_saida}")
        
        return df

    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        return None

arquivo_original = 'STP-20251124190950218.csv'
arquivo_novo = 'dados_inflacao_corrigidos.csv'

df_final = processar_e_salvar_dados(arquivo_original, arquivo_novo)

if df_final is not None:
    print("\nPrimeiras linhas do arquivo gerado:")
    print(df_final.head())
