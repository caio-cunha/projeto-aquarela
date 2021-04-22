import argparse
import csv
import pandas as pd
import sys
import pickle
import numpy as np

from sklearn.preprocessing import StandardScaler

def parser():
    """
    Essa função tem como objetivo ler o parametro --input_file digitado no comando via CMD
    _ _ _ _ _ _ _ _ _ _ _ _ 
    Output: 
        Argparse
    """
    parser = argparse.ArgumentParser(description='Ler arquivo csv')
    parser.add_argument('-i','--input_file', type=argparse.FileType('r'), help='Ler arquivo csv')
    args = parser.parse_args()
    
    return args

def transformar_csv_dataframe(csv):
    """
    Essa função tem como objetivo transformar o csv lido no CMD em um dataframe
    _ _ _ _ _ _ _ _ _ _ _ _ 
    Input: 
        csv: csv_reader object
    Output: 
        df: pandas Dataframe do csv lido
    """
    
    df = pd.DataFrame(csv_reader)
    df.columns = df.iloc[0]
    df = df.drop(0)
    df = df.reset_index(drop=True)
    df = df.set_index(['index'])
    
    return df

def tratar_dados_para_predicao(df):
    """
    Essa função tem como objetivo tratar os dados importados via CMD.
    _ _ _ _ _ _ _ _ _ _ _ _ 
    Input: 
        df: dataFrame pandas com os dados importados
    Output: 
        df_tratado: dataFrame pandas com os dados tratados
    """
    ##Transformamando valores vazios em NANs
    df = df.replace(r'', np.nan, regex=True)   

    ##Tratando as valores nans
    df = df.fillna(method='ffill')
    
    ##transformando todos os valores em float
    df_tratado = df.round(0).astype(float)
    
    return df_tratado

def predicao_modelo(df):
    """
    Essa funçã tem como objetivo de carregar o modelo e realizar predições sobre o modelo treinado
    _ _ _ _ _ _ _ _ _ _ _ _ _ 
    Input:
        df: dataframe Pandas com os dados
    Output: 
        df_saida: dataframe com as predições
    """
    ##Modelo treinado com o notbook 'treinador_modelo.ipnyb'
    filename = 'modelo_treinado.sav'
    
    ##Carregando modelo treinado
    modelo_carregado = pickle.load(open(filename, 'rb'))
    if(modelo_carregado):
        print('Modelo carregado com sucesso!')
                           
    ##Predição do modelo treinado
    predicao = []
    for array_valores in df.values:
         predicao.append(modelo_carregado.predict(array_valores.reshape(1, -1)))
           
    return predicao
       
    
if __name__ == '__main__':
    
    ##Parser para obter CSV escolhido pelo usuário no CMD
    args = parser()
    
    ##Ler CSV recebido no PARSER
    csv_reader = csv.reader(args.input_file)
    
    ##Transformar CSV READER em DATAFRAME
    df = transformar_csv_dataframe(csv_reader)
    
    ##Tratar os dados 
    df_tratado = tratar_dados_para_predicao(df)
    
    ##Predição modelo treinado
    preditos = predicao_modelo(df_tratado)
    df_preditos = pd.DataFrame(preditos,columns=['sex'])
    
    ##Transformando valores 0 e 1 em M e F
    df_preditos.loc[(df_preditos['sex'] == 1,"sex")] = 'M'
    df_preditos.loc[(df_preditos['sex'] == 0,"sex")] = 'F'
    
    ##Salvando CSV com RESULTADOS
    df_preditos.to_csv ('newsample_PREDICTIONS_{Caio_Henrique_Oliveira_Cunha}.csv', index=False, header=True)    
    print('Resultados salvos com sucesso!')
    
    
    

    
    
    
    
