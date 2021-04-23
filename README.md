# Sobre o projeto
Esse projeto tem como objetivo criar um modelo capaz de identificar o sexo de um paciente com base em algumas informações do mesmo, bem como inferir novos dados ao modelo.
- Importação dos dados
- Análise dos dados
- Tratamento dos dados
- Normalização dos dados
- Separação dos dados de treinamento e teste
- Treinamento do modelo (Rede neural, Random Forest, XGB)
- Validação do Modelo
- Inferência de novos dados ao modelo

# Pré requisitos 
A aplicação depedende dos seguintes componentes.

- Python > 3.6.x 
- Git 

# Criação virtualenv do Python
Criar um ambiente virtual. 
- Abra o anaconda prompt ou o prompt (devidamente setado a variável de ambiente do python e git).
- Crie um diretório para o projeto

## Criar
```
virtualenv 'nome_ambiente'
```
## Ativar (Windows)
```
nome_ambiente/Scripts/activate
```
## Ativar (Mac OS / Linux)
```
source nome_ambiente/bin/activate
```

# Baixar Projeto
- Após ativar o ambiente vá ao diretório que deseja baixar o projeto.
```
git clone https://github.com/caio-cunha/projeto-portal-telemedicina.git
```

# Instalar dependências
- Entre no projeto baixado e digite o comando abaixo
```
python -m pip install -r requirements.txt
```

# Executar script de predição utilizando o modelo treinado
```
(ambiente venv) python sex_predictor.py --input_file newsample.csv
```

# Executar script de treinamento
```
Entrar no script treinador_modelo.ipynb e seguir as orientações do relatório
```


