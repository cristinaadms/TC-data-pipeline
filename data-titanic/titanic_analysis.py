# Análise da base Titanic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# 1. Preparação de Dados
# Criar pasta para gráficos
graficos_dir = os.path.join(os.path.dirname(__file__), 'graficos')
os.makedirs(graficos_dir, exist_ok=True)

# Carregar a base
csv_path = os.path.join(os.path.dirname(__file__), 'Titanic-Dataset.csv')
df = pd.read_csv(csv_path)

# 2. Pré-processamento
# Inspecionar valores ausentes
def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    return mis_val_table_ren_columns

print('Valores ausentes:')
print(missing_values_table(df))

# Tratar valores ausentes (remoção ou imputação)
df = df.dropna()

# Detectar e tratar outliers (usando IQR para variáveis numéricas)
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

num_cols = ['Age', 'Fare', 'SibSp', 'Parch']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=num_cols)
df = remove_outliers(df, num_cols)

# Converter variáveis categóricas para numéricas
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Normalizar variáveis numéricas
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# 3. Visualizações
# Estatísticas descritivas
desc = df.describe().T
desc['var'] = df.var()
print('Estatísticas descritivas:')
print(desc[['mean', '50%', 'std', 'var']])

# Gráficos
plt.figure(figsize=(10,6))
sns.histplot(data=df, x='Age', bins=30, kde=True)
plt.title('Histograma de Idade')
plt.savefig(os.path.join(graficos_dir, 'hist_age.png'))
plt.close()

plt.figure(figsize=(10,6))
sns.boxplot(x='Survived', y='Fare', data=df)
plt.title('Boxplot: Fare vs Sobrevivência')
plt.savefig(os.path.join(graficos_dir, 'box_fare_survived.png'))
plt.close()

plt.figure(figsize=(10,6))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)
plt.title('Dispersão: Idade vs Fare')
plt.savefig(os.path.join(graficos_dir, 'scatter_age_fare.png'))
plt.close()

plt.figure(figsize=(10,6))
sns.barplot(x='Pclass', y='Survived', data=df, ci=None)
plt.title('Sobrevivência por Classe')
plt.savefig(os.path.join(graficos_dir, 'bar_pclass_survived.png'))
plt.close()

# 4. Análise de Padrões e Tendências
# Correlações
corr = df.corr()
print('Correlação com sobrevivência:')
print(corr['Survived'].sort_values(ascending=False))

# Relação entre variáveis e sobrevivência
print('Sobrevivência média por sexo:')
print(df.groupby('Sex')['Survived'].mean())

print('Sobrevivência média por classe:')
print(df.groupby('Pclass')['Survived'].mean())

print('Sobrevivência média por idade:')
print(df.groupby('Age')['Survived'].mean().head())

print('Sobrevivência média por tarifa:')
print(df.groupby('Fare')['Survived'].mean().head())

print('Análise concluída. Gráficos salvos como PNG.')
