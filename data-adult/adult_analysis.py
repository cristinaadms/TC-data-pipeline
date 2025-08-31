# Análise da base Adult (Census Income)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# 1. Preparação de Dados
# Carregar a base
graficos_dir = os.path.join(os.path.dirname(__file__), 'graficos')
os.makedirs(graficos_dir, exist_ok=True)
col_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]
df = pd.read_csv('adult.csv', header=None, names=col_names, na_values='?')

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

num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
# Remover linhas com valores não numéricos
df = df.dropna(subset=num_cols)
df = remove_outliers(df, num_cols)

# Converter variáveis categóricas para numéricas
cat_cols = df.select_dtypes(include=['object']).columns.drop('income')
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Converter variável alvo (income)
df['income'] = df['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)

# Normalizar variáveis numéricas
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# 3. Visualizações
# Estatísticas descritivas
desc = df.describe().T
# Adicionar variância manualmente
desc['var'] = df.var()
print('Estatísticas descritivas:')
print(desc[['mean', '50%', 'std', 'var']])

# Gráficos
plt.figure(figsize=(10,6))
sns.histplot(data=df, x='age', bins=30, kde=True)
plt.title('Histograma de Idade')
plt.savefig(os.path.join(graficos_dir, 'hist_age.png'))
plt.close()

plt.figure(figsize=(10,6))
sns.boxplot(x='income', y='hours-per-week', data=df)
plt.title('Boxplot: Horas por semana vs Renda')
plt.savefig(os.path.join(graficos_dir, 'box_hours_income.png'))
plt.close()

plt.figure(figsize=(10,6))
sns.scatterplot(x='age', y='hours-per-week', hue='income', data=df)
plt.title('Dispersão: Idade vs Horas por semana')
plt.savefig(os.path.join(graficos_dir, 'scatter_age_hours.png'))
plt.close()

plt.figure(figsize=(10,6))
sns.barplot(x='education-num', y='income', data=df, ci=None)
plt.title('Renda média por escolaridade')
plt.savefig(os.path.join(graficos_dir, 'bar_education_income.png'))
plt.close()

# 4. Análise de Padrões e Tendências
# Correlações
corr = df.corr()
print('Correlação com renda:')
print(corr['income'].sort_values(ascending=False))

# Relação entre variáveis e renda
print('Renda média por ocupação:')
print(df.groupby('occupation')['income'].mean())

print('Renda média por escolaridade:')
print(df.groupby('education-num')['income'].mean())

print('Renda média por idade:')
print(df.groupby('age')['income'].mean().head())

print('Renda média por horas de trabalho:')
print(df.groupby('hours-per-week')['income'].mean().head())

print('Análise concluída. Gráficos salvos como PNG.')
