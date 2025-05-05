import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('dataset_8.csv')
graph_n = 0

def save_fig():
    global graph_n
    plt.savefig(f'Graphs/graph_{graph_n}')
    graph_n += 1

############################################################################################################

descricao = df.describe(include='all').transpose()
descricao['moda'] = df.mode().iloc[0]
descricao['mediana'] = df.median(numeric_only=True)
descricao['q3'] = df.quantile(0.75, numeric_only=True)
descricao['q1'] = df.quantile(0.25, numeric_only=True)
descricao['iqr'] = descricao['q3'] - descricao['q1']
descricao['nulos'] = df.isnull().sum()
descricao['desvio_padrao'] = df.std(numeric_only=True)

print(descricao)

cat_vars = ['sistema_operacional', 'tipo_hd', 'tipo_processador']
plt.figure(figsize=(15, 5))
for i, var in enumerate(cat_vars, 1):
    plt.subplot(1, 3, i)
    df[var].value_counts().plot(kind='bar')
    plt.title(f'Distribuição de {var}')
    plt.xticks(rotation=45)
plt.tight_layout()
save_fig()

num_vars = ['cpu_cores', 'ram_gb', 'latencia_ms', 'armazenamento_tb', 'tempo_resposta']
plt.figure(figsize=(15, 10))
for i, var in enumerate(num_vars, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[var], kde=True)
    plt.title(f'Distribuição de {var}')
plt.tight_layout()
save_fig()

############################################################################################################

vars_numericas = ['cpu_cores', 'ram_gb', 'armazenamento_tb', 'latencia_ms', 'tempo_resposta']
matriz_corr = df[vars_numericas].corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(matriz_corr))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

ax = sns.heatmap(matriz_corr, annot=True, fmt='.2f', cmap=cmap, vmin=-1, vmax=1,
               square=True, linewidths=0.5, cbar_kws={"shrink": .8})

plt.title('Matriz de Correlação - Variáveis Numéricas', fontsize=14)
plt.tight_layout()
save_fig()

############################################################################################################

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.boxplot(x='sistema_operacional', y='tempo_resposta', data=df)
plt.title('Tempo de Resposta por Sistema Operacional')

plt.subplot(2, 2, 2)
sns.boxplot(x='tipo_hd', y='tempo_resposta', data=df)
plt.title('Tempo de Resposta por Tipo de armazenamento')

plt.subplot(2, 2, 3)
sns.boxplot(x='tipo_processador', y='tempo_resposta', data=df)
plt.title('Tempo de Resposta por Tipo de Processador')

plt.tight_layout()
save_fig()

############################################################################################################

numeric_cols = ['cpu_cores', 'ram_gb', 'latencia_ms', 'armazenamento_tb']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

categorical_cols = ['sistema_operacional', 'tipo_hd', 'tipo_processador']
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0]).astype(str)
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

for col in df_encoded.columns:
    df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')

df_encoded = df_encoded.dropna(axis=1, how='all')

y = df_encoded['tempo_resposta'].astype(float)
X = df_encoded.drop('tempo_resposta', axis=1).select_dtypes(include=[np.number])
X = sm.add_constant(X)

modelo = sm.OLS(y, X).fit()

print("\n" + "="*80)
print("Modelo de regressão")
print("="*80)
print(modelo.summary())


vif_data = pd.DataFrame(columns=['Variável', 'VIF'])

X_numeric = X.select_dtypes(include=[np.number])
X_filtered = X_numeric.loc[:, X_numeric.std() > 1e-6]

for i, col in enumerate(X_filtered.columns):
    try:
        vif = variance_inflation_factor(X_filtered.values, i)
        if np.isfinite(vif):
            vif_data.loc[len(vif_data)] = [col, vif]
    except:
        continue

vif_data = vif_data.sort_values('VIF', ascending=False)

print("\n" + "="*80)
print("Multicolinearidade")
print("="*80)
print(vif_data)

print("\n" + "="*80)
print("Heterocastidade")
print("="*80)

plt.figure(figsize=(10, 6))
sns.regplot(x=modelo.fittedvalues, y=modelo.resid, lowess=True, 
            scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.axhline(y=0, color='k', linestyle='--')
plt.title('Resíduos vs Valores Ajustados', fontsize=14)
plt.xlabel('Valores Ajustados')
plt.ylabel('Resíduos')
save_fig()

bp_test = sms.het_breuschpagan(modelo.resid, modelo.model.exog)
print("\nTeste de Breusch-Pagan:")
print(f"Estatística: {bp_test[0]:.4f}, p-valor: {bp_test[1]:.4f}")
print("Heterocedasticidade" if bp_test[1] < 0.05 else "Sem heterocedasticidade")

############################################################################################################

modelo_completo = sm.OLS(y, X).fit()
print("\nImpacto das variáveis")
print("=" * 80)

resultados_teste_f = []

for coluna in X.columns:
    if coluna != 'const':
        X_restrito = X.drop(coluna, axis=1)
        modelo_restrito = sm.OLS(y, X_restrito).fit()
        
        n = len(y)
        k_full = X.shape[1]
        k_reduced = X_restrito.shape[1]
        df1 = k_full - k_reduced
        df2 = n - k_full
        
        sse_full = sum(modelo_completo.resid**2)
        sse_reduced = sum(modelo_restrito.resid**2)
        
        f_stat = ((sse_reduced - sse_full) / df1) / (sse_full / df2)
        
        import scipy.stats as stats
        p_valor = 1 - stats.f.cdf(f_stat, df1, df2)
        
        resultados_teste_f.append({
            'Variável': coluna,
            'Estatística F': round(f_stat, 6),
            'p-valor': round(p_valor, 6),     
            'R² Modelo Restrito': round(modelo_restrito.rsquared_adj, 4)
        })

resultados_df = pd.DataFrame(resultados_teste_f)
resultados_df = resultados_df.sort_values('p-valor', ascending=False)

print(resultados_df)

############################################################################################################

X1 = df_encoded.drop('tempo_resposta', axis=1).select_dtypes(include=[np.number])
X1 = sm.add_constant(X1)
modelo1 = sm.OLS(y, X1).fit()

print("\nTodas as variáveis")
print("=" * 80)
print(modelo1.summary())

X2 = df_encoded.drop(['tempo_resposta', 'armazenamento_tb'], axis=1).select_dtypes(include=[np.number])
X2 = sm.add_constant(X2)
modelo2 = sm.OLS(y, X2).fit()

print("\nSem a variável armazenamento_tb")
print("=" * 80)
print(modelo2.summary())

print("\nComparação")
print("=" * 80)
comparativo = pd.DataFrame({
    'Modelo 1 (Completo)': [modelo1.rsquared, modelo1.rsquared_adj, modelo1.fvalue, modelo1.f_pvalue, X1.shape[1]-1],
    'Modelo 2 (Sem armazenamento_tb)': [modelo2.rsquared, modelo2.rsquared_adj, modelo2.fvalue, modelo2.f_pvalue, X2.shape[1]-1]
}, index=['R²', 'R² Ajustado', 'Estatística F', 'p-valor (F)', 'Número de variáveis'])

pd.set_option('display.float_format', '{:.4f}'.format)
print(comparativo)

############################################################################################################