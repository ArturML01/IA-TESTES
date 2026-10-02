import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

# 1. Carregamento da base Excel
script_dir = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(script_dir, "Online Retail.xlsx")

print("Carregando os dados...")
df = pd.read_excel(excel_path)

# 2. Limpeza dos dados
# Remover registros sem identificação de cliente e transações inválidas/canceladas
df_clean = df.dropna(subset=['CustomerID']).copy()
df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]

# Criar coluna de Valor Total por item
df_clean['TotalSpend'] = df_clean['Quantity'] * df_clean['UnitPrice']

# 3. Engenharia de Recursos: Análise RFM por Cliente
max_date = df_clean['InvoiceDate'].max()
ref_date = max_date + pd.Timedelta(days=1)

rfm = df_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (ref_date - x.max()).days,  # Recency (dias)
    'InvoiceNo': 'nunique',                              # Frequency (compras únicas)
    'TotalSpend': 'sum'                                  # Monetary (gasto total)
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
print(f"Total de clientes únicos identificados: {len(rfm)}")

# 4. Pré-processamento: Transformação Logarítmica + Escalonamento
rfm_log = np.log1p(rfm[['Recency', 'Frequency', 'Monetary']])

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

# 5. Avaliação do Coeficiente de Silhueta para determinar o k ideal
faixa_k = range(2, 11)
scores_silhueta = []

print("\n=== Avaliação do Coeficiente de Silhueta ===")
for k in faixa_k:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(rfm_scaled)
    
    score = silhouette_score(rfm_scaled, labels)
    scores_silhueta.append(score)
    print(f"k = {k}: Coeficiente de Silhueta = {score:.4f}")

# Determinar o melhor k
melhor_k = faixa_k[np.argmax(scores_silhueta)]
melhor_score = max(scores_silhueta)
print(f"\nO k ideal recomendado pelo maior Coeficiente de Silhueta é k = {melhor_k} ({melhor_score:.4f})")

# Plot do Gráfico do Coeficiente de Silhueta
plt.figure(figsize=(8, 4))
plt.plot(faixa_k, scores_silhueta, marker='o', linestyle='--', color='b')
plt.title('Coeficiente de Silhueta para Escolha de k (Online Retail RFM)')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Coeficiente de Silhueta')
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Aplicação do K-Means com o k ideal e análise dos resultados
kmeans_final = KMeans(n_clusters=melhor_k, random_state=42, n_init=10)
rfm['Cluster'] = kmeans_final.fit_predict(rfm_scaled)

print("\n=== Perfil Médio de Cada Cluster (Valores Reais em RFM) ===")
perfil = rfm.groupby('Cluster').agg({
    'CustomerID': 'count',
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean'
}).rename(columns={'CustomerID': 'Qtd_Clientes'})

# Formatação para melhor leitura dos valores médios
perfil['Recency'] = perfil['Recency'].round(1)
perfil['Frequency'] = perfil['Frequency'].round(1)
perfil['Monetary'] = perfil['Monetary'].map('£{:,.2f}'.format)

print(perfil)

# Criar a figura com projeção 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Dispersão dos pontos dos clientes (usando dados escalonados)
scatter = ax.scatter(
    rfm_scaled[:, 0],  # Recência
    rfm_scaled[:, 1],  # Frequência
    rfm_scaled[:, 2],  # Valor Monetário
    c=rfm['Cluster'], 
    cmap='viridis', 
    s=30, 
    alpha=0.5
)

# Plotar os centroides dos grupos
centroides = kmeans_final.cluster_centers_
ax.scatter(
    centroides[:, 0],
    centroides[:, 1],
    centroides[:, 2],
    c='red', 
    marker='X', 
    s=250, 
    linewidths=2,
    label='Centroides'
)

# Rótulos e estilização
ax.set_title('Dispersão 3D dos Clusters no Espaço RFM Padronizado', fontsize=12, pad=15)
ax.set_xlabel('Recência (Escalonada)', fontsize=10)
ax.set_ylabel('Frequência (Escalonada)', fontsize=10)
ax.set_zlabel('Monetário (Escalonado)', fontsize=10)
ax.legend(loc='upper right')

# Barra de cores para referência dos grupos
cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Cluster')

plt.tight_layout()
plt.show()