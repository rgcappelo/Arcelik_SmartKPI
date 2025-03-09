import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

# Configuración de la página
st.set_page_config(
    page_title="Dashboard NPS-E Arçelik",
    page_icon="📊",
    layout="wide",
)

# Estilo CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #3366ff;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #0047AB;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .metric-container {
        background-color: #f5f7ff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .info-text {
        color: #555;
        font-size: 1rem;
    }
    .highlight {
        background-color: #e6f2ff;
        padding: 5px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Título del dashboard
st.markdown('<div class="main-header">Dashboard NPS-E Arçelik</div>', unsafe_allow_html=True)

# Descripción breve
st.markdown("""
<div class="info-text">
Este dashboard visualiza el <span class="highlight">Net Promoter Score Evolutivo (NPS-E)</span>, 
un indicador avanzado que combina la <strong>emocionalidad</strong> (NES), 
<strong>conversión</strong> (RPS) y <strong>contexto social</strong> (CPI) en la experiencia del cliente.
</div>
""", unsafe_allow_html=True)

# Carga de datos
@st.cache_data
def load_data():
    data = {
        "Fecha": ["2022-01", "2022-02", "2022-03", "2022-04", "2022-05", "2022-06",
                  "2022-07", "2022-08", "2022-09", "2022-10", "2022-11", "2022-12",
                  "2023-01", "2023-02", "2023-03", "2023-04", "2023-05", "2023-06",
                  "2023-07", "2023-08", "2023-09", "2023-10", "2023-11", "2023-12",
                  "2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06",
                  "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12",
                  "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
                  "2025-07", "2025-08", "2025-09"],
        "NES": [75, 74, 76, 78, 77, 75, 73, 72, 74, 76, 79, 78,
                80, 81, 82, 79, 77, 76, 75, 74, 73, 72, 74, 75,
                76, 78, 80, 81, 82, 83, 84, 85, 83, 82, 81, 80,
                79, 78, 77, 76, 75, 74, 73, 72, 71],
        "RPS": [60, 61, 63, 62, 64, 66, 67, 68, 65, 63, 62, 61,
                60, 62, 64, 65, 67, 66, 64, 63, 62, 60, 58, 57,
                59, 61, 63, 64, 66, 67, 68, 69, 67, 66, 65, 64,
                63, 62, 61, 60, 59, 58, 57, 56, 55],
        "CPI": [50, 52, 54, 53, 55, 56, 57, 58, 55, 53, 52, 51,
                50, 52, 53, 54, 55, 56, 57, 58, 56, 54, 52, 51,
                50, 52, 54, 55, 57, 58, 59, 60, 58, 56, 55, 54,
                53, 52, 51, 50, 49, 48, 47, 46, 45]
    }
    
    df = pd.DataFrame(data)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df["NPS-E"] = (0.4 * df["NES"]) + (0.4 * df["RPS"]) + (0.2 * df["CPI"])
    
    # Marcar datos históricos vs proyectados
    actual_date = datetime(2024, 10, 31)  # Fecha de corte conocida
    df["Tipo"] = df["Fecha"].apply(lambda x: "Histórico" if x <= actual_date else "Proyectado")
    
    # Anotar tendencias
    df["Tendencia_NPS_E"] = df["NPS-E"].diff().apply(lambda x: "▲" if x > 0 else "▼" if x < 0 else "◆")
    
    # Crear trimestres para agregación
    df["Trimestre"] = df["Fecha"].dt.to_period('Q').astype(str)
    
    # Categorías para análisis
    conditions = [
        (df["NPS-E"] >= 75),
        (df["NPS-E"] >= 65) & (df["NPS-E"] < 75),
        (df["NPS-E"] >= 55) & (df["NPS-E"] < 65),
        (df["NPS-E"] < 55)
    ]
    categories = ["Excelente", "Bueno", "Regular", "Necesita mejora"]
    df["Categoría_NPS_E"] = np.select(conditions, categories, default="No definido")
    
    return df

df = load_data()

# Filtro por rango de fechas
col1, col2 = st.columns(2)
with col1:
    min_date = df["Fecha"].min().to_pydatetime()
    max_date = df["Fecha"].max().to_pydatetime()
    start_date = st.date_input("Fecha inicial", min_date, min_value=min_date, max_value=max_date)
with col2:
    end_date = st.date_input("Fecha final", max_date, min_value=min_date, max_value=max_date)

# Convertir a formato datetime para filtrar
start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date)
filtered_df = df[(df["Fecha"] >= start_date) & (df["Fecha"] <= end_date)]

# Línea divisoria
st.markdown("<hr>", unsafe_allow_html=True)

# Panel de métricas clave
st.markdown('<div class="sub-header">Métricas Clave</div>', unsafe_allow_html=True)

# Crear fila de métricas
metric_cols = st.columns(4)

# Último valor de NPS-E
with metric_cols[0]:
    last_npse = filtered_df["NPS-E"].iloc[-1]
    previous_npse = filtered_df["NPS-E"].iloc[-2] if len(filtered_df) > 1 else last_npse
    delta_npse = last_npse - previous_npse
    
    st.metric(
        label="NPS-E Actual",
        value=f"{last_npse:.1f}",
        delta=f"{delta_npse:.1f}",
        delta_color="normal"
    )

# Promedio de NES
with metric_cols[1]:
    avg_nes = filtered_df["NES"].mean()
    st.metric(
        label="NES Promedio (Emocional)",
        value=f"{avg_nes:.1f}",
    )

# Promedio de RPS
with metric_cols[2]:
    avg_rps = filtered_df["RPS"].mean()
    st.metric(
        label="RPS Promedio (Conversión)",
        value=f"{avg_rps:.1f}",
    )

# Promedio de CPI
with metric_cols[3]:
    avg_cpi = filtered_df["CPI"].mean()
    st.metric(
        label="CPI Promedio (Cultural)",
        value=f"{avg_cpi:.1f}",
    )

# Línea divisoria
st.markdown("<hr>", unsafe_allow_html=True)

# Función para crear gráficos
def create_plots(df):
    # 1. Gráfico de líneas: Evolución del NPS-E
    def plot_npse_evolution():
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Separar datos históricos y proyectados
        historical = df[df["Tipo"] == "Histórico"]
        projected = df[df["Tipo"] == "Proyectado"]
        
        # Gráfico con datos históricos (línea sólida)
        ax.plot(historical["Fecha"], historical["NPS-E"], 
                marker='o', linestyle='-', linewidth=2, 
                color='#1f77b4', label='NPS-E Histórico')
        
        # Gráfico con datos proyectados (línea punteada)
        if not projected.empty:
            ax.plot(projected["Fecha"], projected["NPS-E"], 
                    marker='o', linestyle='--', linewidth=2, 
                    color='#ff7f0e', label='NPS-E Proyectado')
        
        # Elementos visuales adicionales
        ax.axhline(y=75, color='green', linestyle='--', alpha=0.5, label='Excelente (75+)')
        ax.axhline(y=65, color='#85C1E9', linestyle='--', alpha=0.5, label='Bueno (65-75)')
        ax.axhline(y=55, color='orange', linestyle='--', alpha=0.5, label='Regular (55-65)')
        
        # Formato de gráfico
        ax.set_title('Evolución del NPS-E a lo largo del tiempo', fontsize=16, fontweight='bold')
        ax.set_xlabel('Fecha', fontsize=12)
        ax.set_ylabel('Valor del NPS-E', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')
        
        # Formato de fechas en el eje X
        plt.gcf().autofmt_xdate()
        date_format = mdates.DateFormatter('%Y-%m')
        ax.xaxis.set_major_formatter(date_format)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        
        # Añadir anotaciones para los valores más destacados
        max_npse = df["NPS-E"].max()
        max_date = df.loc[df["NPS-E"].idxmax(), "Fecha"]
        min_npse = df["NPS-E"].min()
        min_date = df.loc[df["NPS-E"].idxmin(), "Fecha"]
        
        ax.annotate(f'Máx: {max_npse:.1f}',
                   xy=(max_date, max_npse),
                   xytext=(10, 10),
                   textcoords='offset points',
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                   fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
        
        ax.annotate(f'Mín: {min_npse:.1f}',
                   xy=(min_date, min_npse),
                   xytext=(-10, -20),
                   textcoords='offset points',
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                   fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
        
        # Ajustar límites del eje Y para mejor visualización
        y_min = max(0, df["NPS-E"].min() - 5)
        y_max = min(100, df["NPS-E"].max() + 5)
        ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        return fig
    
    # 2. Gráfico de dispersión: NES vs RPS
    def plot_nes_vs_rps():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Colores según el valor de CPI
        scatter = ax.scatter(df["NES"], df["RPS"], 
                   c=df["CPI"], cmap='viridis', 
                   s=80, alpha=0.7, edgecolors='w')
        
        # Añadir línea de tendencia
        z = np.polyfit(df["NES"], df["RPS"], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df["NES"].min(), df["NES"].max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f'Tendencia (r={np.corrcoef(df["NES"], df["RPS"])[0,1]:.2f})')
        
        # Formato de gráfico
        ax.set_title('Relación entre Emoción (NES) y Conversión (RPS)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Net Emotion Score (NES)', fontsize=12)
        ax.set_ylabel('Revenue Promoter Score (RPS)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Barra de color para CPI
        cbar = plt.colorbar(scatter)
        cbar.set_label('Cultural Promoter Index (CPI)', fontsize=10)
        
        # Añadir etiquetas para puntos destacados
        for i, row in df.iterrows():
            if (row["NES"] == df["NES"].max() or 
                row["NES"] == df["NES"].min() or 
                row["RPS"] == df["RPS"].max() or 
                row["RPS"] == df["RPS"].min()):
                ax.annotate(row["Fecha"].strftime('%Y-%m'),
                           (row["NES"], row["RPS"]),
                           xytext=(5, 5),
                           textcoords='offset points',
                           fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
        
        ax.legend(loc='lower right')
        plt.tight_layout()
        return fig
    
    # 3. Gráfico de barras: CPI por trimestre
    def plot_cpi_by_quarter():
        # Agrupar por trimestre
        quarterly_data = df.groupby('Trimestre')['CPI'].mean().reset_index()
        quarterly_data = quarterly_data.sort_values('Trimestre')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Crear paleta de colores
        colors = plt.cm.viridis(np.linspace(0, 1, len(quarterly_data)))
        
        # Crear barras
        bars = ax.bar(quarterly_data['Trimestre'], quarterly_data['CPI'], 
                    color=colors, width=0.6, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Añadir valor encima de cada barra
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)
        
        # Formato del gráfico
        ax.set_title('Cultural Promoter Index (CPI) por Trimestre', fontsize=16, fontweight='bold')
        ax.set_xlabel('Trimestre', fontsize=12)
        ax.set_ylabel('CPI Promedio', fontsize=12)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Rotar etiquetas del eje X para mejor visualización
        plt.xticks(rotation=45, ha='right')
        
        # Añadir línea de referencia
        ax.axhline(y=df['CPI'].mean(), color='red', linestyle='--', alpha=0.7, 
                  label=f'Promedio general: {df["CPI"].mean():.1f}')
        
        ax.legend(loc='best')
        ax.set_ylim(0, max(quarterly_data['CPI']) * 1.15)  # Dar un poco de espacio arriba
        
        plt.tight_layout()
        return fig
    
    # 4. Gráfico de radar: Comparación entre NES, RPS y CPI
    def plot_radar_comparison():
        # Calcular promedios
        avg_data = df.groupby('Tipo')[['NES', 'RPS', 'CPI']].mean().reset_index()
        
        # Preparar el gráfico
        fig = plt.figure(figsize=(10, 8))
        
        # Crear un subplot con proyección polar para el radar
        ax = fig.add_subplot(111, polar=True)
        
        # Categorías (dimensiones)
        categories = ['NES', 'RPS', 'CPI']
        N = len(categories)
        
        # Completar el círculo repitiendo el primer valor
        historico_values = avg_data[avg_data['Tipo'] == 'Histórico'][categories].values.flatten().tolist()
        historico_values += historico_values[:1]
        
        proyectado_values = []
        if 'Proyectado' in avg_data['Tipo'].values:
            proyectado_values = avg_data[avg_data['Tipo'] == 'Proyectado'][categories].values.flatten().tolist()
            proyectado_values += proyectado_values[:1]
        
        # Ángulos para cada eje
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Cerrar el círculo
        
        # Dibujar ejes y etiquetas
        plt.xticks(angles[:-1], categories, fontsize=12)
        
        # Dibujar límites de ejes y gráfico
        ax.set_rlabel_position(0)
        plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="grey", size=10)
        plt.ylim(0, 100)
        
        # Dibujar datos históricos
        ax.plot(angles, historico_values, linewidth=2, linestyle='solid', color='#1f77b4', label='Histórico')
        ax.fill(angles, historico_values, '#1f77b4', alpha=0.25)
        
        # Dibujar datos proyectados si existen
        if proyectado_values:
            ax.plot(angles, proyectado_values, linewidth=2, linestyle='dashed', color='#ff7f0e', label='Proyectado')
            ax.fill(angles, proyectado_values, '#ff7f0e', alpha=0.25)
        
        # Título y leyenda
        plt.title('Comparación de Dimensiones del NPS-E', fontsize=16, fontweight='bold', y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        return fig
    
    return {
        "evolución": plot_npse_evolution(),
        "dispersión": plot_nes_vs_rps(),
        "barras": plot_cpi_by_quarter(),
        "radar": plot_radar_comparison()
    }

# Crear los gráficos
plots = create_plots(filtered_df)

# Mostrar gráficos
st.markdown('<div class="sub-header">1️⃣ Evolución de la Percepción General del Cliente</div>', unsafe_allow_html=True)
st.pyplot(plots["evolución"])
st.markdown("""
<div class="info-text">
Este gráfico muestra la evolución del <strong>NPS-E</strong> a lo largo del tiempo, permitiendo identificar tendencias
y patrones en la percepción general del cliente. Las líneas punteadas representan umbrales de clasificación.
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="sub-header">2️⃣ Relación entre Emoción y Conversión</div>', unsafe_allow_html=True)
st.pyplot(plots["dispersión"])
st.markdown("""
<div class="info-text">
Este gráfico de dispersión muestra la relación entre el <strong>Net Emotion Score (NES)</strong> y 
el <strong>Revenue Promoter Score (RPS)</strong>, mientras que el color indica el nivel de 
<strong>Cultural Promoter Index (CPI)</strong>. Permite evaluar cómo las emociones positivas se traducen en conversiones efectivas.
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="sub-header">3️⃣ Impacto Cultural por Trimestre</div>', unsafe_allow_html=True)
st.pyplot(plots["barras"])
st.markdown("""
<div class="info-text">
Este gráfico de barras muestra el <strong>Cultural Promoter Index (CPI)</strong> promedio por trimestre,
permitiendo evaluar el peso de la marca dentro del ecosistema cultural y social a lo largo del tiempo.
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="sub-header">4️⃣ Equilibrio entre Dimensiones</div>', unsafe_allow_html=True)
st.pyplot(plots["radar"])
st.markdown("""
<div class="info-text">
Este gráfico de radar muestra la comparación entre las tres dimensiones del NPS-E: 
<strong>NES</strong> (emocional), <strong>RPS</strong> (conversión) y <strong>CPI</strong> (social),
permitiendo visualizar el equilibrio entre estos aspectos en la recomendación del cliente.
</div>
""", unsafe_allow_html=True)

# Línea divisoria
st.markdown("<hr>", unsafe_allow_html=True)

# Tabla de datos
st.markdown('<div class="sub-header">Datos
