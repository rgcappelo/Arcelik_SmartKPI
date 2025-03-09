import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import seaborn as sns
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(
    page_title="Dashboard NPS-E Arçelik",
    page_icon="📊",
    layout="wide"
)

# Título y descripción
st.title("📊 Dashboard NPS-E Arçelik")
st.markdown("""
Este dashboard permite visualizar y analizar el NPS-E (Net Promoter Score Emocional) de Arçelik 
desde distintas perspectivas: evolución temporal, correlación entre emociones y conversiones, 
presencia cultural y alineación de dimensiones.
""")

# Carga de datos
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
    
    # Datos regionales simulados para el CPI (ya que no están en los datos originales)
    regions = ["Europa Norte", "Europa Sur", "Asia Central", "Asia Este", "Medio Oriente", "África Norte"]
    cpi_regions = {
        region: np.random.randint(40, 65) for region in regions
    }
    
    return df, cpi_regions

df, cpi_regions = load_data()

# Configuración del filtro de fechas
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
        "Fecha de inicio",
        value=df["Fecha"].min().date(),
        min_value=df["Fecha"].min().date(),
        max_value=df["Fecha"].max().date()
    )
with col2:
    end_date = st.date_input(
        "Fecha de fin",
        value=df["Fecha"].max().date(),
        min_value=df["Fecha"].min().date(),
        max_value=df["Fecha"].max().date()
    )

# Conversión a datetime para filtrar
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Filtrado de datos
filtered_df = df[(df["Fecha"] >= start_date) & (df["Fecha"] <= end_date)]

# Función para crear gráfico de radar
def radar_factory(num_vars, frame='circle'):
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    class RadarAxes(PolarAxes):
        name = 'radar'
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')
            
        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)
            
        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
                
        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)
                
        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)
            
        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)
                
        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                spine_type = 'circle'
                verts = unit_poly_verts(num_vars)
                verts.append(verts[0])
                path = Path(verts)
                spine = Spine(self, spine_type, path)
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def unit_poly_verts(num_vars):
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    verts = [(0.5 * np.cos(t) + 0.5, 0.5 * np.sin(t) + 0.5) for t in theta]
    return verts

# Configuración de estilo
colors = {
    'NPS-E': '#2980b9',
    'NES': '#e74c3c',
    'RPS': '#27ae60',
    'CPI': '#8e44ad'
}

plt.style.use('seaborn-v0_8-darkgrid')
st.markdown("---")

# 1. Gráfico de líneas: Evolución del NPS-E
st.subheader("1️⃣ Evolución temporal del NPS-E")
st.markdown("*Cómo varía la percepción general de recomendación del cliente a lo largo del tiempo.*")

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(filtered_df['Fecha'], filtered_df['NPS-E'], marker='o', linestyle='-', color=colors['NPS-E'], linewidth=2)

# Proyección (para datos futuros si están presentes)
future_data = filtered_df[filtered_df['Fecha'] > datetime.now()]
if not future_data.empty:
    ax1.plot(future_data['Fecha'], future_data['NPS-E'], linestyle='--', color=colors['NPS-E'], alpha=0.7)
    ax1.axvline(x=datetime.now(), color='gray', linestyle='--', alpha=0.5)
    ax1.text(datetime.now() + timedelta(days=15), min(filtered_df['NPS-E']), 'Proyección', 
             fontsize=10, color='gray', ha='left', va='bottom')

# Líneas de referencia para NPS-E
ax1.axhline(y=filtered_df['NPS-E'].mean(), color='gray', linestyle='--', alpha=0.5)
ax1.text(filtered_df['Fecha'].min(), filtered_df['NPS-E'].mean() + 1, f'Media: {filtered_df["NPS-E"].mean():.1f}', 
         fontsize=10, color='gray')

# Formato del eje x para mostrar meses
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)

# Etiquetas y título
ax1.set_xlabel('Fecha')
ax1.set_ylabel('Valor NPS-E')
ax1.set_title('Evolución del NPS-E', fontsize=14)

# Mejoras visuales
ax1.grid(True, alpha=0.3)
fig1.tight_layout()

st.pyplot(fig1)

# 2. Gráfico de dispersión: NES vs RPS
st.markdown("---")
st.subheader("2️⃣ Relación entre emoción y conversión")
st.markdown("*Somos capaces de transformar las emociones positivas en conversiones efectivas.*")

fig2, ax2 = plt.subplots(figsize=(10, 6))
scatter = ax2.scatter(filtered_df['NES'], filtered_df['RPS'], 
                      c=filtered_df['Fecha'].astype(int), 
                      cmap='viridis', 
                      s=80, alpha=0.7)

# Línea de tendencia
z = np.polyfit(filtered_df['NES'], filtered_df['RPS'], 1)
p = np.poly1d(z)
ax2.plot(filtered_df['NES'], p(filtered_df['NES']), "r--", alpha=0.7)

# Correlación
corr = filtered_df['NES'].corr(filtered_df['RPS'])
ax2.text(0.05, 0.95, f'Correlación: {corr:.2f}', transform=ax2.transAxes, 
         fontsize=12, va='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

# Etiquetas y título
ax2.set_xlabel('Net Emotion Score (NES)')
ax2.set_ylabel('Revenue Promoter Score (RPS)')
ax2.set_title('Relación entre NES y RPS', fontsize=14)

# Colorbar para mostrar la evolución temporal
cbar = plt.colorbar(scatter)
cbar.set_label('Tiempo (cronológico)')

# Mejoras visuales
ax2.grid(True, alpha=0.3)
fig2.tight_layout()

st.pyplot(fig2)

# 3. Gráfico de barras: CPI por región
st.markdown("---")
st.subheader("3️⃣ Presencia cultural por región")
st.markdown("*Cuánto peso tiene la marca dentro del ecosistema cultural y social.*")

fig3, ax3 = plt.subplots(figsize=(10, 6))

# Ordenar regiones por valor de CPI
sorted_regions = sorted(cpi_regions.items(), key=lambda x: x[1], reverse=True)
regions = [r[0] for r in sorted_regions]
cpi_values = [r[1] for r in sorted_regions]

# Crear barras
bars = ax3.bar(regions, cpi_values, color=colors['CPI'], alpha=0.7)

# Añadir valores encima de las barras
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.0f}', ha='center', va='bottom')

# Etiquetas y título
ax3.set_xlabel('Región')
ax3.set_ylabel('Cultural Promoter Index (CPI)')
ax3.set_title('CPI por Región', fontsize=14)

# Línea de referencia para el CPI promedio
cpi_mean = np.mean(list(cpi_regions.values()))
ax3.axhline(y=cpi_mean, color='red', linestyle='--', alpha=0.5)
ax3.text(0, cpi_mean + 1, f'Media: {cpi_mean:.1f}', color='red')

# Mejoras visuales
ax3.set_ylim(0, max(cpi_values) * 1.15)  # Espacio para etiquetas
ax3.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45, ha='right')
fig3.tight_layout()

st.pyplot(fig3)

# 4. Gráfico de radar: NES, RPS y CPI
st.markdown("---")
st.subheader("4️⃣ Alineación de dimensiones")
st.markdown("*Qué tan alineadas están las dimensiones: emocional, conversión y social.*")

# Crear gráfico de radar
theta = radar_factory(3, frame='polygon')

# Datos promedio para el radar chart
latest_date = filtered_df['Fecha'].max()
latest_data = filtered_df[filtered_df['Fecha'] == latest_date].iloc[0]

# Normalizar valores para el radar (escala 0-1)
data = [
    ['NES', 'RPS', 'CPI'],
    [
        [(latest_data['NES'] - 0) / 100], 
        [(latest_data['RPS'] - 0) / 100], 
        [(latest_data['CPI'] - 0) / 100]
    ]
]

fig4, ax4 = plt.subplots(figsize=(8, 6), subplot_kw=dict(projection='radar'))

# Colores y estilos
color = 'blue'
ax4.plot(theta, data[1][0], color=color)
ax4.fill(theta, data[1][0], facecolor=color, alpha=0.25)
ax4.set_varlabels(data[0])

# Añadir valores numéricos
for i, val in enumerate(data[1][0]):
    angle = i * 2 * np.pi / 3
    ax4.text(angle, val[0] + 0.1, f"{val[0]*100:.1f}%", 
             ha='center', va='center', size=12,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

# Título
ax4.set_title('Alineación de Dimensiones NES, RPS y CPI', fontsize=14, y=1.05)

# Mejoras visuales
fig4.tight_layout()

st.pyplot(fig4)

# Panel de métricas clave
st.markdown("---")
st.subheader("📌 Métricas clave")

# Calcular métricas
latest_nps = filtered_df.iloc[-1]['NPS-E']
avg_nps = filtered_df['NPS-E'].mean()
trend = latest_nps - filtered_df.iloc[-6]['NPS-E'] if len(filtered_df) > 5 else 0

# Mostrar métricas en columnas
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("NPS-E Actual", f"{latest_nps:.1f}", f"{trend:.1f}")
with col2:
    st.metric("Promedio NPS-E", f"{avg_nps:.1f}")
with col3:
    st.metric("Correlación NES-RPS", f"{corr:.2f}")

# Información adicional
st.markdown("---")
st.subheader("ℹ️ Información del NPS-E")
st.markdown("""
**NPS-E (Net Promoter Score Emocional)** es un indicador compuesto que mide la propensión del cliente a recomendar 
la marca basándose en tres dimensiones clave:

- **NES (Net Emotion Score)**: Mide el nivel de emoción positiva que genera la marca en los clientes.
- **RPS (Revenue Promoter Score)**: Indica el porcentaje de conversiones derivadas de recomendaciones.
- **CPI (Cultural Promoter Index)**: Evalúa la integración de la marca en el entorno cultural y social.

La fórmula para calcular el NPS-E es: `NPS-E = (0.4 * NES) + (0.4 * RPS) + (0.2 * CPI)`
""")

# Añadir opciones de descarga
st.markdown("---")
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Descargar datos filtrados como CSV",
    data=csv,
    file_name='nps_e_data_filtered.csv',
    mime='text/csv',
)