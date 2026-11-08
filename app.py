import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ==============================================================================
# CONFIGURACIÓN: Todo se guarda DENTRO de la carpeta Tarea08
# ==============================================================================
carpeta_trabajo = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Tarea08 - Rendimiento Académico", layout="wide")
st.title("🎓 Sistema Inteligente de Rendimiento Académico")
st.markdown("**SENATI | Ing. de Software con IA | Instructor: Inst. Jesús Ponce | Semana 12**")

# ==============================================================================
# GENERACIÓN DE DATOS (100% LOCAL, SIN INTERNET)
# ==============================================================================
@st.cache_data
def generar_datos():
    np.random.seed(42)
    n = 150
    horas = np.random.randint(1, 10, n)
    asistencia = np.random.randint(40, 100, n)
    calif = np.random.randint(5, 20, n)
    puntaje = (horas * 3) + (asistencia * 0.5) + (calif * 2)
    resultado = np.where(puntaje > 55, 'Aprobado', 'Desaprobado')
    return pd.DataFrame({
        'Horas_Estudio': horas,
        'Asistencia_Porcentaje': asistencia,
        'Calificacion_Previa': calif,
        'Resultado': resultado
    })

df = generar_datos()

# ==============================================================================
# MENÚ LATERAL
# ==============================================================================
menu = st.sidebar.selectbox("📂 Secciones", [
    "🏠 Inicio y Dataset",
    "📊 Parte A: Aprendizaje Supervisado",
    "🔍 Parte B: Aprendizaje No Supervisado",
    "📝 Entregables (Pág. 46)"
])

# ==============================================================================
# SECCIÓN 1: DATASET
# ==============================================================================
if menu == "🏠 Inicio y Dataset":
    st.header("1. Conjunto de Datos Generado")
    st.dataframe(df.head(10))
    
    # Guardar CSV DENTRO de Tarea08
    ruta_csv = os.path.join(carpeta_trabajo, 'dataset_academico.csv')
    df.to_csv(ruta_csv, index=False, encoding='utf-8')
    st.success(f"✅ Dataset guardado en: {ruta_csv}")
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar CSV", csv, 'dataset_academico.csv', 'text/csv')

# ==============================================================================
# SECCIÓN 2: APRENDIZAJE SUPERVISADO (Pág. 44)
# ==============================================================================
elif menu == "📊 Parte A: Aprendizaje Supervisado":
    st.header("Parte A: Predicción con Random Forest")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🔮 Predecir estudiante")
        h = st.slider("Horas de estudio", 1, 10, 5)
        a = st.slider("Asistencia (%)", 40, 100, 80)
        c = st.slider("Calificación previa", 5, 20, 14)
        
        if st.button("Predecir"):
            X = df[['Horas_Estudio', 'Asistencia_Porcentaje', 'Calificacion_Previa']]
            y = df['Resultado'].map({'Desaprobado': 0, 'Aprobado': 1})
            modelo = RandomForestClassifier(random_state=42).fit(X, y)
            pred = modelo.predict(pd.DataFrame({'Horas_Estudio':[h],'Asistencia_Porcentaje':[a],'Calificacion_Previa':[c]}))[0]
            st.markdown(f"### {'✅ APROBADO' if pred==1 else '❌ DESAPROBADO'}")

    with col2:
        st.subheader("📈 Evaluación del Modelo")
        X = df[['Horas_Estudio', 'Asistencia_Porcentaje', 'Calificacion_Previa']]
        y = df['Resultado'].map({'Desaprobado': 0, 'Aprobado': 1})
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        modelo = RandomForestClassifier(random_state=42).fit(X_tr, y_tr)
        preds = modelo.predict(X_te)
        acc = accuracy_score(y_te, preds)
        st.metric("Exactitud", f"{acc:.2%}")
        
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_te, preds), annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Desaprobado','Aprobado'], yticklabels=['Desaprobado','Aprobado'])
        ax.set_title('Matriz de Confusión'); ax.set_xlabel('Predicción'); ax.set_ylabel('Real')
        st.pyplot(fig)
        
        # Guardar imagen DENTRO de Tarea08
        ruta_img = os.path.join(carpeta_trabajo, 'matriz_confusion.png')
        fig.savefig(ruta_img, dpi=200, bbox_inches='tight')
        st.success(f"✅ Imagen guardada en: {ruta_img}")

# ==============================================================================
# SECCIÓN 3: APRENDIZAJE NO SUPERVISADO (Pág. 45)
# ==============================================================================
elif menu == "🔍 Parte B: Aprendizaje No Supervisado":
    st.header("Parte B: Agrupamiento K-Means")
    
    X_ns = df[['Horas_Estudio', 'Asistencia_Porcentaje', 'Calificacion_Previa']]
    X_norm = StandardScaler().fit_transform(X_ns)
    df['Grupo'] = KMeans(n_clusters=3, random_state=42).fit_predict(X_norm)
    X_pca = PCA(n_components=2).fit_transform(X_norm)
    df['PCA1'], df['PCA2'] = X_pca[:,0], X_pca[:,1]
    
    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Grupo', palette='viridis', s=80, ax=ax)
        ax.set_title('Clústeres de Estudiantes (PCA)')
        st.pyplot(fig)
        
        ruta_img = os.path.join(carpeta_trabajo, 'clusteres_kmeans.png')
        fig.savefig(ruta_img, dpi=200, bbox_inches='tight')
        st.success(f"✅ Imagen guardada en: {ruta_img}")
    
    with col2:
        st.subheader("📋 Interpretación")
        inter = df.groupby('Grupo')[['Horas_Estudio','Asistencia_Porcentaje','Calificacion_Previa']].mean().round(1)
        st.dataframe(inter)
        st.markdown("""
        - **Grupo 0:** Rendimiento medio  
        - **Grupo 1:** Rendimiento bajo  
        - **Grupo 2:** Rendimiento alto
        """)

# ==============================================================================
# SECCIÓN 4: ENTREGABLES (Pág. 46)
# ==============================================================================
elif menu == "📝 Entregables (Pág. 46)":
    st.header("Resumen de los 6 Entregables")
    
    # Generar reporte de texto DENTRO de Tarea08
    reporte = f"""
============================================================
TRABAJO GRUPAL - SISTEMA DE RENDIMIENTO ACADÉMICO
SENATI | Ing. de Software con IA | Inst. Jesús Ponce
============================================================

ENTREGABLE 1: CÓDIGO FUENTE PYTHON
- Archivo: app.py (este mismo archivo ejecutado con Streamlit)

ENTREGABLE 2: CONJUNTO DE DATOS
- Archivo: dataset_academico.csv (150 registros generados localmente)

ENTREGABLE 3: EVIDENCIA DE EJECUCIÓN
- Esta interfaz Streamlit es la evidencia visual.
- Tomar capturas de las secciones A y B para el informe.

ENTREGABLE 4: GRÁFICOS DE RESULTADOS
- matriz_confusion.png (Parte A)
- clusteres_kmeans.png (Parte B)

ENTREGABLE 5: INTERPRETACIÓN DE MODELOS
- Supervisado (Random Forest): Predice aprobación/desaprobación con alta exactitud.
- No Supervisado (K-Means): Identifica 3 perfiles de estudiantes sin usar etiquetas.

ENTREGABLE 6: CONCLUSIONES
1. Supervisado: Ideal para alertas tempranas automáticas.
2. No Supervisado: Ideal para descubrir patrones ocultos y segmentar estudiantes.
============================================================
"""
    ruta_reporte = os.path.join(carpeta_trabajo, 'reporte_entregables.txt')
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write(reporte)
    
    st.success(f"✅ Reporte guardado en: {ruta_reporte}")
    st.code(reporte)