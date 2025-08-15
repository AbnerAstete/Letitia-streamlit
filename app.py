import streamlit as st
import json
import os
from datetime import datetime
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

from util_analysis import comprehensive_dataset_evaluation, extract_csv_metadata_streamlit

os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]

api = KaggleApi()
api.authenticate()

st.title("Data Quality Analysis and Metadata Transformation")

st.markdown("""
This project implements a standardized pipeline to analyze the **Data quality** of Kaggle datasets and transform their metadata into entities compatible with the **Research Processes Curation Metamodel (RPCM)**.
            
Data quality is evaluated across five key dimensions: **accuracy**, **representativeness**, **completeness**, **relevance**, and **human/system reliability**. The analysis is based on the **Five Data Quality (DQ) Facets**: **Data**, **Source**, **System**, **Task**, and **Human**. With a primary focus on the **Data** and **Source** facets, which were the most relevant for this project.  

- **Source facet**: Review of traceability and quality of information from Kaggle, including dataset authors and descriptions, followed by metadata extraction from Kaggle projects and application of corresponding transformations.
- **Data facet**: Analysis of structure, data types, null values, duplicates, and schema consistency.  
""")

st.title("Dataset Reliability Evaluation - Kaggle")

dataset_name = st.text_input("📦 Introduce el nombre del dataset (owner/dataset):", placeholder="zynicide/wine-reviews")

if st.button("Evaluar dataset"):
    if dataset_name:
        try:
            # Crear carpeta de reportes
            reports_dir = "./dataset_reports"
            os.makedirs(reports_dir, exist_ok=True)

            # Evaluar
            full_evaluation = comprehensive_dataset_evaluation(dataset_name)
            dataset_info = full_evaluation["dataset_info"]
            assessment = full_evaluation["reliability_assessment"]

            # Información general compacta
            st.markdown(f"**{dataset_info.get('dataset_name','')}**  —  por **{dataset_info.get('author','')}**")
            st.markdown(f"[🌐 Ver en Kaggle]({dataset_info.get('kaggle_url')}) | Licencia: **{dataset_info.get('license','')}**")
            st.caption(f"Descargas: {dataset_info.get('total_downloads',0):,}  |  Votos: {dataset_info.get('votes',0)}")

            # Evaluación en tabla compacta
            rows = []
            for key, val in assessment.items():
                name = key.split("_", 1)[1].replace("_", " ").title()
                result = val["assessment"]
                emoji = "✅" if "✓" in result else ("⚠️" if "⚠" in result else "❌")
                rows.append({"Sección": name, "Estado": f"{emoji} {result}"})
            df_results = pd.DataFrame(rows)

            st.table(df_results)

            # Detalles opcionales en un expander
            with st.expander("📄 Ver detalles completos"):
                st.json(full_evaluation)

            # Guardar reporte
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_folder = os.path.join(reports_dir, dataset_name.replace("/", "_"))
            os.makedirs(dataset_folder, exist_ok=True)
            json_path = os.path.join(dataset_folder, f"reliability_report_{timestamp}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(full_evaluation, f, indent=2, ensure_ascii=False, default=str)

            st.success(f"📁 Reporte guardado en: {json_path}")

        except Exception as e:
            st.error(f"Error durante la evaluación: {e}")
    else:
        st.warning("Por favor, introduce un dataset válido.")


st.title("Analizador de CSV con Streamlit 📊")

st.title("📂 Analizador de CSV con Histogramas y Boxplots")
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
if uploaded_file is not None:
    extract_csv_metadata_streamlit(uploaded_file)