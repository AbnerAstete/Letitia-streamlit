import streamlit as st
from PIL import Image
import json
import os
from datetime import datetime
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

from util_analysis import show_csv_metadata_from_json_path,plot_boxplots_streamlit,plot_histograms_streamlit
from util_retail import show_csv_metadata_from_json_path_retail

# os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
# os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]

api = KaggleApi()
api.authenticate()

st.title("Data Pipeline - Kaggle Dataset Quality & RPCM Transformation")

st.markdown("""
This page explains the **end-to-end data flow** implemented in this dashboard, starting from
the extraction of a Kaggle project and ending with the generation of a **JSON file** ready for
integration into a metadata management system. Projects are selected based on popularity (votes), variety, relevance for information extraction, and implementation in Python.


The standardized pipeline includes:

1. **Data Quality Assessment:** Download the selected Kaggle dataset and evaluate its structure,
   completeness, and potential quality issues.
2. **Metadata Extraction:** Extract metadata from Kaggle and consolidate it into a single JSON file. This unified file serves as the basis for subsequent transformations into the Research Processes Curation Metamodel (RPCM).
3. **Transformation to RPCM Entities:** Explain how the consolidated JSON is transformed into RPCM  entities, and present the resulting structured output ready for integration into a metadata system.
""")

# --- Título de la sección de selección ---
st.markdown("### Select Project:")

st.markdown("""
The main difference between the presented projects is that the **"Student Performance Analysis"** project has a fully optimized dataset, so no data cleaning is required. In contrast, the other project contains a dataset that does not meet the quality criteria, so data cleaning will be performed in **Step 1**. 

Subsequently, **Step 2** and **Step 3** are essentially the same, as they are executed through the same pipeline. However, the generated data will still differ, since they belong to different projects.
""")

# --- Diccionario de datasets ---
datasets_dict = {
    "Student Performance Analysis": "rabieelkharoua/students-performance-dataset",
    "Retail Data Analytics": "manjeetsingh/retaildataset",
}
# --- Selectbox ---
selected_key = st.selectbox("", list(datasets_dict.keys()))
dataset_name = datasets_dict[selected_key]


st.markdown(" ## Step 1: Data Quality Assessment")

st.markdown("""

The dataset quality is analyzed across five key dimensions: **accuracy**, **representativeness**, **completeness**, **relevance**, and **human/system reliability**.  

The **Five Data Quality (DQ) Facets** were used: **Data**, **Source**, **System**, **Task**, and **Human**, with a focus on **Data** and **Source**, which are most relevant in this context.

- **Source facet:** The reliability of the data source on Kaggle was evaluated by reviewing the author, license, and traceability. This ensures the information comes from a trustworthy and legally reusable source.
- **Data facet:** The dataset was analyzed. Key aspects included examining statistical distributions, identifying missing values, detecting outliers.
""")

st.markdown("""
                    
        Currently, there is no universal standard for data cleaning, so in our approach we focus on three main dimensions: completeness, uniqueness, and outlier detection. 
        Users can adjust the weights assigned to these dimensions or even add new ones according to their own cleaning criteria, distributing the weights accordingly. 
        Likewise, the global quality threshold (default 75%) can be modified depending on the strictness required by the user.

        - **Completeness (45%)** is the most important factor, as missing data can severely affect analysis. The score is calculated as the proportion of non-missing values across all columns.

        - **Uniqueness (25%)** evaluates whether rows are duplicated, ensuring that each entry represents a distinct observation. Duplicates reduce the quality score proportionally.

        - **Outliers (30%)** focuses on numerical variables, penalizing datasets where extreme values deviate more than three standard deviations from the mean, since these may indicate anomalies or data errors.

        The global score is the weighted sum of these metrics, reflecting a balanced view of data quality The approach is flexible, users can modify or extend the calculation functions to adapt the quality metrics to new criteria or project specific needs.

        If this score falls below the defined threshold (by default 75%), the dataset is flagged as potentially problematic, requiring further cleaning or validation before use.
                
        """)

if selected_key == "Student Performance Analysis":

    json_path = "jsons/dataset_reliability_report.json"

    # --- Mostrar contenido del JSON directamente al seleccionar ---
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                full_evaluation = json.load(f)

            dataset_info = full_evaluation["dataset_info"]
            assessment = full_evaluation["reliability_assessment"]

            st.markdown(""" ### 🕵️‍♂️ Source Facet """)
            # Información general
            st.markdown(f"**{dataset_info.get('dataset_name','')}** — by **{dataset_info.get('author','')}**")
            st.markdown(f"[🌐 View on Kaggle]({dataset_info.get('kaggle_url')}) | License: **{dataset_info.get('license','')}**")
            st.caption(f"Downloads: {dataset_info.get('total_downloads',0):,}  |  Votes: {dataset_info.get('votes',0)}")

            # Evaluación en tabla compacta
            rows = []
            for key, val in assessment.items():
                name = key.split("_", 1)[1].replace("_", " ").title()
                result = val["assessment"]
                emoji = "✅" if "✓" in result else ("⚠️" if "⚠" in result else "❌")
                rows.append({"Section": name, "State": f"{emoji} {result[1:]}"})
            df_results = pd.DataFrame(rows)
            st.table(df_results)

            # Detalles completos en un expander
            with st.expander("📄 See full details"):
                st.json(full_evaluation)

        except Exception as e:
            st.error(f"Error cargando el JSON: {e}")
    else:
        st.warning("El archivo JSON no existe.")


    st.markdown("""
    The evaluation shows that the Students Performance Dataset has strong indicators of reliability, with a verified author, clear licensing (Attribution 4.0 International, CC BY 4.0), and high community engagement through votes (731), downloads (55466), and notebook usage (216 notebooks). While it lacks detailed descriptions and version history, the available metadata provides enough context for confident use. These aspects, aligned with the Source facet of data quality, ensure that the dataset can be effectively integrated and analyzed despite minor metadata gaps.
    """)

    json_path_csv = "jsons/datasets_analysis.json"
    show_csv_metadata_from_json_path(json_path_csv)



    csv_path = "datasets/Student_performance_data _.csv"
    file_data = {"file_path": csv_path, "filename": "Student Performance Data"}

    st.markdown(" ### 📊 Plots")

    tab1, tab2 = st.tabs(["Histograms", "Boxplots"])

    with tab1:
        # Aquí puedes incluir histogramas y boxplots
        plot_histograms_streamlit(file_data, group_size=4)

    with tab2:
        plot_boxplots_streamlit(file_data, group_size=4)

    scores = {
        "completeness": 100.0,
        "uniqueness": 100.0,
        "outliers": 100.0,
        "global_score": 100.0
    }

    st.subheader(" Data Quality Scores")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Completeness", f"{scores['completeness']:.1f}%")
    col2.metric("Uniqueness", f"{scores['uniqueness']:.1f}%")
    col3.metric("Outliers", f"{scores['outliers']:.1f}%")
    col4.metric("Global Score", f"{scores['global_score']:.1f}%")

    st.success("Dataset meets quality standards and is accepted for use. Now we can move on to the next step.")


    st.markdown(" ## Step 2: Metadata Extraction")

    st.markdown(
        """
        <div style="text-align: justify;">
        In this section, we describe the extraction process of metadata from the Kaggle platform using its official API. 
        Metadata and associated attributes are programmatically retrieved. We focus on <b>three main sources</b> 
        to build a complete and structured representation of Kaggle projects. Each extraction is saved in its respective JSON - Open standard file format and data interchange format that uses human-readable text to store and transmit data objects - file (<i>metadata.json</i>, <i>log_analysis.json</i>, <i>insights.json</i>) 
        for later unification into a single structured JSON prepared for the transformation.
        <br><br>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Cargar metadata desde el JSON
    json_metadata_path = "jsons/kernel-metadata.json"
    with open(json_metadata_path, "r") as f:
        metadata = json.load(f)
    # st.markdown(" ### Project Metadata ")
    # st.markdown(f"- **ID:** {metadata.get('id_no', 'N/A')}")
    # st.markdown(f"- **Title:** {metadata.get('title', 'N/A')}")
    # st.markdown(f"- **Code File:** {metadata.get('code_file', 'N/A')}")
    # st.markdown(f"- **Language:** {metadata.get('language', 'N/A').capitalize()}")
    # st.markdown(f"- **Dataset Sources:** {', '.join(metadata.get('dataset_sources', [])) if metadata.get('dataset_sources') else 'N/A'}")
    # with st.expander("Full Metadata JSON"):
    #     st.json(metadata)


    json_logs_path = "jsons/log_analysis.json"
    with open(json_logs_path, "r") as f:
        log_data = json.load(f)
    # st.markdown(" ### Log File Overview")
    # st.markdown(f"- **Filename:** {log_data['file_info'].get('filename', 'N/A')}")
    # st.markdown(f"- **Filepath:** {log_data['file_info'].get('filepath', 'N/A')}")
    # st.markdown(f"- **Created At:** {log_data['file_info'].get('created_at', 'N/A')}")
    # st.markdown(f"- **Models:** {', '.join([m['name'] for m in log_data.get('models', [])]) if log_data.get('models') else 'None'}")
    # st.markdown(f"- **Number of Lines:** {log_data['file_info'].get('num_lines', 'N/A')}")
    # st.markdown(f"- **Encoding:** {log_data['file_info'].get('encoding', 'N/A')}")
    # st.markdown(f"- **Total Bytes:** {log_data['file_info'].get('total_bytes', 'N/A')}")
    # with st.expander("Full Log Analysis JSON"):
    #     st.json(log_data)


    json_insights_path = "jsons/insights-notebook.json"
    with open(json_insights_path, "r") as f:
        insights_data = json.load(f)
    # st.markdown(" ### Notebook Insights Overview")
    # # Models
    # st.markdown("**Models Detected:**")
    # if insights_data.get("models"):
    #     for model in insights_data["models"]:
    #         st.markdown(f"- {model}")
    # else:
    #     st.markdown("- None")
    # # Graphs
    # st.markdown("**Graphs Identified:**")
    # if insights_data.get("graphs"):
    #     for graph in insights_data["graphs"]:
    #         st.markdown(f"- {graph['name']} | Section: {graph['section']} | Model: {graph['model']}")
    # else:
    #     st.markdown("- None")
    # with st.expander("Full Insights Notebook JSON"):
    #     st.json(insights_data)

    st.markdown(
        """
        <div style="text-align: justify;">
        <b>1. Project Metadata: 🔗 <a href="https://www.kaggle.com/code/joelknapp/student-performance-analysis/notebook" target="_blank">View Project</a> </b><br>
        Using the official Kaggle API, we obtain core information such as dataset names, descriptions, schemas, tags, and project attributes. 
        Additionally, we download the <i>kernel-metadata.json</i>, associated datasets and the complete project notebook in <i>.ipynb</i> format. 
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("📂 Metadata JSON (Click to expand)"):
        st.markdown(
            """
            Below is the structured metadata extracted from the Kaggle project. 
            You can explore the project details, datasets, and attributes in this JSON.
            """,
            unsafe_allow_html=True
        )
        st.json(metadata, expanded=False) 

    st.markdown(
        """
        <b>2. Execution Logs: 🔗 <a href="https://www.kaggle.com/code/joelknapp/student-performance-analysis/log" target="_blank">View Project's Log</a> </b><br>
        Our pipeline automatically downloads and processes the kernel’s <i>.log</i> file in JSON format, ensuring proper encoding detection. 
        From this source, we extract dataset structure (rows, columns, data types, memory usage), trained models (name, accuracy, confusion matrix, class-level metrics), 
        execution times, and metadata such as file size and creation date. 
        All extracted details are consolidated into a structured file (<i>log_analysis.json</i>) for integration.
        """,
        unsafe_allow_html=True
    )

    with st.expander("📝 Log Analysis JSON (Click to expand)"):
        st.markdown(
            """
            This JSON shows the contents found within the Kaggle Project's Log, if available. 
            It may include datasets, trained models, execution logs, and other relevant files depending on the project. Expand to inspect the structured information extracted from the Log.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.json(log_data, expanded=False)

    st.markdown(
        """
        <b>3. Notebook Content Analysis:</b><br>
        The <i>.ipynb</i> file is parsed in JSON format to scan all code and markdown cells. 
        From here, we identify datasets loaded (e.g., <i>pd.read_csv</i> calls), machine learning models referenced, evaluation metrics (accuracy, F1 score, confusion matrix, classification reports), 
        and visual outputs like plots or images. Notebook structure is also tracked by mapping extracted insights to markdown headers, ensuring contextualized interpretation.
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("📓 Insights Notebook JSON (Click to expand)"):
        st.markdown(
            """
            This JSON displays the contents found within the Kaggle notebook, if available. 
            It may include datasets loaded, machine learning models referenced, evaluation metrics, plots, and other code artifacts, depending on the project. 
            Expand to explore the structured insights extracted from the notebook.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.json(insights_data, expanded=False)


    st.markdown("### 🧩 Unified Kaggle Entities")

    st.markdown(
        """
        Once the pipeline generates the three files containing the extracted Kaggle data, we can proceed to unify them into a single JSON. This process follows the structure of our Kaggle metadata model, ensuring a consistent order and facilitating subsequent transformations or analysis.
        """
    )

    image_kaggle = Image.open("images/kaggle.png")
    st.image(image_kaggle, caption="Kaggle Metatada Model", use_column_width=True)

        
    st.markdown(
        """
        All this information is structured into a single dictionary with **eight main sections**:
        - **Project:** Kaggle project metadata (title, keywords).  
        - **Owner:** Notebook owner / user information.  
        - **Notebook:** Main execution file (.ipynb).  
        - **File:** Files used or generated during execution.  
        - **DataSets:** External semantic datasets linked to the project.  
        - **Log:** Execution logs captured from the kernel.  
        - **Model:** Trained machine learning models.  
        - **CodeLine:** Code artifacts such as graphs and models referenced in code.
        """
    )

    json_entities_kaggle_path = "jsons/entities-kaggle.json"
    with open(json_entities_kaggle_path, "r") as f:
        entities_kaggle_data = json.load(f)

    with st.expander("🧩 Entities Kaggle JSON (Click to expand)"):
        st.json(entities_kaggle_data)


    st.markdown(" ## Step 3: Transformation to RPCM Entities")


    st.markdown("""
    The idea is to translate raw Kaggle information into a standardized structure that can be integrated with broader workflows. This standardization ensures that every project has a clearly defined user, experiment, and execution process, while also documenting how data was used, produced, and approved. The goal is not just to store Kaggle information, but to map it into a structured context, making it traceable.
    """)


    image_rpcm = Image.open("images/Metamodelo-Atlas.png")
    st.image(image_rpcm, caption="Research Processes Curation Metamodel (RPCM)", use_column_width=True)

    st.markdown("""
    When metadata is taken from Kaggle (information about datasets, notebooks, models, and results), it arrives in a raw and fragmented form. To make this information useful for research management, it is transformed into structured entities that can be stored in the metadata system.

    Each entity corresponds to a specific part of the research workflow:

    - **User:** Who created the work.  
    - **Project:** The overall research effort.  
    - **Experiment, Stage, Iteration:** the steps showing how the work progresses.  
    - **Action:** The execution of a notebook or analysis.  
    - **Used Data:** The datasets, files, models, or graphs involved.  
    - **Consensus:** The approval or validation of the work.  

    These entities are connected through unique identifiers, allowing the system to track what was done, who did it, with which data, in which order, and what results were obtained. This also makes it possible to clearly identify which elements (datasets, models, graphs, or outputs) were generated directly by a specific project or notebook.

    The transformation ensures that Kaggle projects are:

    - **Clear in their outputs:** It is possible to see exactly which elements were produced by the project or notebook.  
    - **Reusable:** Information can be compared or integrated with other projects.  
    - **Standardized:** Projects can be represented consistently, even if their content differs.  

    Additionally, some fields can be enriched manually, allowing users to complement the automatically extracted information with more context or specific details.
    """)

    json_entities_RPCM_path = "jsons/entities-bulk-atlas.json"

    if os.path.exists(json_entities_RPCM_path):
        with open(json_entities_RPCM_path, "r", encoding="utf-8") as f:
            entities_RPCM_data = json.load(f)

        with st.expander("📂 Entities RPCM JSON"):
            st.json(entities_RPCM_data)
    else:
        st.warning(f"⚠️ File not found: `{json_entities_RPCM_path}`")

elif selected_key == "Retail Data Analytics":

    json_path_retail = "jsons-retail/dataset_reliability_report.json"

    # --- Mostrar contenido del JSON directamente al seleccionar ---
    if os.path.exists(json_path_retail):
        try:
            with open(json_path_retail, "r", encoding="utf-8") as f:
                full_evaluation = json.load(f)

            dataset_info = full_evaluation["dataset_info"]
            assessment = full_evaluation["reliability_assessment"]

            st.markdown(""" ### 🕵️‍♂️ Source Facet """)
            # Información general
            st.markdown(f"**{dataset_info.get('dataset_name','')}** — by **{dataset_info.get('author','')}**")
            st.markdown(f"[🌐 View on Kaggle]({dataset_info.get('kaggle_url')}) | License: **{dataset_info.get('license','')}**")
            st.caption(f"Downloads: {dataset_info.get('total_downloads',0):,}  |  Votes: {dataset_info.get('votes',0)}")

            # Evaluación en tabla compacta
            rows = []
            for key, val in assessment.items():
                name = key.split("_", 1)[1].replace("_", " ").title()
                result = val["assessment"]
                emoji = "✅" if "✓" in result else ("⚠️" if "⚠" in result else "❌")
                rows.append({"Section": name, "State": f"{emoji} {result[1:]}"})
            df_results = pd.DataFrame(rows)
            st.table(df_results)

            # Detalles completos en un expander
            with st.expander("📄 See full details"):
                st.json(full_evaluation)

        except Exception as e:
            st.error(f"Error cargando el JSON: {e}")
    else:
        st.warning("El archivo JSON no existe.")

    st.markdown("""
    The evaluation shows that the Retail Data Analytics dataset has strong indicators of reliability, with a verified author (Manjeet Singh), a clear license (CC0: Public Domain), and high community engagement through votes (1082), downloads (97875), and notebook usage (89 notebooks). While it lacks detailed descriptions, temporal metadata, and version history, the available information provides enough context for confident use. These aspects, aligned with the Source facet of data quality, ensure that the dataset can be effectively integrated and analyzed despite minor metadata gaps.
    """)


    datasets = {
    "Features Dataset": {
    "file_path": "datasets-retail/Features data set.csv",
    "json_path": "jsons-retail/datasets_analysis.json",
    "scores": {
        "completeness": 75.5,
        "uniqueness": 100.0,
        "outliers": 30.0,
        "global_score": 67.9
        }
    },
    "Stores Dataset": {
        "file_path": "datasets-retail/stores data-set.csv",
        "json_path": "jsons-retail/datasets_analysis.json",
        "scores": {
            "completeness": 100.0,
            "uniqueness": 100.0,
            "outliers": 100.0,
            "global_score": 100.0
        }
    },
    "Sales Dataset": {
        "file_path": "datasets-retail/sales data-set.csv",
        "json_path": "jsons-retail/datasets_analysis.json",
        "scores": {
            "completeness": 100.0,
            "uniqueness": 100.0,
            "outliers": 66.0,
            "global_score": 90.
        }
    }
    }

    # Selección del dataset
    selected_dataset_name = st.selectbox(" ### Select a dataset to analyze", list(datasets.keys()))
    dataset = datasets[selected_dataset_name]
    file_data = {"file_path": dataset["file_path"], "filename": selected_dataset_name}

    # Mostrar metadata solo del dataset seleccionado
    show_csv_metadata_from_json_path_retail(dataset["json_path"], selected_dataset_name)

    # Plots solo del dataset seleccionado
    st.markdown("### 📊 Plots")
    tab1, tab2 = st.tabs(["Histograms", "Boxplots"])
    with tab1:
        plot_histograms_streamlit(file_data, group_size=4)
    with tab2:
        plot_boxplots_streamlit(file_data, group_size=4)

    # Mostrar scores solo del dataset seleccionado
    scores = dataset["scores"]
    st.subheader("Data Quality Scores")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Completeness", f"{scores['completeness']:.1f}%")
    col2.metric("Uniqueness", f"{scores['uniqueness']:.1f}%")
    col3.metric("Outliers", f"{scores['outliers']:.1f}%")
    col4.metric("Global Score", f"{scores['global_score']:.1f}%")

    if selected_dataset_name != "Features Dataset":
        st.success(f"{selected_dataset_name} meets quality standards and is accepted for use.")
    else: 
        st.error(f"{selected_dataset_name} does not meet quality standards and cannot be used. The dataset needs to be cleaned first.")
        show_csv_metadata_from_json_path_retail("jsons-retail/datasets_analysis_clean.json","clean_Features data set.csv")

        file_path = "datasets-retail/clean_Features data set.csv"
        file_data = {"file_path": file_path, "filename": "Clean Features Dataset"}

        # Plots
        st.markdown("### 📊 Plots")
        tab1, tab2 = st.tabs(["Histograms", "Boxplots"])
        with tab1:
            #plot_histograms_streamlit(file_data, group_size=4)
            st.markdown("A")
        with tab2:
            plot_boxplots_streamlit(file_data, group_size=4)

        # Scores del dataset limpio
        scores = {
            "completeness": 100.0,
            "uniqueness": 100.0,
            "outliers": 70.0,
            "global_score": 91.0
        }

        st.subheader("Data Quality Scores")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Completeness", f"{scores['completeness']:.1f}%")
        col2.metric("Uniqueness", f"{scores['uniqueness']:.1f}%")
        col3.metric("Outliers", f"{scores['outliers']:.1f}%")
        col4.metric("Global Score", f"{scores['global_score']:.1f}%")

        st.success("Clean Features Dataset meets quality standards and is ready for use.")
    
    st.markdown(" ## Step 2: Metadata Extraction")

    st.markdown(
        """
        <div style="text-align: justify;">
        In this section, we describe the extraction process of metadata from the Kaggle platform using its official API. 
        Metadata and associated attributes are programmatically retrieved. We focus on <b>three main sources</b> 
        to build a complete and structured representation of Kaggle projects. Each extraction is saved in its respective JSON - Open standard file format and data interchange format that uses human-readable text to store and transmit data objects - file (<i>metadata.json</i>, <i>log_analysis.json</i>, <i>insights.json</i>) 
        for later unification into a single structured JSON prepared for the transformation.
        <br><br>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Cargar metadata desde el JSON
    json_retil_metadata_path = "jsons-retail/kernel-metadata.json"
    with open(json_retil_metadata_path, "r") as f:
        metadata = json.load(f)
    # st.markdown(" ### Project Metadata ")
    # st.markdown(f"- **ID:** {metadata.get('id_no', 'N/A')}")
    # st.markdown(f"- **Title:** {metadata.get('title', 'N/A')}")
    # st.markdown(f"- **Code File:** {metadata.get('code_file', 'N/A')}")
    # st.markdown(f"- **Language:** {metadata.get('language', 'N/A').capitalize()}")
    # st.markdown(f"- **Dataset Sources:** {', '.join(metadata.get('dataset_sources', [])) if metadata.get('dataset_sources') else 'N/A'}")
    # with st.expander("Full Metadata JSON"):
    #     st.json(metadata)


    json_retail_logs_path = "jsons-retail/log_analysis.json"
    with open(json_retail_logs_path, "r") as f:
        log_data = json.load(f)
    # st.markdown(" ### Log File Overview")
    # st.markdown(f"- **Filename:** {log_data['file_info'].get('filename', 'N/A')}")
    # st.markdown(f"- **Filepath:** {log_data['file_info'].get('filepath', 'N/A')}")
    # st.markdown(f"- **Created At:** {log_data['file_info'].get('created_at', 'N/A')}")
    # st.markdown(f"- **Models:** {', '.join([m['name'] for m in log_data.get('models', [])]) if log_data.get('models') else 'None'}")
    # st.markdown(f"- **Number of Lines:** {log_data['file_info'].get('num_lines', 'N/A')}")
    # st.markdown(f"- **Encoding:** {log_data['file_info'].get('encoding', 'N/A')}")
    # st.markdown(f"- **Total Bytes:** {log_data['file_info'].get('total_bytes', 'N/A')}")
    # with st.expander("Full Log Analysis JSON"):
    #     st.json(log_data)


    json_retail_insights_path = "jsons-retail/insights-notebook.json"
    with open(json_retail_insights_path, "r") as f:
        insights_data = json.load(f)
    # st.markdown(" ### Notebook Insights Overview")
    # # Models
    # st.markdown("**Models Detected:**")
    # if insights_data.get("models"):
    #     for model in insights_data["models"]:
    #         st.markdown(f"- {model}")
    # else:
    #     st.markdown("- None")
    # # Graphs
    # st.markdown("**Graphs Identified:**")
    # if insights_data.get("graphs"):
    #     for graph in insights_data["graphs"]:
    #         st.markdown(f"- {graph['name']} | Section: {graph['section']} | Model: {graph['model']}")
    # else:
    #     st.markdown("- None")
    # with st.expander("Full Insights Notebook JSON"):
    #     st.json(insights_data)

    st.markdown(
        """
        <div style="text-align: justify;">
        <b>1. Project Metadata: 🔗 <a href="https://www.kaggle.com/code/aremoto/retail-sales-forecast" target="_blank">View Project</a> </b><br>
        Using the official Kaggle API, we obtain core information such as dataset names, descriptions, schemas, tags, and project attributes. 
        Additionally, we download the <i>kernel-metadata.json</i>, associated datasets and the complete project notebook in <i>.ipynb</i> format. 
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("📂 Metadata JSON (Click to expand)"):
        st.markdown(
            """
            Below is the structured metadata extracted from the Kaggle project. 
            You can explore the project details, datasets, and attributes in this JSON.
            """,
            unsafe_allow_html=True
        )
        st.json(metadata, expanded=False) 

    st.markdown(
        """
        <b>2. Execution Logs: 🔗 <a href="https://www.kaggle.com/code/aremoto/retail-sales-forecast/log" target="_blank">View Project's Log</a> </b><br>
        Our pipeline automatically downloads and processes the kernel’s <i>.log</i> file in JSON format, ensuring proper encoding detection. 
        From this source, we extract dataset structure (rows, columns, data types, memory usage), trained models (name, accuracy, confusion matrix, class-level metrics), 
        execution times, and metadata such as file size and creation date. 
        All extracted details are consolidated into a structured file (<i>log_analysis.json</i>) for integration.
        """,
        unsafe_allow_html=True
    )

    with st.expander("📝 Log Analysis JSON (Click to expand)"):
        st.markdown(
            """
            This JSON shows the contents found within the Kaggle Project's Log, if available. 
            It may include datasets, trained models, execution logs, and other relevant files depending on the project. Expand to inspect the structured information extracted from the Log.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.json(log_data, expanded=False)

    st.markdown(
        """
        <b>3. Notebook Content Analysis:</b><br>
        The <i>.ipynb</i> file is parsed in JSON format to scan all code and markdown cells. 
        From here, we identify datasets loaded (e.g., <i>pd.read_csv</i> calls), machine learning models referenced, evaluation metrics (accuracy, F1 score, confusion matrix, classification reports), 
        and visual outputs like plots or images. Notebook structure is also tracked by mapping extracted insights to markdown headers, ensuring contextualized interpretation.
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("📓 Insights Notebook JSON (Click to expand)"):
        st.markdown(
            """
            This JSON displays the contents found within the Kaggle notebook, if available. 
            It may include datasets loaded, machine learning models referenced, evaluation metrics, plots, and other code artifacts, depending on the project. 
            Expand to explore the structured insights extracted from the notebook.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.json(insights_data, expanded=False)


    st.markdown("### 🧩 Unified Kaggle Entities")

    st.markdown(
        """
        Once the pipeline generates the three files containing the extracted Kaggle data, we can proceed to unify them into a single JSON. This process follows the structure of our Kaggle metadata model, ensuring a consistent order and facilitating subsequent transformations or analysis.
        """
    )

    image_kaggle = Image.open("images/kaggle.png")
    st.image(image_kaggle, caption="Kaggle Metatada Model", use_column_width=True)

        
    st.markdown(
        """
        All this information is structured into a single dictionary with **eight main sections**:
        - **Project:** Kaggle project metadata (title, keywords).  
        - **Owner:** Notebook owner / user information.  
        - **Notebook:** Main execution file (.ipynb).  
        - **File:** Files used or generated during execution.  
        - **DataSets:** External semantic datasets linked to the project.  
        - **Log:** Execution logs captured from the kernel.  
        - **Model:** Trained machine learning models.  
        - **CodeLine:** Code artifacts such as graphs and models referenced in code.
        """
    )

    json_entities_kaggle_retail_path = "jsons-retail/entities-kaggle.json"
    with open(json_entities_kaggle_retail_path, "r") as f:
        entities_kaggle_data = json.load(f)

    with st.expander("🧩 Entities Kaggle JSON (Click to expand)"):
        st.json(entities_kaggle_data)


    st.markdown(" ## Step 3: Transformation to RPCM Entities")


    st.markdown("""
    The idea is to translate raw Kaggle information into a standardized structure that can be integrated with broader workflows. This standardization ensures that every project has a clearly defined user, experiment, and execution process, while also documenting how data was used, produced, and approved. The goal is not just to store Kaggle information, but to map it into a structured context, making it traceable.
    """)


    image_rpcm = Image.open("images/Metamodelo-Atlas.png")
    st.image(image_rpcm, caption="Research Processes Curation Metamodel (RPCM)", use_column_width=True)

    st.markdown("""
    When metadata is taken from Kaggle (information about datasets, notebooks, models, and results), it arrives in a raw and fragmented form. To make this information useful for research management, it is transformed into structured entities that can be stored in the metadata system.

    Each entity corresponds to a specific part of the research workflow:

    - **User:** Who created the work.  
    - **Project:** The overall research effort.  
    - **Experiment, Stage, Iteration:** the steps showing how the work progresses.  
    - **Action:** The execution of a notebook or analysis.  
    - **Used Data:** The datasets, files, models, or graphs involved.  
    - **Consensus:** The approval or validation of the work.  

    These entities are connected through unique identifiers, allowing the system to track what was done, who did it, with which data, in which order, and what results were obtained. This also makes it possible to clearly identify which elements (datasets, models, graphs, or outputs) were generated directly by a specific project or notebook.

    The transformation ensures that Kaggle projects are:

    - **Clear in their outputs:** It is possible to see exactly which elements were produced by the project or notebook.  
    - **Reusable:** Information can be compared or integrated with other projects.  
    - **Standardized:** Projects can be represented consistently, even if their content differs.  

    Additionally, some fields can be enriched manually, allowing users to complement the automatically extracted information with more context or specific details.
    """)

    json_entities_RPCM_path = "jsons-retail/entities-bulk-atlas.json"

    if os.path.exists(json_entities_RPCM_path):
        with open(json_entities_RPCM_path, "r", encoding="utf-8") as f:
            entities_RPCM_data = json.load(f)

        with st.expander("📂 Entities RPCM JSON"):
            st.json(entities_RPCM_data)
    else:
        st.warning(f"⚠️ File not found: `{json_entities_RPCM_path}`")