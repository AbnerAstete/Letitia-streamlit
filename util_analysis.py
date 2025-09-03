import os
import re
import zipfile
import json

from sklearn.impute import SimpleImputer

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import math


def show_csv_metadata_from_json_path(json_path_csv):
    """Read JSON with CSV metadata and display it in Streamlit."""
    with open(json_path_csv, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    for _, file_data in json_data.get("files_analysis", {}).items():
        filename = file_data.get('filename', 'File')
        st.markdown(f"### 👨‍💻 Data Facet - {filename}")

        st.markdown("""
            <div style="text-align: justify;">
            This is an example of an ideal Kaggle dataset, where the data meets high-quality standards: completeness with minimal missing values, unique observations without duplicates, and very few outliers. Such datasets can be used almost directly for the transformation step.
            In contrast, the next project presents a dataset that requires a structured cleaning process. By comparing both cases, it becomes clear how data quality directly impacts analysis,  and why cleaning steps such as handling missing values, removing duplicates, or treating outliers are often essential.
            <br><br>          
            Based on the statistical analysis from the pipeline, we observe that most students are between 15 and 18 years old, 
            with a fairly balanced gender distribution. Parental education and support levels vary, which may influence students’ study habits 
            and academic outcomes. Weekly study time shows a wide range, reflecting differences in dedication or workload, while absences also 
            vary considerably, potentially impacting GPA. The average GPA of 1.91 indicates a mix of high and low performers. Participation 
            in extracurricular activities such as sports, music, and volunteering is generally low, suggesting that students focus more on 
            academics than on additional activities.
            <br><br>          
            This is a basic analysis of what we can understand just by looking at the dataset statistics. As we said, there is no universal standard for data cleaning. There are modern approaches thattend to focus more on the context of the dataset. If you want to explore the dataset further and understand the data shown here in more detail, you can visit the dataset link provided.
            </div>
            """,
            unsafe_allow_html=True
        )

        # === 📋 Data Sample ===
        sample_data = file_data.get('sample_data', {})
        if sample_data:
            st.markdown(" ####  Data Sample")
            st.markdown(
                "Here a small **sample of the data** from the file, so you can get a quick look at some examples of what each column contains."
            )
            preview_dict = {col: vals.get('first_values', []) for col, vals in sample_data.items()}
            st.dataframe(pd.DataFrame(preview_dict), use_container_width=True)

        # === 📊 Schema ===
        schema = file_data.get('csv_schema', {})
        st.markdown(" #### Schema")
        st.markdown(
            "This section shows the **structure of the dataset**. You can see the total number of rows and columns, "
            "how much memory it uses, and the type of data stored in each column (for example, numbers or text)."
        )
        st.write(
            f"**Rows:** {schema.get('total_rows', 0):,} | "
            f"**Columns:** {schema.get('total_columns', 0)} | "
            f"**Memory (MB):** {schema.get('memory_usage_mb', 0)}"
        )

        if schema.get('columns') and schema.get('column_types'):
            df_schema = pd.DataFrame({
                "Column": schema['columns'],
                "Data Type": [schema['column_types'][c] for c in schema['columns']]
            })
            st.dataframe(df_schema, use_container_width=True)

        # === ✅ Data Quality ===
        
        dq = file_data.get('data_quality', {})
        st.markdown(" ####  Data Quality")
        st.markdown(
        "This section shows how complete and clean it is. "
        "It checks for missing data, duplicate rows, and gives an overall completeness score (100% means no missing data)."
        )
        st.write(
            f"**Total Null Values:** {dq.get('total_null_values', 0):,} | "
            f"**Duplicate Rows:** {dq.get('duplicate_rows', 0):,} | "
            f"**Completeness Score:** {dq.get('completeness_score', 0)}%"
        )
        if dq.get('null_values_per_column'):
            df_quality = pd.DataFrame({
                "Column": list(dq['null_values_per_column'].keys()),
                "Null Values": list(dq['null_values_per_column'].values()),
                "% Null": list(dq['null_percentage_per_column'].values()),
                "Unique Values": list(dq['unique_values_per_column'].values())
            })
            st.dataframe(df_quality, use_container_width=True)

        # === 📈 Numeric Statistics ===
        num_stats = file_data.get('numeric_statistics', {})
        if num_stats:
            st.markdown("#### Numeric Statistics")
            st.markdown(
                "This section shows **basic statistics for numeric columns** in the dataset. "
                "It includes measures like minimum, maximum, mean, and standard deviation, "
                "helping you understand the distribution and range of your numbers."
            )
            df_stats = pd.DataFrame(num_stats).T
            st.dataframe(df_stats, use_container_width=True)

        # === 🔠 Categorical Analysis ===
        cat_analysis = file_data.get('categorical_analysis', {})
        if cat_analysis:
            st.markdown(" ####  Categorical Analysis")
            for col, analysis in cat_analysis.items():
                st.markdown(f"**{col}** - {analysis.get('unique_count', 0)} unique values, most frequent: `{analysis.get('most_frequent', '')}`")
                top_values = analysis.get('top_5_values', {})
                if top_values:
                    st.bar_chart(pd.Series(top_values))


def plot_histograms_streamlit(file_data, group_size=4):
    file_path = file_data.get('file_path')
    filename = file_data.get('filename', 'File')

    st.markdown("The frequency and values of each column of the dataset are presented.")

    if not file_path or not Path(file_path).exists():
        st.warning(f"⚠️ Cannot open {filename} for plotting.")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error reading {filename}: {e}")
        return

    numeric_columns = df.select_dtypes(include=['number']).columns
    if len(numeric_columns) == 0:
        st.info("No numeric columns to plot.")
        return

    total_cols = min(group_size, len(numeric_columns))
    rows = math.ceil(len(numeric_columns) / total_cols)

    fig = make_subplots(
        rows=rows, cols=total_cols,
        subplot_titles=numeric_columns,
        horizontal_spacing=0.1, vertical_spacing=0.15
    )

    for i, col in enumerate(numeric_columns):
        col_data = df[col].dropna()
        unique_vals = col_data.nunique()

        # Dynamic bins
        if unique_vals <= 5:
            nbins = unique_vals
        elif unique_vals <= 30:
            nbins = 15
        else:
            q75, q25 = np.percentile(col_data, [75 ,25])
            iqr = q75 - q25
            bin_width = 2 * iqr * len(col_data) ** (-1/3)
            nbins = max(int(np.ceil((col_data.max() - col_data.min()) / bin_width)), 10)

        row = i // total_cols + 1
        col_pos = i % total_cols + 1

        fig.add_trace(
            go.Histogram(
                x=col_data,
                nbinsx=nbins,
                marker_color='skyblue',
                marker_line=dict(color='black', width=1),
                name=col
            ),
            row=row, col=col_pos
        )

        # # ✅ Set axis labels for each subplot
        # fig.update_xaxes(title_text="Value", row=row, col=col_pos)
        # fig.update_yaxes(title_text="Frequency", row=row, col=col_pos)

    fig.update_layout(
        showlegend=False,
        height=250 * rows,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)



def plot_boxplots_streamlit(file_data, group_size=4):
    file_path = file_data.get('file_path')
    filename = file_data.get('filename', 'File')

    if not file_path or not Path(file_path).exists():
        st.warning(f"⚠️ Cannot open {filename} for plotting.")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error reading {filename}: {e}")
        return

    numeric_columns = df.select_dtypes(include=['number']).columns
    if len(numeric_columns) == 0:
        st.info("No numeric columns to plot.")
        return

    total_cols = min(group_size, len(numeric_columns))
    rows = math.ceil(len(numeric_columns) / total_cols)

    fig = make_subplots(rows=rows, cols=total_cols,
                        subplot_titles=numeric_columns,
                        horizontal_spacing=0.1, vertical_spacing=0.15)

    for i, col in enumerate(numeric_columns):
        col_data = df[col].dropna()

        # Detect outliers
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
        n_outliers = len(outliers)

        row = i // total_cols + 1
        col_pos = i % total_cols + 1

        fig.add_trace(
            go.Box(
                y=col_data,
                boxpoints='outliers',
                marker_color='lightblue',
                line_color='deepskyblue',
                name=f"{col} ({n_outliers} outliers)"
            ),
            row=row, col=col_pos
        )

    fig.update_layout(
        showlegend=False,
        height=350 * rows,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

def taxonomy_queries_section_analysis_project():
    """
    Streamlit section to display taxonomy queries for the 
    Student Performance Analysis project with interactive interface
    """
    
    st.markdown("""
    This section presents the taxonomy queries validation phase, executed at the end of our pipeline 
    on the **Retail Data Analytics** project. These queries demonstrate how the RPCM (Research Process Conceptual Model) 
    enables comprehensive queries on different aspects of a research project, from organization to validation of results.
    
    The following examples show concrete query executions and their results, validating that our model 
    behaves correctly and performs as expected for this specific project. All queries were constructed 
    using the **Domain Specific Language (DSL)** that forms the foundational query interface supported by ATLAS.
    """)
    
    st.divider()
    
    # Define queries organized by categories - UPDATED WITH REAL JSON DATA
    queries_data = {
        "Project and Organizational Queries": {
            "List project with keywords": {
                "query": """from Project 
qualifiedName, keywords""",
                "result": """qualifiedName = "student-performance-analysis@StudentPerformanceAn"
keywords = []"""
            },
            "User responsible for creating a project": {
                "query": """from Project  select createdBy""",
                "result": """createdBy = "joelknapp" """
            }
        },
        
        "Process and Workflow Queries": {
            "Experiment stages": {
                "query": """from Experiment  select stages""",
                "result": """name = "stage@StudentPerformanceAn" """
            },
            "Actions with status and execution details": {
                "query": """from Action  where name = "Action - Student Performance Analysis"  
select qualifiedName, inputData, outputData, status""",
                "result": """qualifiedName = "notebook-student-performance-analysis.ipynb-v1@StudentPerformanceAn"
status = "Completed"
inputData = "Analysis of 1 datasets: Student_performance_data _.csv"
outputData = "Generated 16 outputs including models and visualizations"""
            }
        },
        
        "Data Lineage and Traceability Queries": {
            "Input datasets used in the project": {
                "query": """from UsedData  
where format = "csv"  
select name, size""",
                "result": """document = ["Student_performance_data _.csv"]
size = [166901]"""
            },
            "All visualization artifacts produced": {
                "query": """from UsedData where format = "png" """,
                "result": """- Chart: Figure 1 - KNeighborsClassifier - Correlation Among Features
- Chart: Figure 2 - KNeighborsClassifier - Selecting a Classification Model  
- Chart: Figure 3 - KNeighborsClassifier - Model Evaluation
- Chart: Figure 4 - SVC - Model Evaluation (again)
- Chart: Figure 5 - GradientBoostingClassifier - Reducing Dimensionality"""
            }
        },
        
        "Consensus and Validation Queries": {
            "Validation results for the project": {
                "query": """from Consensus  select typeConsensus, result""",
                "result": """typeConsensus = "Individual Review"
result = "approved" """
            },
            "Actions with approved consensus": {
                "query": """from Consensus
where result = "approved"
select action""",
                "result": """action.name = "Action - Student Performance Analysis" """
            }
        }
    }
    
    # Category selector
    st.subheader("📋 Select Query Category")
    selected_category = st.selectbox(
        "Choose a category:",
        list(queries_data.keys()),
        help="Select the query category you want to explore"
    )
    
    # Display queries from selected category
    if selected_category:
        st.subheader(f"🔎 {selected_category}")
        
        # Create tabs for each query in the category
        query_names = list(queries_data[selected_category].keys())
        tabs = st.tabs([f"Query {i+1}" for i in range(len(query_names))])
        
        for i, (query_name, query_info) in enumerate(queries_data[selected_category].items()):
            with tabs[i]:
                st.markdown(f"**{query_name}**")
                
                # Complete query at the top
                st.markdown("**🔍 Query:**")
                st.code(query_info["query"], language="sql")
                
                # Complete result at the bottom
                st.markdown("**📊 Result:**")
                st.code(query_info["result"], language="text")
                
                # Button to copy the query
                if st.button(f"📋 Copy Query", key=f"copy_{selected_category}_{i}"):
                    st.success("Query copied 🙂 ")
    
    # Additional information
    st.divider()
    st.info("""
    💡 **Note:** These queries demonstrate how the RPCM (Research Process Conceptual Model) 
    enables comprehensive queries on different aspects of a research project, 
    from organization to validation of results.
    """)


    