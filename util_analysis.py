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

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()




# Get all reliability information
def comprehensive_dataset_evaluation(dataset_name):
    owner_slug, dataset_slug = dataset_name.split('/')
    ref = f"{owner_slug}/{dataset_slug}"

    # 1. Basic dataset information (you already have this)
    results = api.dataset_list(search=dataset_slug, user=owner_slug)
    dataset_info = None

    for ds in results:
        if ds.ref == ref:
            dataset_info = {
                "dataset_name": ds.title,
                "subtitle": getattr(ds, "subtitle", None),
                "description": getattr(ds, "description", None),
                "license": getattr(ds, "license_name", None),
                "author": ds.creator_name,
                "kaggle_id": ref,
                "kaggle_url": f"https://www.kaggle.com/datasets/{ref}",
                "total_downloads": ds.download_count,
                "votes": getattr(ds, 'vote_count', 0),
                "is_private": ds.is_private,
                "is_featured": ds.is_featured,
                "usability_rating": getattr(ds, "usabilityRating", None),
                "download_date": datetime.now().date().isoformat(),
                "creation_date": getattr(ds, "creationDate", "Not available"),
                "last_updated": getattr(ds, "lastUpdated", "Not available")
            }
            break

    if not dataset_info:
        raise ValueError(f"Dataset {ref} not found")

    # 2. Author information (reputation and activity)
    def get_all_datasets(user):
        page = 1
        all_datasets = []
        while True:
            datasets = api.dataset_list(user=user, page=page)
            if not datasets:
                break
            all_datasets.extend(datasets)
            page += 1
        return all_datasets

    def get_all_kernels(user):
        page = 1
        all_kernels = []
        while True:
            kernels = api.kernels_list(user=user, page=page)
            if not kernels:
                break
            all_kernels.extend(kernels)
            page += 1
        return all_kernels

    def get_user_followers(username):
        """
        Try to get follower information from user profile
        Note: Kaggle API has limited public access to follower data
        """
        try:
            # Try to get basic user info first
            user_info = api.user_read(username)
            followers = getattr(user_info, "followerCount", "Not available")
            following = getattr(user_info, "followingCount", "Not available")
            tier = getattr(user_info, "tier", "Not available")
            return followers, following, tier
        except:
            return "Not available", "Not available", "Not available"

    try:
        # Get datasets and notebooks using working methods
        author_datasets = get_all_datasets(owner_slug)
        total_datasets = len(author_datasets)

        author_notebooks = get_all_kernels(owner_slug)
        total_notebooks = len(author_notebooks)

        # Try to get follower information
        followers, following, tier = get_user_followers(owner_slug)

        # Calculate additional metrics from datasets
        total_downloads = sum(getattr(ds, 'download_count', 0) for ds in author_datasets)
        total_votes = sum(getattr(ds, 'vote_count', 0) for ds in author_datasets)

        # Calculate metrics from notebooks
        notebook_votes = sum(getattr(nb, 'voteCount', 0) for nb in author_notebooks)

        author_stats = {
            "total_datasets": total_datasets,
            "total_notebooks": total_notebooks,
            "total_dataset_downloads": total_downloads,
            "total_dataset_votes": total_votes,
            "total_notebook_votes": notebook_votes,
            "follower_count": followers,
            "following_count": following,
            "author_tier": tier,
            "avg_downloads_per_dataset": round(total_downloads / total_datasets, 2) if total_datasets > 0 else 0,
            "avg_votes_per_dataset": round(total_votes / total_datasets, 2) if total_datasets > 0 else 0
        }

    except Exception as e:
        print(f"Could not retrieve author statistics: {e}")
        author_stats = {"error": "Author information not available"}

    # 3. Notebooks using this dataset
    try:
        # Get notebooks
        notebooks = api.kernels_list(dataset=ref, page_size=100)

        # Try to get more pages quietly
        all_notebooks = notebooks.copy()
        for page in range(2, 6):  # Try pages 2-5
            try:
                more_notebooks = api.kernels_list(dataset=ref, page=page, page_size=100)
                if not more_notebooks:
                    break
                all_notebooks.extend(more_notebooks)
            except:
                break

        # Remove duplicates
        unique_notebooks = []
        seen_refs = set()
        for nb in all_notebooks:
            if nb.ref not in seen_refs:
                unique_notebooks.append(nb)
                seen_refs.add(nb.ref)

        # Sort by votes
        sorted_notebooks = sorted(unique_notebooks, key=lambda x: getattr(x, 'voteCount', 0), reverse=True)

        notebook_stats = {
            "total_notebooks_using_dataset": len(unique_notebooks),
            "popular_notebooks": []
        }

        # Store top notebooks info
        for nb in sorted_notebooks[:10]:
            notebook_stats["popular_notebooks"].append({
                "title": nb.title,
                "author": nb.author,
                "votes": getattr(nb, 'voteCount', 0),
                "url": f"https://www.kaggle.com/code/{nb.ref}",
                "medal": getattr(nb, 'medal', None),
                "language": getattr(nb, 'language', 'Unknown'),
                "ref": nb.ref
            })

    except Exception as e:
        print(f"Could not retrieve notebooks: {e}")
        notebook_stats = {"error": "Notebook information not available"}

    # 4. Dataset versions (traceability)
    try:
        versions = api.dataset_list_versions(ref)
        version_info = {
            "total_versions": len(versions),
            "current_version": versions[0].versionNumber if versions else 1,
            "version_history": []
        }

        for version in versions[:5]:  # Last 5 versions
            version_info["version_history"].append({
                "version": version.versionNumber,
                "creation_date": getattr(version, "creationDate", "Not available"),
                "status": getattr(version, "status", "Not available")
            })
    except Exception as e:
        print(f"Could not retrieve version information: {e}")
        version_info = {"error": "Version information not available"}

    # 5. Compile reliability assessment
    reliability_assessment = {
        "1_author_info": {
            "author": dataset_info["author"],
            "statistics": author_stats,
            "assessment": "✓ Available" if "error" not in author_stats else "✗ Not available"
        },

        "2_publication_date": {
            "creation_date": dataset_info["creation_date"],
            "last_updated": dataset_info["last_updated"],
            "assessment": "✓ Temporal information available"
        },

        "3_license": {
            "license": dataset_info["license"],
            "assessment": "⚠ Unknown license" if dataset_info["license"] == "Unknown" else f"✓ License: {dataset_info['license']}"
        },

        "4_external_source": {
            "description": dataset_info["description"],
            "assessment": "⚠ No detailed description" if not dataset_info["description"] else "✓ Description available"
        },

        "5_notebook_usage": {
            "total_notebooks": notebook_stats.get("total_notebooks_using_dataset", 0),
            "popular_notebooks": notebook_stats.get("popular_notebooks", []),
            "assessment": f"✓ Used by {notebook_stats.get('total_notebooks_using_dataset', 0)} notebooks"
        },

        "6_traceability": {
            "versions": version_info,
            "assessment": f"✓ {version_info.get('total_versions', 0)} versions available" if "error" not in version_info else "⚠ No version information"
        },

        "7_description": {
            "title": dataset_info["dataset_name"],
            "subtitle": dataset_info["subtitle"],
            "description": dataset_info["description"],
            "assessment": "✓ Clear title and subtitle" if dataset_info["subtitle"] else "⚠ Limited description"
        },

        "8_community_feedback": {
            "votes": dataset_info.get("votes", 0),
            "downloads": dataset_info.get("total_downloads", 0),
            "featured": dataset_info.get("is_featured", False),
            "assessment": f"✓ {dataset_info.get('votes', 0)} votes, {dataset_info.get('total_downloads', 0)} downloads"
        }
    }

    return {
        "dataset_info": dataset_info,
        "reliability_assessment": reliability_assessment
    }

# Function to display reliability report
def print_reliability_report(evaluation):
    print("=" * 80)
    print("DATASET RELIABILITY ASSESSMENT REPORT")
    print("=" * 80)

    assessment = evaluation["reliability_assessment"]

    for key, value in assessment.items():
        element = key.split("_", 1)[1].replace("_", " ").title()
        print(f"\n{key.split('_')[0]}. {element}")
        print("-" * 50)
        print(f"Assessment: {value['assessment']}")

        # Show specific details according to the element
        if "author" in key:
            print(f"• Author name: {value['author']}")
            if "error" not in value["statistics"]:
                stats = value["statistics"]
                print(f"• Tier: {stats.get('author_tier', 'N/A')}")
                print(f"• Published datasets: {stats.get('total_datasets', 0)}")
                print(f"• Published notebooks: {stats.get('total_notebooks', 0)}")
                print(f"• Total dataset downloads: {stats.get('total_dataset_downloads', 0):,}")
                print(f"• Total dataset votes: {stats.get('total_dataset_votes', 0):,}")
                print(f"• Total notebook votes: {stats.get('total_notebook_votes', 0):,}")
                print(f"• Followers: {stats.get('follower_count', 'N/A')}")
                print(f"• Following: {stats.get('following_count', 'N/A')}")
                print(f"• Avg downloads per dataset: {stats.get('avg_downloads_per_dataset', 0)}")
                print(f"• Avg votes per dataset: {stats.get('avg_votes_per_dataset', 0)}")

        elif "publication" in key:
            print(f"• Creation date: {value['creation_date']}")
            print(f"• Last updated: {value['last_updated']}")
            print(f"• Downloaded on: {evaluation['dataset_info']['download_date']}")

        elif "notebook" in key:
            print(f"• Total notebooks using it: {value['total_notebooks']}")
            if value['popular_notebooks']:
                print("• Most popular notebooks:")
                for nb in value['popular_notebooks'][:3]:
                    print(f"  - {nb['title']} ({nb['votes']} votes)")

        elif "feedback" in key:
            print(f"• Votes: {value['votes']}")
            print(f"• Downloads: {value['downloads']:,}")
            print(f"• Featured: {'Yes' if value['featured'] else 'No'}")

        elif "description" in key:
            print(f"• Title: {value['title']}")
            if value['subtitle']:
                print(f"• Subtitle: {value['subtitle']}")
            if value['description']:
                print(f"• Description: {value['description'][:200]}{'...' if len(value['description']) > 200 else ''}")
            else:
                print("• Description: Not provided")

# Execute complete evaluation
try:
    full_evaluation = comprehensive_dataset_evaluation(dataset_name)
    print_reliability_report(full_evaluation)

    # Save complete report
    with open('./dataset_reliability_report.json', 'w', encoding='utf-8') as f:
        json.dump(full_evaluation, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n\nComplete report saved to: ./dataset_reliability_report.json")

except Exception as e:
    print(f"Error during evaluation: {e}")

    





# ---------------------
# HISTOGRAMAS con Plotly
# ---------------------
def plot_histograms_streamlit(file_data, group_size=4):
    file_path = file_data.get('file_path')
    filename = file_data.get('filename', 'File')

    if not file_path or not Path(file_path).exists():
        st.warning(f"⚠️ No se puede abrir {filename} para graficar.")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error leyendo {filename}: {e}")
        return

    numeric_columns = df.select_dtypes(include=['number']).columns
    if len(numeric_columns) == 0:
        st.info("No hay columnas numéricas para graficar.")
        return

    st.subheader(f"📊 Histogramas - {filename}")

    total_cols = min(group_size, len(numeric_columns))
    rows = math.ceil(len(numeric_columns) / total_cols)

    fig = make_subplots(rows=rows, cols=total_cols,
                        subplot_titles=numeric_columns,
                        horizontal_spacing=0.1, vertical_spacing=0.15)

    for i, col in enumerate(numeric_columns):
        col_data = df[col].dropna()
        unique_vals = col_data.nunique()

        # Ajuste dinámico de bins
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

    fig.update_layout(
        showlegend=False,
        height=300 * rows,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------
# BOXPLOTS con Plotly
# ---------------------
def plot_boxplots_streamlit(file_data, group_size=4):
    file_path = file_data.get('file_path')
    filename = file_data.get('filename', 'File')

    if not file_path or not Path(file_path).exists():
        st.warning(f"⚠️ No se puede abrir {filename} para graficar.")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error leyendo {filename}: {e}")
        return

    numeric_columns = df.select_dtypes(include=['number']).columns
    if len(numeric_columns) == 0:
        st.info("No hay columnas numéricas para graficar.")
        return

    st.subheader(f"📦 Boxplots - {filename}")

    total_cols = min(group_size, len(numeric_columns))
    rows = math.ceil(len(numeric_columns) / total_cols)

    fig = make_subplots(rows=rows, cols=total_cols,
                        subplot_titles=numeric_columns,
                        horizontal_spacing=0.1, vertical_spacing=0.15)

    for i, col in enumerate(numeric_columns):
        col_data = df[col].dropna()
        
        # Identificar outliers
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
                boxpoints='outliers',  # muestra solo outliers
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


def extract_csv_metadata_streamlit(uploaded_file):
    # Configuración
    WEIGHTS = {
        "completeness": 0.45,   
        "uniqueness": 0.25,   
        "outliers": 0.30   
    }
    ALERT_THRESHOLD = 70
    
    def calculate_completeness(null_percentage_per_column):
        """
        Calcula el porcentaje de datos completos (no nulos) en el dataset.
        """
        if not null_percentage_per_column:
            return 100.0
        return 100 - np.mean(list(null_percentage_per_column.values()))


    def calculate_uniqueness(total_rows, duplicate_rows):
        """
        Calcula el porcentaje de unicidad de filas, penalizando duplicados.
        """
        if total_rows == 0:
            return 0.0
        return max(0.0, 100 - (duplicate_rows / total_rows * 100))


    def calculate_outliers(data, severity=3.0):
        """
        Calcula el porcentaje de outliers en columnas numéricas.
        Penaliza según la proporción de outliers y el parámetro 'severity'.
        """
        penalties = []
        numeric_stats = data.get("numeric_statistics", {})
        
        # Si tienes un DataFrame dentro de data
        df = data.get("df", None)
        if df is None:
            raise ValueError("Se requiere 'df' dentro de data para calcular outliers con esta lógica")
        
        for col in numeric_stats.keys():
            col_data = df[col].dropna()
            if len(col_data) == 0:
                penalties.append(0)
                continue
            
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            
            if iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                outlier_percentage = (len(outliers) / len(col_data)) * 100
                penalties.append(min(100, outlier_percentage * severity))
            else:
                penalties.append(0)
        
        avg_outlier_percentage = np.mean(penalties) if penalties else 0
        return max(0, 100 - avg_outlier_percentage)


    def calculate_global_score(dataset):
        """
        Calcula el puntaje global combinando completitud, unicidad y outliers.
        """
        completeness = calculate_completeness(dataset["data_quality"]["null_percentage_per_column"])
        uniqueness = calculate_uniqueness(dataset["csv_schema"]["total_rows"], dataset["data_quality"]["duplicate_rows"])
        outliers = calculate_outliers(dataset)

        global_score = (
            completeness * WEIGHTS["completeness"] +
            uniqueness * WEIGHTS["uniqueness"] +
            outliers * WEIGHTS["outliers"]
        )

        return {
            "completeness": completeness,
            "uniqueness": uniqueness,
            "outliers": outliers,
            "global_score": global_score
        }
    
    def clean_dataset(df, lower_quantile=0.05, upper_quantile=0.95, max_missing_frac=0.1):
        # Hacer copia para no modificar el original
        df_clean = df.copy()
        
        # Select numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        # Impute missing numeric values with median
        if len(numeric_cols) > 0:
            imputer = SimpleImputer(strategy='median')
            df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
        
        # Handle missing categorical values
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            missing_frac = df_clean[col].isna().mean()
            if 0 < missing_frac <= max_missing_frac:
                df_clean = df_clean[df_clean[col].notna()]
        
        # Drop duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Clip numeric outliers
        for col in numeric_cols:
            lower = df_clean[col].quantile(lower_quantile)
            upper = df_clean[col].quantile(upper_quantile)
            df_clean[col] = np.clip(df_clean[col], lower, upper)
        
        return df_clean
    
    # Metadata inicial
    metadata = {
        'filename': uploaded_file.name,
        'analysis_timestamp': datetime.now().isoformat(),
        'csv_schema': {},
        'data_quality': {},
        'numeric_statistics': {},
        'errors': [],
        'file_path': None,
        'df': None
    }
    
    try:
        df = pd.read_csv(uploaded_file)
        metadata['df'] = df
        metadata['file_path'] = uploaded_file.name
        
        # ---------------------
        # ESQUEMA Y CALIDAD DE DATOS (código existente)
        # ---------------------
        metadata['csv_schema'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'column_types': df.dtypes.astype(str).to_dict(),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024*1024), 2),
            'shape': df.shape
        }
        
        st.subheader("📊 Esquema del CSV")
        st.write(f"Filas: {metadata['csv_schema']['total_rows']}, Columnas: {metadata['csv_schema']['total_columns']}, Memoria (MB): {metadata['csv_schema']['memory_usage_mb']}")
        st.dataframe(pd.DataFrame({
            "Columna": metadata['csv_schema']['columns'],
            "Tipo de dato": [metadata['csv_schema']['column_types'][c] for c in metadata['csv_schema']['columns']]
        }), use_container_width=True)
        
        metadata['data_quality'] = {
            'null_values_per_column': df.isnull().sum().to_dict(),
            'null_percentage_per_column': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            'total_null_values': int(df.isnull().sum().sum()),
            'duplicate_rows': int(df.duplicated().sum()),
            'unique_values_per_column': df.nunique().to_dict(),
            'completeness_score': round((1 - df.isnull().sum().sum() / df.size) * 100, 2)
        }
        
        st.subheader("✅ Calidad de Datos")
        st.write(f"Valores nulos totales: {metadata['data_quality']['total_null_values']}, Filas duplicadas: {metadata['data_quality']['duplicate_rows']}, Puntaje de completitud: {metadata['data_quality']['completeness_score']}%")
        st.dataframe(pd.DataFrame({
            "Columna": list(df.columns),
            "Valores Nulos": df.isnull().sum().values,
            "% Nulos": (df.isnull().sum() / len(df) * 100).round(2).values,
            "Valores Únicos": df.nunique().values
        }), use_container_width=True)
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            metadata['numeric_statistics'] = {
                col: {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'q1': df[col].quantile(0.25),
                    'q3': df[col].quantile(0.75),
                    'median': df[col].median()
                }
                for col in numeric_cols
            }
        
        st.subheader("📋 Vista previa de los datos")
        st.dataframe(df.head(), use_container_width=True)
        
        if len(numeric_cols) > 0:
            tab1, tab2 = st.tabs(["📊 Histogramas", "📦 Boxplots"])
            with tab1:
                plot_histograms_streamlit(metadata)
            with tab2:
                plot_boxplots_streamlit(metadata)
        
        # ---------------------
        # PUNTAJES DE CALIDAD
        # ---------------------
        scores = calculate_global_score(metadata)
        
        st.subheader("🏆 Puntaje de Calidad de Datos")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Completitud", f"{scores['completeness']:.1f}%")
        col2.metric("Unicidad", f"{scores['uniqueness']:.1f}%")
        col3.metric("Outliers", f"{scores['outliers']:.1f}%")
        col4.metric("Puntaje Global", f"{scores['global_score']:.1f}%")
        
        st.progress(int(scores['global_score']))
        
        # ---------------------
        # LIMPIEZA DE DATOS (nueva funcionalidad)
        # ---------------------
        if scores['global_score'] < ALERT_THRESHOLD:
            st.warning(f"⚠️ El dataset no cumple con el estándar mínimo de calidad ({ALERT_THRESHOLD}%)")
            
            if st.button("¿Desea limpiar el dataset?"):
                with st.spinner("Limpiando dataset..."):
                    # Limpiar el dataset
                    df_clean = clean_dataset(df)
                    
                    # Generar nuevo análisis
                    clean_metadata = {
                        'filename': f"clean_{metadata['filename']}",
                        'df': df_clean,
                        'file_path': None
                    }
                    
                    # Actualizar metadata con datos limpios
                    clean_metadata['csv_schema'] = {
                        'total_rows': len(df_clean),
                        'total_columns': len(df_clean.columns),
                        'columns': list(df_clean.columns),
                        'column_types': df_clean.dtypes.astype(str).to_dict(),
                        'memory_usage_mb': round(df_clean.memory_usage(deep=True).sum() / (1024*1024), 2),
                        'shape': df_clean.shape
                    }
                    
                    clean_metadata['data_quality'] = {
                        'null_values_per_column': df_clean.isnull().sum().to_dict(),
                        'null_percentage_per_column': (df_clean.isnull().sum() / len(df_clean) * 100).round(2).to_dict(),
                        'total_null_values': int(df_clean.isnull().sum().sum()),
                        'duplicate_rows': int(df_clean.duplicated().sum()),
                        'unique_values_per_column': df_clean.nunique().to_dict(),
                        'completeness_score': round((1 - df_clean.isnull().sum().sum() / df_clean.size) * 100, 2)
                    }
                    
                    numeric_cols_clean = df_clean.select_dtypes(include=['number']).columns
                    if len(numeric_cols_clean) > 0:
                        clean_metadata['numeric_statistics'] = {
                            col: {
                                'mean': df_clean[col].mean(),
                                'std': df_clean[col].std(),
                                'min': df_clean[col].min(),
                                'max': df_clean[col].max(),
                                'q1': df_clean[col].quantile(0.25),
                                'q3': df_clean[col].quantile(0.75),
                                'median': df_clean[col].median()
                            }
                            for col in numeric_cols_clean
                        }
                    
                    # Mostrar resultados de la limpieza
                    st.success("✅ Dataset limpiado exitosamente!")
                    
                    # Mostrar nuevo análisis
                    st.subheader("📊 Análisis del Dataset Limpiado")
                    
                    # Esquema
                    st.write(f"Filas: {clean_metadata['csv_schema']['total_rows']} (original: {metadata['csv_schema']['total_rows']})")
                    st.write(f"Columnas: {clean_metadata['csv_schema']['total_columns']}")
                    st.write(f"Memoria: {clean_metadata['csv_schema']['memory_usage_mb']} MB (original: {metadata['csv_schema']['memory_usage_mb']} MB)")
                    
                    # Calidad de datos
                    st.write(f"Valores nulos totales: {clean_metadata['data_quality']['total_null_values']} (original: {metadata['data_quality']['total_null_values']})")
                    st.write(f"Filas duplicadas: {clean_metadata['data_quality']['duplicate_rows']} (original: {metadata['data_quality']['duplicate_rows']})")
                    
                    # Nuevos puntajes
                    clean_scores = calculate_global_score(clean_metadata)
                    
                    st.subheader("🏆 Nuevo Puntaje de Calidad")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Completitud", f"{clean_scores['completeness']:.1f}%", f"{clean_scores['completeness']-scores['completeness']:.1f}%")
                    col2.metric("Unicidad", f"{clean_scores['uniqueness']:.1f}%", f"{clean_scores['uniqueness']-scores['uniqueness']:.1f}%")
                    col3.metric("Outliers", f"{clean_scores['outliers']:.1f}%", f"{clean_scores['outliers']-scores['outliers']:.1f}%")
                    col4.metric("Puntaje Global", f"{clean_scores['global_score']:.1f}%", f"{clean_scores['global_score']-scores['global_score']:.1f}%")
                    
                    st.progress(int(clean_scores['global_score']))
                    
                    # Opción para descargar el dataset limpio
                    csv = df_clean.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Descargar dataset limpio",
                        data=csv,
                        file_name=f"clean_{uploaded_file.name}",
                        mime='text/csv'
                    )
                    
                    # Mostrar vista previa del dataset limpio
                    with st.expander("📋 Ver dataset limpio"):
                        st.dataframe(df_clean.head())
                    
                    # Mostrar gráficos del dataset limpio
                    if len(numeric_cols_clean) > 0:
                        tab1, tab2 = st.tabs(["📊 Histogramas (limpio)", "📦 Boxplots (limpio)"])
                        with tab1:
                            plot_histograms_streamlit(clean_metadata)
                        with tab2:
                            plot_boxplots_streamlit(clean_metadata)
        else:
            st.success(f"✅ El dataset cumple con los estándares de calidad (≥{ALERT_THRESHOLD}%)")
            
        # Mostrar pesos usados
        with st.expander("🔍 Ver detalles de ponderación"):
            st.write("Los puntajes se calculan con los siguientes pesos:")
            st.write(f"- Completitud: {WEIGHTS['completeness']*100}%")
            st.write(f"- Unicidad: {WEIGHTS['uniqueness']*100}%")
            st.write(f"- Outliers: {WEIGHTS['outliers']*100}%")
            st.write(f"Umbral mínimo aceptable: {ALERT_THRESHOLD}%")
            
    except Exception as e:
        metadata['errors'].append(str(e))
        st.error(f"❌ Error: {str(e)}")
    
    return metadata