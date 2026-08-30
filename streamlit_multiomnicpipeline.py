import os
import time
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import io
from scipy.stats import gaussian_kde

# ==============================================================================
# 1. API MANAGEMENT & EXTERNAL INTEGRATIONS
# ACADEMIC RATIONALE: Serverless Clinical Database Integration
# Replaces the local Ensembl-VEP binary and 40GB+ GRCh38 cache with live calls 
# to the Ensembl REST API. Allows real-time clinical variant annotation against 
# the public human genome database without local cache requirements.
# Matches original script Step 13.
# ==============================================================================
ENSEMBL_REST_SERVER = "https://rest.ensembl.org"
ENSEMBL_VEP_ENDPOINT = "/vep/human/hgvs/{variant_hgvs}"
API_HEADERS = {"Content-Type": "application/json"}

@st.cache_data(ttl=3600)
def fetch_ensembl_vep_live(variant_hgvs: str) -> dict:
    url = f"{ENSEMBL_REST_SERVER}{ENSEMBL_VEP_ENDPOINT.format(variant_hgvs=variant_hgvs)}"
    try:
        response = requests.get(url, headers=API_HEADERS, timeout=5)
        if response.ok:
            data = response.json()[0]
            conseq = data.get("transcript_consequences", [{}])[0]
            return {
                "Assembly": "GRCh38",
                "Consequence": data.get("most_severe_consequence", "unknown").replace("_", " ").title(),
                "Gene": conseq.get("gene_symbol", "unknown"),
                "Impact": conseq.get("impact", "unknown")
            }
        return {"Status": "Variant not found in current build."}
    except Exception as e:
        return {"Status": f"API Connection Error: {str(e)}"}

# ==============================================================================
# 2. STREAMING I/O ENGINE
# ACADEMIC RATIONALE: Streaming IO
# Handles large uploaded FASTQ/FASTA/BAM files via chunked byte buffers. 
# Prevents RAM overflow on serverless platforms by yielding data line-by-line 
# rather than loading multi-gigabyte genomic files into memory simultaneously.
# ==============================================================================
def process_streaming_upload(uploaded_file, chunk_size=8192):
    """Generator to process files in memory-safe chunks."""
    if uploaded_file is not None:
        buffer = io.TextIOWrapper(uploaded_file, encoding='utf-8')
        while True:
            chunk = buffer.read(chunk_size)
            if not chunk:
                break
            yield chunk

# ==============================================================================
# 3. PAGE CONFIGURATION & STATE ENGINE
# ACADEMIC RATIONALE: Stateful Pipeline Engine
# Uses st.session_state to step through the 15 stages dynamically. 
# Mimics the original orchestrator's '--start-at' crash recovery and resume logic.
# ==============================================================================
st.set_page_config(page_title="EV Cargo Pipeline", page_icon="🔬", layout="wide")

if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'assay' not in st.session_state:
    st.session_state.assay = "cfDNA"
if 'pipeline_stage' not in st.session_state:
    st.session_state.pipeline_stage = 1

def reset_app():
    st.session_state.analyzed = False
    st.session_state.pipeline_stage = 1

# ==============================================================================
# 4. INTERACTIVE UX (PLOTLY ENGINE)
# ACADEMIC RATIONALE: Interactive UX
# Upgrades static Matplotlib outputs to interactive Plotly graphs (Volcano, 
# Lollipop, Fragment distribution) for clinical review. Allows zooming on specific
# genomic coordinates or transcriptomic outliers.
# ==============================================================================
def generate_vaf_plot():
    """Step 9: VAF Spectrum Distribution Histogram"""
    vaf_mock = np.random.exponential(scale=0.01, size=100)
    vaf_mock = vaf_mock[vaf_mock < 0.1]
    fig = px.histogram(x=vaf_mock * 100, nbins=30, color_discrete_sequence=["#D55E00"])
    fig.add_vline(x=0.1, line_dash="dash", line_color="black", annotation_text="LOD (0.1%)")
    fig.update_layout(title="Step 9: VAF Spectrum", template="simple_white", xaxis_title="VAF (%)", yaxis_title="Count")
    fig.update_xaxes(showline=True, linewidth=1.5, linecolor='black', mirror=True, ticks="outside")
    fig.update_yaxes(showline=True, linewidth=1.5, linecolor='black', mirror=True, ticks="outside")
    return fig

def generate_volcano_plot():
    """Generates an RNA Differential Expression Volcano Plot."""
    n_genes = 500
    df = pd.DataFrame({
        'Gene': [f"GENE_{i}" for i in range(n_genes)],
        'log2FC': np.random.normal(0, 1.2, n_genes),
        'neg_log10_pval': np.random.exponential(0.8, n_genes)
    })
    
    outliers = pd.DataFrame({
        'Gene': ['ERBB2', 'CD274', 'MYC'],
        'log2FC': [4.5, 3.1, 3.8],
        'neg_log10_pval': [8.2, 5.4, 7.5]
    })
    df = pd.concat([df, outliers], ignore_index=True)
    
    df['Status'] = 'Not Significant'
    df.loc[(df['log2FC'] >= 1.5) & (df['neg_log10_pval'] >= 1.3), 'Status'] = 'Upregulated'
    df.loc[(df['log2FC'] <= -1.5) & (df['neg_log10_pval'] >= 1.3), 'Status'] = 'Downregulated'
    
    color_map = {'Not Significant': 'grey', 'Upregulated': '#D55E00', 'Downregulated': '#0072B2'}
    
    fig = px.scatter(df, x='log2FC', y='neg_log10_pval', color='Status', hover_name='Gene', color_discrete_map=color_map)
    fig.add_vline(x=1.5, line_dash="dash", line_color="black", opacity=0.4)
    fig.add_vline(x=-1.5, line_dash="dash", line_color="black", opacity=0.4)
    fig.add_hline(y=1.3, line_dash="dash", line_color="black", opacity=0.4, annotation_text="p=0.05")
    
    fig.update_layout(title="<b>Fig 2.</b> Differential Expression Profile", template="simple_white", xaxis_title="log2(Fold Change)", yaxis_title="-log10(p-value)")
    fig.update_xaxes(showline=True, linewidth=1.5, linecolor='black', mirror=True, ticks="outside")
    fig.update_yaxes(showline=True, linewidth=1.5, linecolor='black', mirror=True, ticks="outside")
    return fig

def plot_academic_fragment_size(sim_sizes, assay_type):
    """Step 6: Generates a publication-ready KDE overlay on normalized histogram."""
    df = pd.DataFrame({'Size': sim_sizes})
    kde = gaussian_kde(df['Size'])
    x_range = np.linspace(df['Size'].min(), df['Size'].max(), 500)
    y_kde = kde(x_range)
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df['Size'], histnorm='probability density', 
        name='Observed Fragments', marker_color='#E69F00', opacity=0.6, nbinsx=80
    ))
    fig.add_trace(go.Scatter(
        x=x_range, y=y_kde, mode='lines', 
        name='Kernel Density Estimate', line=dict(color='#0072B2
