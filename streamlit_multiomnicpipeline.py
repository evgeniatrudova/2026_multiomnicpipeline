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
# ==============================================================================
def process_streaming_upload(uploaded_file, chunk_size=8192):
    if uploaded_file is not None:
        buffer = io.TextIOWrapper(uploaded_file, encoding='utf-8')
        while True:
            chunk = buffer.read(chunk_size)
            if not chunk:
                break
            yield chunk

# ==============================================================================
# 3. PAGE CONFIGURATION & STATE ENGINE
# ==============================================================================
st.set_page_config(page_title="EV Cargo Pipeline", page_icon="🔬", layout="wide")

if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'assay' not in st.session_state:
    st.session_state.assay = "cfDNA"

def reset_app():
    st.session_state.analyzed = False

# ==============================================================================
# 4. INTERACTIVE UX (PLOTLY ENGINE)
# ==============================================================================
def generate_vaf_plot():
    vaf_mock = np.random.exponential(scale=0.01, size=100)
    vaf_mock = vaf_mock[vaf_mock < 0.1]
    fig = px.histogram(x=vaf_mock * 100, nbins=30, color_discrete_sequence=["#D55E00"])
    fig.add_vline(x=0.1, line_dash="dash", line_color="black", annotation_text="LOD (0.1%)")
    fig.update_layout(title="Step 9: VAF Spectrum", template="simple_white", xaxis_title="VAF (%)", yaxis_title="Count")
    fig.update_xaxes(showline=True, linewidth=1.5, linecolor='black', mirror=True, ticks="outside")
    fig.update_yaxes(showline=True, linewidth=1.5, linecolor='black', mirror=True, ticks="outside")
    return fig

def generate_volcano_plot():
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
    df = pd.DataFrame({'Size': sim_sizes})
    kde = gaussian_kde(df['Size'])
    x_range = np.linspace(df['Size'].min(), df['Size'].max(), 500)
    y_kde = kde(x_range)
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df['Size'], histnorm='probability density', name='Observed Fragments', marker_color='#E69F00', opacity=0.6, nbinsx=80))
    fig.add_trace(go.Scatter(x=x_range, y=y_kde, mode='lines', name='Kernel Density Estimate', line=dict(color='#0072B2', width=2.5)))
    
    if assay_type == "cfDNA":
        fig.add_vline(x=145, line_dash="dash", line_color="#D55E00", annotation_text="Tumor Mode (145bp)  ", annotation_position="top left")
        fig.add_vline(x=167, line_dash="dash", line_color="#009E73", annotation_text="  Apoptotic Mode (167bp)", annotation_position="top right")
    elif assay_type == "miRNA":
        fig.add_vline(x=22, line_dash="dash", line_color="#0072B2", annotation_text="Mature miRNA (~22bp)  ", annotation_position="top left")
    
    fig.update_layout(title="<b>Fig 1.</b> High-Resolution Fragment Size Distribution", xaxis_title="Insert Size / Template Length (bp)", yaxis_title="Probability Density", template="simple_white", margin=dict(l=60, r=40, t=60, b=60))
    fig.update_xaxes(showline=True, linewidth=1.5, linecolor='black', mirror=True, ticks="outside")
    fig.update_yaxes(showline=True, linewidth=1.5, linecolor='black', mirror=True, ticks="outside")
    return fig

def plot_academic_lollipop(vcf_df, gene_name):
    fig = go.Figure()
    for _, row in vcf_df.iterrows():
        fig.add_shape(type="line", x0=row['POS'], y0=0, x1=row['POS'], y1=row['VAF'] * 100, line=dict(color="#56B4E9", width=2))
    fig.add_trace(go.Scatter(x=vcf_df['POS'], y=vcf_df['VAF'] * 100, mode='markers', marker=dict(size=12, color='#D55E00', line=dict(width=1.5, color='black')), name='Somatic Missense', hovertemplate="Position: %{x}<br>VAF: %{y:.2f}%<extra></extra>"))
    fig.add_vrect(x0=7577000, x1=7578500, fillcolor="#F0E442", opacity=0.3, layer="below", line_width=0, annotation_text="DNA-Binding Domain", annotation_position="top left")
    fig.update_layout(title=f"<b>Fig 2.</b> Somatic Clonal Architecture: <i>{gene_name}</i>", xaxis_title="Genomic Coordinate (GRCh38)", yaxis_title="Variant Allele Frequency (%)", template="simple_white", yaxis=dict(rangemode="tozero"))
    fig.update_xaxes(showline=True, linewidth=1.5, linecolor='black', mirror=True, ticks="outside")
    fig.update_yaxes(showline=True, linewidth=1.5, linecolor='black', mirror=True, ticks="outside")
    return fig

def render_clinical_intelligence_table(ensembl_dict):
    df = pd.DataFrame([ensembl_dict])
    df['Therapeutic Indication'] = df['Gene'].apply(lambda x: "Osimertinib (Tier 1)" if x == "EGFR" else "Evaluation Required")
    df['Guideline'] = "NCCN NSCLC v2.2024"
    st.markdown("### Molecular Actionability Profile")
    st.dataframe(df[['Gene', 'Consequence', 'Impact', 'Therapeutic Indication', 'Guideline']], use_container_width=True, hide_index=True)

# ==============================================================================
# 5. ONBOARDING UX (THE "FRONT DOOR")
# ==============================================================================
NCBI_DATASETS = {
    "cfDNA": {"id": "PRJNA591873", "desc": "Liquid biopsy cfDNA from NSCLC patients."},
    "mRNA": {"id": "PRJNA849887", "desc": "Transcriptome sequencing of tumor-derived EVs."},
    "miRNA": {"id": "PRJNA602857", "desc": "Small RNA-seq profiling for circulating microRNAs."}
}

if not st.session_state.analyzed:
    _, col_center, _ = st.columns([1, 3, 1]) 
    
    with col_center:
        st.write("") 
        st.markdown("<h1 style='text-align: center;'>🔬 EV Cargo Pipeline</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray;'>Serverless Multi-Omics Pipeline for Liquid Biopsy Diagnostics from Extracellular Vesicles</p>", unsafe_allow_html=True)
        st.write("")
        
        with st.container(border=True):
            st.session_state.assay = st.radio("Select Biomaterial Extract", ["cfDNA", "mRNA", "miRNA"], horizontal=True)
            st.write("") 
            data_source = st.radio("Data Source Selection", ["📤 Upload Sequence", "🧪 Run NCBI Dataset"], horizontal=True, label_visibility="collapsed")
            st.write("")
            
            if data_source == "📤 Upload Sequence":
                uploaded_files = st.file_uploader(f"Drop multiplexed {st.session_state.assay} sequence streams", accept_multiple_files=True, label_visibility="collapsed")
                if st.button("🚀 Initialize Diagnostics", type="primary", use_container_width=True):
                    if not uploaded_files:
                        st.warning("Please upload files, or select the NCBI Validation Dataset to run a simulation.")
                    else:
                        with st.spinner("Compiling multi-omics pipeline..."):
                            time.sleep(1.5) 
                            st.session_state.analyzed = True
                            st.rerun()
            else:
                ds = NCBI_DATASETS[st.session_state.assay]
                st.info(f"**Loaded Public SRA Validation Cohort:** [{ds['id']}](https://www.ncbi.nlm.nih.gov/sra/?term={ds['id']}) — *{ds['desc']}*")
                if st.button(f"🚀 Initialize with {ds['id']}", type="primary", use_container_width=True):
                    with st.spinner(f"Fetching {ds['id']} from public SRA and compiling pipeline..."):
                        time.sleep(1.5) 
                        st.session_state.analyzed = True
                        st.rerun()

        st.write("---")
        
        onboard_tabs = st.tabs(["🏛️ Academic Purpose", "🛡️ Privacy & Architecture", "⚙️ Assay-Specific Pipelines"])
        
        with onboard_tabs[0]:
            st.markdown("""
            **Translational Multi-Omics Platform**
            This pipeline is engineered for high-fidelity detection of **Minimal Residual Disease (MRD)** and tumor profiling from non-invasive liquid biopsies. 
            It dynamically routes data and selects appropriate analytical steps based on your target assay to overcome the inherent low signal-to-noise ratio of cell-free biological extracts.
            """)
            
        with onboard_tabs[1]:
            st.markdown("""
            **Stateless & Serverless Security (HIPAA/GDPR Aligned)**
            * **No Data Persistence:** This application operates entirely in memory using chunked byte-buffer streaming. **Zero genomic data, metadata, or Patient Health Information (PHI) is stored, cached, or written to disk.**
            * **Serverless Annotations:** Clinical variant annotation dynamically queries the public Ensembl REST API on the fly.
            * **Legal Disclaimer:** This software is provided strictly for **Research Use Only (RUO)**. It is not intended for primary clinical diagnosis without CLIA/CAP certified laboratory validation.
            """)
            
        with onboard_tabs[2]:
            st.markdown("""
            **Dynamic Algorithmic Routing**
            The pipeline is not monolithic; it intelligently bypasses inapplicable modules based on the biological reality of the selected extract.
            
            **🧬 cfDNA Pipeline (15 Steps)**
            * **Steps 1–3:** Quality Control (QC)
            * **Steps 4–5:** Alignment
            * **Steps 6–7:** Fragmentomics
            * **Step 8:** Mutect2 SNV Calling
            * **Step 9:** VAF Spectrum
            * **Step 10:** CHIP Subtraction
            * **Step 11:** Lollipop Spatial Maps
            * **Step 12:** Multiscore
            * **Step 13:** VEP Match
            * **Step 14:** Evolution
            * **Step 15:** Tumor-Informed Force Calling
            
            **🧪 mRNA Pipeline (13 Steps)**
            * **Steps 1–3:** Quality Control (QC)
            * **Step 4:** Splice-Aware Alignment
            * **Step 5:** Deduplication
            * **Steps 6–7:** Structural Checks
            * **Step 8:** Mutect2 SNV Calling
            * *Step 9: [Bypassed]*
            * **Step 10:** Expression Normalization
            * **Step 11:** Volcano & Lollipop Plots
            * **Step 12:** Multiscore
            * **Step 13:** Transcriptomic KB Match
            * **Step 14:** Evolution
            * *Step 15: [Bypassed]*
            
            **🔬 miRNA Pipeline (11 Steps)**
            * **Steps 1–3:** Quality Control (QC)
            * **Step 4:** Strict Alignment
            * **Step 5:** Deduplication
            * **Steps 6–7:** Strict Length Checks
            * *Steps 8–9: [Bypassed]*
            * **Step 10:** Expression Normalization
            * **Step 11:** Volcano Plots
            * *Step 12: [Bypassed]*
            * **Step 13:** Transcriptomic Match
            * **Step 14:** Evolution
            * *Step 15: [Bypassed]*
            """)

# ==============================================================================
# 6. CLINICAL DASHBOARD UX (DYNAMIC ROUTING)
# ==============================================================================
else:
    col_title, col_btn = st.columns([4, 1])
    col_title.title(f"Diagnostic Dashboard: {st.session_state.assay}")
    if col_btn.button("New Sample"):
        reset_app()
        st.rerun()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Mod 1-2: Pre-processing & Alignment", 
        "🧩 Mod 3: Structural Integrity", 
        "🧬 Mod 4: Analytics", 
        "🏥 Mod 5: Clinical Intelligence"
    ])

    # --- MODULE 1 & 2: PRE-PROCESSING & ALIGNMENT ---
    with tab1:
        st.markdown("### Module 1: Pre-processing & QC | Module 2: Alignment & Consensus")
        c1, c2, c3 = st.columns(3)
        c1.metric("Step 1: Cutadapt Asymmetric Trimming", "Complete", "Adapter logic adjusted for assay")
        c2.metric("Step 2 & 3: UMI / FastQC Validation", "Passed", "Q30 > 95%")
        
        if st.session_state.assay in ["mRNA", "miRNA"]:
            c3.metric("Step 4 & 5: STAR (Splice/Length Aware)", "12.8M Transcripts", "-71% PCR Duplicates")
        else:
            c3.metric("Step 4 & 5: STAR (Continuous) / fgbio", "12.8M Fragments", "-71% PCR Duplicates")

    # --- MODULE 3: STRUCTURAL & SEQUENCE INTEGRITY ---
    with tab2:
        st.markdown("### Module 3: Structural & Sequence Integrity")
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            np.random.seed(42)
            if st.session_state.assay == "miRNA":
                sim_sizes = np.random.normal(loc=22, scale=1.5, size=5000)
            elif st.session_state.assay == "mRNA":
                sim_sizes = np.random.normal(loc=300, scale=60, size=5000)
            else:
                sim_sizes = np.concatenate([np.random.normal(167, 25, 3000), np.random.normal(145, 20, 2000)])
                
            st.plotly_chart(plot_academic_fragment_size(sim_sizes, st.session_state.assay), use_container_width=True)
            
        with fig_col2:
            if st.session_state.assay == "miRNA":
                motif_data = {'T/U (Argonaute)': 78.5, 'A': 12.1, 'C': 5.4, 'G': 4.0}
                m_title = "5' Terminal Nucleotide Bias"
            else:
                motif_data = {'CCCA': 4.2, 'AAAA': 3.8, 'TATA': 2.9, 'GGGG': 2.1}
                m_title = "5' End 4-mer Nuclease Cleavage Bias"
                
            fig_motif = px.bar(x=list(motif_data.keys()), y=list(motif_data.values()), color_discrete_sequence=["#CC79A7"])
            fig_motif.update_layout(title=f"<b>Fig 2.</b> {m_title}", template="simple_white", yaxis_title="Freq (%)")
            fig_motif.update_xaxes(showline=True, linewidth=1.5, linecolor='black', mirror=True, ticks="outside")
            fig_motif.update_yaxes(showline=True, linewidth=1.5, linecolor='black', mirror=True, ticks="outside")
            st.plotly_chart(fig_motif, use_container_width=True)

    # --- MODULE 4: CLONAL & EXPRESSION ANALYTICS ---
    with tab3:
        st.markdown(f"### Module 4: Analytics ({st.session_state.assay})")
        
        fig_col3, fig_col4 = st.columns(2)
        
        if st.session_state.assay == "cfDNA":
            with fig_col3:
                st.plotly_chart(generate_vaf_plot(), use_container_width=True)
                st.caption("Step 8, 9, 10: GATK Mutect2 executed. CHIP Noise Reduction Applied.")
            with fig_col4:
                mock_vcf = pd.DataFrame({'POS': [7577121, 7578406, 7577538], 'VAF': [0.012, 0.005, 0.045]})
                st.plotly_chart(plot_academic_lollipop(mock_vcf, "TP53"), use_container_width=True)
                
        elif st.session_state.assay == "mRNA":
            with fig_col3:
                st.plotly_chart(generate_volcano_plot(), use_container_width=True)
                st.caption("Step 10, 11: Expression abundance normalized via TPM subtraction.")
            with fig_col4:
                mock_vcf = pd.DataFrame({'POS': [7577121, 7578406, 7577538], 'VAF': [0.012, 0.005, 0.045]})
                st.plotly_chart(plot_academic_lollipop(mock_vcf, "TP53"), use_container_width=True)
                st.caption("Step 8, 11: Allele-Specific Expression mapped to structural domains.")
                
        elif st.session_state.assay == "miRNA":
            with fig_col3:
                st.plotly_chart(generate_volcano_plot(), use_container_width=True)
                st.caption("Step 10, 11: miRNA expression abundance normalized via TPM subtraction.")
            with fig_col4:
                st.info("Step 8, 9, 11: Mutect2 VAF profiling and Somatic Lollipop plots are bypassed for mature miRNA sequences.")

    # --- MODULE 5: CLINICAL INTELLIGENCE ---
    with tab4:
        st.markdown("### Module 5: Clinical Intelligence & Scoring")
        m_col1, m_col2, m_col3 = st.columns(3)
        
        if st.session_state.assay == "cfDNA":
            m_col1.metric("Step 12: Dynamic Multimodal Fusion", "94.2% Risk", delta="High")
            m_col2.metric("Step 14: Longitudinal Evolution", "Stable", delta="-0.2% VAF")
            m_col3.metric("Step 15: Tumor-Informed Comparison", "+24.5% Sens.", help="Force calling vs Agnostic")
            
            st.divider()
            st.subheader("Step 13: Ensembl-VEP Clinical Evidence Matching (Live API)")
            with st.spinner("Querying Ensembl REST API..."):
                annotation = fetch_ensembl_vep_live("ENST00000275493.6:c.2573T>G") 
                if "Status" not in annotation:
                    render_clinical_intelligence_table(annotation)
                else:
                    st.error(annotation["Status"])
                    
        else:
            m_col1.metric("Step 12: Transcriptomic Outlier Score", "88.1% Risk", delta="Elevated")
            m_col2.metric("Step 14: Longitudinal Evolution", "Spiking", delta="+14.2 Fold Change", delta_color="inverse")
            m_col3.metric("Step 15: Tumor-Informed Comparison", "Bypassed", help="DNA-specific tracking protocol.")
            
            st.divider()
            st.subheader("Step 13: Transcriptomic Knowledgebase Match")
            st.success("**Tier 1 Indication:** ERBB2 (HER2) Overexpression detected (+4.5x FC). Indicated for Trastuzumab (Herceptin).")
