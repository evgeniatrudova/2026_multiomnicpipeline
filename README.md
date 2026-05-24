## Multi-Omics Liquid Biopsy Pipeline

This automated pipeline is designed for the detection of ultra-low frequency somatic mutations. It dynamically routes data and selects appropriate analytical steps based on your target assay: cfDNA, mRNA, or miRNA.To optimize computational resources and ensure system stability, this pipeline operates on a strict two-tier architecture:

Mock Mode: A lightweight, local simulation to verify structural integrity and connectivity.

Production Mode: Heavy, server-side execution for biological data analysis.

⚠️ WARNING: You must never run a Production execution without first passing a Mock test to validate your environment.


##  Environment & Dependencies

Before running the pipeline, activate your Conda environment and install the required bioinformatic tools.
Bash
conda install -c bioconda -c conda-forge cutadapt fastqc star fgbio samtools gatk4 ensembl-vep pyyaml pysam pandas numpy matplotlib seaborn -y

Troubleshooting Missing Dependencies If the pipeline fails due to a missing tool, verify its installation and reinstall it using the following commands (replace [dependency_name] with the missing tool, e.g., fastqc):

Bash
conda --version
which [dependency_name]   
conda install -c bioconda [dependency_name]


## Directory & File Preparation

Download the Pipeline: Place the 2026_multiomnicpipeline directory in an accessible location (e.g., your Desktop).
Windows Users: To find your exact path, right-click the folder and select "Copy as path". It should resemble: "C:\Users\YourName\Desktop\2026_multiomnicpipeline".

Reference Genome: Ensure the GRCh38 Human Reference Genome is downloaded to the server. If it is already hosted centrally, obtain the direct directory path to its location.


##  Server Connectivity

Log into your institutional server to locate your raw FASTQ data and the pipeline scripts.Connect to the server
ssh student@uni.edu

Verify your current directory
pwd
ls

Navigate to the pipeline trial directory
cd /path/to/2026_multiomnicpipeline/trial

Verify your raw data files are present
ls -lh /path/to/raw_data/

##  Local Testing in Mock Mode

Validate your setup locally before initiating a server run. Ensure your terminal is currently inside the 2026_multiomnicpipeline/trial directory.
The following example tests the miRNA track. It will pull dummy data from the mock folder and successfully generate a RESULTS_MIRNA_TEST_01 folder inside the mock_results directory.

Bash
python step0_main_pipeline.py \
  --raw_data "C:\Users\evgen\OneDrive\Skrivbord\2026_multiomnicpipeline\mock" \
  --sample MIRNA_TEST_01 \
  --assay mirna \
  -t 4 \
  --mode mock

## Server Execution in Production Mode

Once the mock test passes, you are ready to analyze real biological data.

--raw_data: The absolute path to the folder containing your raw .fastq.gz files.
--sample: The designated name for your sample (e.g., TUMOR_01).
-t: CPU threads to allocate. Default is 16. (Note: If the server is under heavy load, lower this to 8 or 4 to prevent crashes).
--start-at: Crash recovery. If the pipeline fails, change this to the step number where it stopped to resume progress (Steps 1–4 are universal across all assays).
Select the command corresponding to your assay. Replace the /path/to/... placeholders with your actual server paths.

##  cfDNA

Bash
python step0_main_pipeline.py \
  --raw_data /path/to/my_raw_data_folder \
  --sample TUMOR_CFDNA_01 \
  --assay cfdna \
  -t 16 \
  --start-at 1 \
  --mode prod

## mRNA
Bash
python step0_main_pipeline.py \
  --raw_data /path/to/my_raw_data_folder \
  --sample TUMOR_MRNA_01 \
  --assay mrna \
  -t 16 \
  --start-at 1 \
  --mode prod


## miRNA
Bash
python step0_main_pipeline.py \
  --raw_data /path/to/my_raw_data_folder \
  --sample TUMOR_MIRNA_01 \
  --assay mirna \
  -t 16 \
  --start-at 1 \
  --mode prod
