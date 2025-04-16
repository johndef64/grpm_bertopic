

# GRPM BERTopic Analysis
This pipeline is designed to identify significant patterns and relationships within a set of abstracts, offering insights into potential genetic contributions within the biomedical domain.

The topic modeling and data retrieval pipeline is available as Jupyter notebook (`grpm_bertopic.ipynb`) and is designed to unravel hidden topic among PubMed genetic literature.

[![Manuscript](https://img.shields.io/badge/paper-10.1101/2023.08.04.23293659-blue.svg)](https://link.springer.com/chapter/10.1007/978-3-031-78093-6_1)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/johndef64/grpm_bertopic/blob/main/grpm_bertopic.ipynb)


## Repository Structure
- `data`: Harvest the data produced during the notebook execution.
- `utils`: Contains accessory python code.
- `grpm_bertopic.ipynb` : The Jupyter notebook file contains all the steps, code and detailed information about the GRPM BERTopic analysis process.
- `bertopic_tutorial.ipynb`: Jupyter notebook created for educational purposes.


## Usage
To perform the GRPM BERTopic analysis, follow the steps laid out in the `grpm_bertopic.ipynb` notebook. Following these steps, you'll be able to unravel the intricate connections between genetic variations and MeSH term provided.

The general workflow has been depicted below:
![Workflow](data/workflow_chart.png)

## About GRPM BERTopic Analysis

The GRPM BERTopic Analysis utilizes a structured approach to extract and examine themes from a collection of scholarly abstracts related to human genetic polymorphisms. Below outlines the key steps of the analysis:

 ### **Dataset Acquisition**: 
The analysis starts by retrieving source dataset of scholarly abstracts focusing on human genetic polymorphisms. This dataset, termed the GRPM Dataset, integrates data from sources like LitVar and PubMed. 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14052302.svg)](https://doi.org/10.5281/zenodo.14052302)

### 1. **Data Preprocessing**: 
The preprocessing phase utilizes a user-defined set of Medical Subject Headings (MeSH) terms to curate the corpus of abstracts. An example of MeSH terms is available in the `data/ref-mesh.csv` file. This step is crucial as it refines the abstracts' corpus, preparing it for effective topic modeling.

### 2. **Topic Modeling**: 
The refined corpus undergoes topic modeling using the BERTopic architecture. This framework employs advanced hierarchical clustering techniques to uncover the latent thematic structures of the abstracts, providing a comprehensive overview of the topic model's underlying architecture.

### 3. **Data Post-processing**: 
Finally, selected topics undergo post-processing, highlighting specific themes for in-depth exploration. This stage enhances the understanding of genetic influences pertinent to biomedical fields, as specified by the custom MeSH terms.


If you encounter any issues or have any questions, feel free to open an issue in this repository.

## Requirements
All required libraries and their specific versions used for this project are listed within the `grpm_bertopic.ipynb`. Make sure to install these dependencies before go through the notebook.

## BERTopic Tutorial

For didactic purposes, a commented use case is available, unfolding each BERTopic component.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/johndef64/grpm_bertopic/blob/main/bertopic_tutorial.ipynb)

