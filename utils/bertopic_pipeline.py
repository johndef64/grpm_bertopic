import importlib
import subprocess
try:
    # Check if the module is already installed
    importlib.import_module('torch')
    print("torch is already installed.")
except ImportError:
    # If the module is not installed, try installing it
    subprocess.run(['pip3', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu118'])
    print("torch was installed correctly.")

import torch
print("Torch version:",torch.__version__)
print("Is CUDA enabled?",torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.randn(1).cuda())
else:
    #get torch here: https://pytorch.org/get-started/locally/
    subprocess.run(['pip3', 'uninstall', 'torch'])
    subprocess.run(['pip3', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu118'])


    # Import packages
import re
import os
import io
import sys
import ast
import time
import random
import zipfile
import requests


# Set data directory
save_path = 'data/'



# Define functions

def simple_bool(message):
    choose = input(message+" (y/n): ").lower()
    your_bool = choose in ["y", "yes","yea","sure"]
    return your_bool

def get_and_extract(file, dir = os.getcwd(), ext = '.zip'):
    url='https://zenodo.org/record/8205724/files/'+file+'.zip?download=1'
    zip_file_name = file+ext
    extracted_folder_name = dir
    # Download the ZIP file
    response = requests.get(url)
    if response.status_code == 200:
        # Extract the ZIP contents
        with io.BytesIO(response.content) as zip_buffer:
            with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                zip_ref.extractall(extracted_folder_name)
        print(f"ZIP file '{zip_file_name}' extracted to '{extracted_folder_name}' successfully.")
    else:
        print("Failed to download the ZIP file.")

def get_gitfile(url, flag='', dir = os.getcwd()):
    url = url.replace('blob','raw')
    response = requests.get(url)
    file_name = flag + url.rsplit('/',1)[1]
    file_path = os.path.join(dir, file_name)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully. Saved as {file_name}")
    else:
        print("Unable to download the file.")

def check_and_install_requirements(requirements):
    missing_requirements = []
    for module in requirements:
        try:
            # Check if the module is already installed
            importlib.import_module(module)
        except ImportError:
            missing_requirements.append(module)
    if len(missing_requirements) == 0:
        pass
    else:
        x = simple_bool(str(missing_requirements)+" are missing.\nWould you like to install them all?")
        if x:
            for module in missing_requirements:
                subprocess.check_call(["pip", "install", module])
                print(f"{module}' was installed correctly.")
        else:
            exit()
            
def load_preprocessed(doc_name = 'abs_preprocessed.txt'):
    with open(save_path+doc_name, 'r',encoding='utf-8') as file:
        docs_processed = []
        for line in file:
            docs_processed.append(str(line.strip()))

    print("Imported list:", doc_name)
    return docs_processed

#get_gitfile("https://raw.githubusercontent.com/johndef64/pyutilities_datascience/main/general_utilities.py")

def conservative_lower(text):
    # Split the text into words
    words = text.split()
    # Lowercase the words that don't have consecutive uppercase letters
    processed_words = [word.lower() if not re.search('[A-Z]{2,}', word) else word for word in words]
    # Join the processed words back into text
    processed_text = ' '.join(processed_words)
    return processed_text


docs =[]
#### Load Corpus
def load_corpus_from_csv(doc_name = 'scopus.csv', abs_col= 'Abstract'):
    global docs
    df = pd.read_csv(doc_name, index_col=0)
    docs = df.Abstract.drop_duplicates().to_list()
    print('\nEntry count:',len(df),
          '\nabstract count:', df[abs_col].nunique(),
          '\nEntry without abstract:',len(df)-df.Abstract.nunique())


    # Normalize docs
    # lower text keeping acronyms uppercase
    docs_to_process = docs

    timea = time.time()
    sampled_docs = docs_to_process
    docs_str = str(sampled_docs)
    docs_lower = conservative_lower(docs_str)
    docs_processed = ast.literal_eval(docs_lower)
    print('\nNormalization runtime:',time.time()-timea)
    return docs_processed


######### INSTALL REQUIREMENTS #########
requirements=['nltk','numpy','pandas','matplotlib','bertopic','wordcloud','kaleido']
check_and_install_requirements(requirements)
#!pip install bertopic[visualization]

######### TOPIC MODELING #########

import pandas as pd
import numpy as np
#from tqdm import tqdm
import matplotlib.pyplot as plt
from hdbscan import HDBSCAN
from umap import UMAP
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
from bertopic.representation import KeyBERTInspired

# Plotly: set-up
import plotly.graph_objs as go
import plotly.io as pio

#Set notebook mode to work offline
#import plotly.offline as pyo
#pyo.init_notebook_mode()

if 'google.colab' in sys.modules:
    pio.renderers.default = "colab"
    print('Google Colab detected: pio.renderers.default = "colab"')
else:
    pio.renderers.default = "notebook"
'''
Renderers configuration
    -----------------------
        Default renderer: 'notebook_connected'
        Available renderers:
            ['plotly_mimetype', 'jupyterlab', 'nteract', 'vscode',
             'notebook', 'notebook_connected', 'kaggle', 'azure', 'colab',
             'cocalc', 'databricks', 'json', 'png', 'jpeg', 'jpg', 'svg',
             'pdf', 'browser', 'firefox', 'chrome', 'chromium', 'iframe',
             'iframe_connected', 'sphinx_gallery']
'''

text_models = {1:"allenai-specter",
               # SPECTER is a model trained on scientific citations and can be used to estimate the similarity of two publications. We can use it to find similar papers.
               2:"all-mpnet-base-v2",
               # designed as general purpose model, The all-mpnet-base-v2 model provides the best quality,
               3:"all-MiniLM-L6-v2"  # while all-MiniLM-L6-v2 is 5 times faster and still offers good quality.
               } # https://www.sbert.net/docs/pretrained_models.html

model_id = input('Choose text model (id):\n'+str(text_models))
text_model = text_models[int(model_id)]
time.sleep(2)
sentence_transformer = SentenceTransformer(text_model)

def setup_model(base_embedder = text_model,
                # BERTopic
                min_topic_size = 10,   # 10 default
                top_n_words = 15,      # 10 default
                # UMAP
                n_neighbors  = 15,  # num of high dimensional neighbours
                n_components = 5,   # default:5
                random_state = 1111,
                # HDBSCAN
                min_cluster_size = 5,
                metric='euclidean',
                cluster_selection_method='eom',
                # CoutVectorizer
                stop_words="english",
                # Representation Model
                diversity = 0.1, # 0.1 default
                ):

    global sentence_transformer

    # Step 1 Extract embeddings (SBERT)
    sentence_transformer = SentenceTransformer(base_embedder) # SentenceTransformer


    # Step 2 - Reduce dimensionality
    # uniform manifold approximation and projection (UMAP) to reduce the dimension of embeddings
    umap_model = UMAP(n_neighbors  = n_neighbors,
                      n_components = n_components,
                      min_dist     = 0.0,
                      random_state = random_state) # default:None
    # https://stackoverflow.com/questions/71320201/how-to-fix-random-seed-for-bertopic


    # Step 3 - Cluster reduced embeddings
    # HDBSCAN (hierarchical density-based spatial clustering of applications with Noise)  to generate semantically similar document clusters.
    # Since HDBSCAN is a density-based clustering algorithm, the number of clusters is automatically chosen based on the minimum distance to be considered as a neighbor.
    #min_cluster_size = 5 #5 default HDBSCAN()
    hdbscan_model = HDBSCAN(min_cluster_size = min_cluster_size,
                            metric=metric,
                            cluster_selection_method=cluster_selection_method,
                            prediction_data=True)

    # Step 4 - Tokenize topics
    vectorizer_model = CountVectorizer(stop_words=stop_words, lowercase=False) # lowercase=False to keep Acronyms uppercase

    # Step 5 - Create topic representation (ctIDF)
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True) # False default

    # Step 6 - Fine-tune topic representations with a bertopic.representation model
    # Create your representation model
    representation_model = MaximalMarginalRelevance(diversity = diversity,
                                                    top_n_words = top_n_words)
    #representation_model = KeyBERTInspired() # alternative


    # Use the representation model in BERTopic on top of the default pipeline

    # All steps together
    topic_model = BERTopic(
        min_topic_size = min_topic_size,
        top_n_words = top_n_words,
        calculate_probabilities = True,
        embedding_model = sentence_transformer,      # Step 1 - Extract embeddings
        umap_model = umap_model,                     # Step 2 - Reduce dimensionality
        hdbscan_model = hdbscan_model,               # Step 3 - Cluster reduced embeddings
        vectorizer_model = vectorizer_model,         # Step 4 - Tokenize topics
        ctfidf_model = ctfidf_model,                 # Step 5 - Extract topic words
        representation_model = representation_model  # Step 6 - (Optional) Fine-tune topic representations
    )
    print('Topic Model ready\nBase embedder:',base_embedder, '\nBERTopic','\n  min_topic_size:', min_topic_size,'\n  min_topic_size:', min_topic_size,'\nUMAP','\n  n_neighbors:' ,n_neighbors,'\n  n_components:',n_components,'\n  random_state:',random_state,'\nHDBSCAN','\n  min_cluster_size', min_cluster_size )
    return topic_model


#### FIT MODEL #####
def fit_model(topic_model, docs_processed, embedding_file = ''):
    # Step 1 Embedding documents with sentence_transformer

    if embedding_file != '':
        embeddings = np.loadtxt(embedding_file)
    else:
        embeddings = sentence_transformer.encode(docs_processed,
                                                 show_progress_bar=True)

    # Train with custom embeddings
    topics, probs = topic_model.fit_transform(docs_processed, embeddings=embeddings)
    #topics = topic_model.fit(docs_processed, embeddings=embeddings)

    return topics, probs, embeddings

#### EXPLORE MODEL #####

def get_topic_info(topic_model):
    df = topic_model.get_topic_info()
    return df

def get_topics(topic_model):
    all_topics = topic_model.get_topics()
    topic_df = pd.DataFrame(all_topics)
    return topic_df

def probs2df(probs):
    probs_df = pd.DataFrame(probs)
    return probs_df

def get_topic_freq(topic_model):
    topic_freq = topic_model.get_topic_freq()
    print('total',topic_freq.Count.sum())
    return topic_freq

def get_document_info(topic_model, docs_processed):
    doc_info = topic_model.get_document_info(docs_processed)
    #print(doc_info.columns)
    return doc_info



#### MODEL VISUALIZATION (Plotly) #####

import plotly.graph_objects as go


def visualize_documents(topic_model, docs_processed, sample=1, embeddings='', custom_labels=False):
    if embeddings != '':
        map = topic_model.visualize_documents(docs_processed, embeddings=embeddings, sample=sample, custom_labels=custom_labels)
    else:
        map = topic_model.visualize_documents(docs_processed, sample=sample, custom_labels=custom_labels)

    # Create a figure with the provided map
    fig = go.Figure(data=map)

    # Show the figure
    fig.show()
    return fig


def visualize_distribution(topic_model, probs, document_id=1, width = 600, height= 400):
    fig = topic_model.visualize_distribution(probs[document_id],
                                             min_probability = 0,
                                             custom_labels = False,
                                             width = width,
                                             height= height)
    fig.show()
    return fig


def visualize_similarty(topic_model,
                        topics=None,
                        top_n_topics=None,
                        n_clusters=None,
                        width = 1100, height = 1100, save= False):
    fig = topic_model.visualize_heatmap(topics=topics,
                                        top_n_topics = top_n_topics,
                                        n_clusters = n_clusters,
                                        custom_labels=False,
                                        width = width,
                                        height = height)
    if save:
        fig.write_image("heatmap.png", engine='kaleido')
    fig.show()
    return fig



import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

def show_cosine_similarity(topic_model, topics, topic_info, width=800, height=800):

    distance_matrix = cosine_similarity(np.array(topic_model.topic_embeddings_)[1:, :])

    # get labels
    labels = topic_info[topic_info.Topic.isin(topics)].Name
    # Filter rows and columns based on indices
    topics.sort()
    rows_to_keep = topics
    cols_to_keep = topics
    filtered_distance_matrix = distance_matrix[np.ix_(rows_to_keep, cols_to_keep)]

    fig = px.imshow(filtered_distance_matrix,
                    labels=dict(color="Similarity Score"),
                    x=labels,
                    y=labels,
                    color_continuous_scale='GnBu'
                    )
    title: str = "<b>Similarity Matrix</b>"
    fig.update_layout(
        title={
            'text': f"{title}",
            'y': .95,
            'x': 0.55,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        width=width,
        height=height,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )
    fig.update_layout(showlegend=True)
    fig.update_layout(legend_title_text='Trend')
    fig.write_image("heatmap_short.png", engine="kaleido", scale=3)
    fig.show()
    return fig


def visualize_hierarchy(topic_model, width=700, height=600):
    fig = topic_model.visualize_hierarchy(width=width, height=height) #The topics that were created can be hierarchically reduced.
    fig.show()
    return fig


def visualize_barchart(topic_model, topics, n_words=20):
    fig = topic_model.visualize_barchart(n_words = n_words,
                                         topics = topics,
                                         #top_n_topics=len(topic_info)//4,
                                         )
    fig.show()
    return fig


from wordcloud import WordCloud
import matplotlib.pyplot as plt

def create_wordcloud(topic_model, topic_id):
    text = {word: value for word, value in topic_model.get_topic(topic_id)}
    print(text)
    wc = WordCloud(background_color="white", max_words=1000, width=800, height=400)
    wc.generate_from_frequencies(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    return plt

def create_wordcloud_multiple(topic_model, topics, output_path='wordcloud.png', dpi=300, save=True):
    merged_dict = {}
    for i in topics:
        text = {word: value for word, value in topic_model.get_topic(i)}
        merged_dict.update(text)

    plt.figure(figsize=(12, 8))

    wc = WordCloud(background_color="white", max_words=1000, width=1000, height=500)
    wc.generate_from_frequencies(merged_dict)

    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    if save:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    return plt

def create_wordcloud_from_corpus(corpus, output_path='wordcloud.png', dpi=300, save=True):
    # Combine the text corpus into a single string
    text = ' '.join(corpus)
    # Generate WordCloud from the text
    wc = WordCloud(background_color="white", max_words=1000, width=800, height=400)
    wc.generate(text)
    # Display the WordCloud
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    if save:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    return plt


help=r'''
# Basic Usage as py module

#### IMPORT #####
import bertopic_abstract as bt

#### SETUP MODEL #####
bert_abs = bt.load_corpus_from_csv("data/scopus_trial.csv")
topic_model = bt.setup_model(
                # BERTopic
                min_topic_size = 10,   # 10 default
                top_n_words = 15,      # 10 default
                
                # UMAP
                n_neighbors  = 15, # num of high dimensional neighbours
                n_components = 5,  # default:5 #30
                random_state = 1111,
                
                # HDBSCAN
                min_cluster_size = 5,
                metric='euclidean',
                cluster_selection_method='eom',
                
                # CoutVectorizer
                stop_words="english",
                
                # Representation Model
                diversity = 0.1, # 0.1 default
                 )

#### FIT MODEL #####
topics, probs, embeddings = bt.fit_model(topic_model, bert_abs)

#### EXPLORE MODEL #####
topic_info = bt.get_topic_info(topic_model)
document_info = bt.get_document_info(topic_model, bert_abs)

#### MODEL VISUALIZATION (Plotly) #####
fig_documents = bt.visualize_documents(topic_model, bert_abs)
doc_distribution = bt.visualize_distribution(topic_model, probs, 1)
fig_similarty = bt.visualize_similarty(topic_model, width = 500, height = 500)
fig_hierarchy = bt.visualize_hierarchy(topic_model, width=700, height=600)

#### WORD CLOUDS #####
topic_cloud = bt.create_wordcloud(topic_model, 1) #topic_id
topics_cloud = bt.create_wordcloud_multiple(topic_model, topics, output_path='wordcloud.png', dpi=300, save=True)
'''
#%%
