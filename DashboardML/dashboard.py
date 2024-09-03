import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import pandas as pd
from gensim.models import LdaModel, Word2Vec
from gensim import corpora
import numpy as np
import pickle
import os
import seaborn as sns
from classes import MeanEmbeddingVectorizer, LDAEmbeddingVectorizer, Classifier

# ----------------------------------------------------
#
#              Code for tabs 1 and 2
#
# ----------------------------------------------------
# Load variables from the file (Depends on which folder is open in vscode)
with open('variables.pkl', 'rb') as f:
    lda_model, corpus_bow, dictionary = pickle.load(f)

# Load in our already preprocessed df (Depends on which folder is open in vscode)
file_path = 'dataframe/all_data.csv'
trainset_df = pd.read_csv(file_path)

num_topics = lda_model.num_topics
passes = lda_model.passes

# Function to determine the predominant LDA topic of each document
def get_predominant_topic(lda_vectors):
    predominant_topics = []
    for vector in lda_vectors:
        max_topic = max(vector, key=lambda x: x[1])
        predominant_topics.append(max_topic[0])
    return predominant_topics

# Get vector representation of documents
document_vectors = [lda_model[doc] for doc in corpus_bow]

# Get predominant LDA topic for each document
predominant_topics = get_predominant_topic(document_vectors)
trainset_df['predominant_topic'] = predominant_topics

# Get counts of documents in each topic
topic_counts = trainset_df['predominant_topic'].value_counts().reset_index()
topic_counts.columns = ['Topic', 'Count']

lda_vis_data = gensimvis.prepare(lda_model, corpus_bow, dictionary)

# Get topic-document similarity matrix
similarity_matrix = np.zeros((num_topics, len(corpus_bow)))
for i, doc_vector in enumerate(document_vectors):
    for topic, prob in doc_vector:
        similarity_matrix[topic, i] = prob

# ----------------------------------------------------
#
#                   Code for tab 3
#
# ----------------------------------------------------

# Paths to vcs and knn csv's (Depends on which folder is open in vscode)
svc_path = 'dataframe/SVC/'
knn_path = 'dataframe/KNN/'
svc_files = os.listdir(svc_path)
knn_files = os.listdir(knn_path)
available_datasets = [file.split('_')[0] for file in knn_files] # they have the same files+names except for "_modelname.csv"

# Dictionary for dropdown menu labels
prettyTextNames = {
    "Benefit": "Benefits",
    "Benefit+Side" : "Benefits + Side effects",
    "benefits+sideEffects+comments": "Benefits + Side effects + Comments",
    "Side" : "Side effects",
    "Comment": "Comments"
}
uglyTextNames = {
    "Benefits": "Benefit",
    "Benefits + Side effects" : "Benefit+Side",
    "Benefits + Side effects + Comments" : "benefits+sideEffects+comments",
    "Side effects" : "Side",
    "Comments": "Comment"
}

# Function to load data from CSV files
def load_data(file_name, path): # path is either svc path or knn path 
    file_path = os.path.join(path, file_name)
    return pd.read_csv(file_path)

# Define dropdown options
dropdown_options = [{'label': prettyTextNames[dataset], 'value': dataset} for dataset in available_datasets]

# Function to plot svc graph
def create_graph(df, texts, model_name, x_axis_value):
    data = []
    for column in df.columns:
        if (column == "Bag-of-Words") or (column == "TF-IDF") or (column == "Word2Vec") or (column == "LDA"):
            x_max = df[x_axis_value][df[column].idxmax()]  # Get index of maximum value
            y_max = df[column].max()  # Maximum value
            # Line plot
            trace = go.Scatter(
                x=df[x_axis_value],
                y=df[column],
                mode='lines',
                name=f"{model_name} : {column}"
            )
            data.append(trace)
            # Dot with highest value
            trace_max = go.Scatter(
                x=[x_max],
                y=[y_max],
                mode='markers',
                marker=dict(color='red', size=10),
                showlegend=False  # Do not show in legend
            )
            data.append(trace_max)
    if (model_name == "KNN"):
        xaxis = "K-Nearest Neighbours"
    else:
        xaxis = "C"

    layout = go.Layout(
        title=f"Accuracy score for training on {prettyTextNames[texts]} for different vectorizers",
        xaxis=dict(title=xaxis),
        yaxis=dict(title="Score"),
        legend=dict(x=0, y=1)
    )
    return go.Figure(data=data, layout=layout)

# ----------------------------------------------------
#
#                   Code for tab 4
#
# ----------------------------------------------------

with open('violinVariables.pkl', 'rb') as f:
    grid_searches, configurations = pickle.load(f)

# Define your function to generate and plot the graphs
def extract_and_plot_grid_search_results(grid_searches, configurations, plot_type='violin'):
    graphs = []  # Store generated graphs here
    for grid_search, config in zip(grid_searches, configurations):
        model_name = config['model_name']
        vectorizer_name = config['vectorizer_name']
        print(f"Plotting for {model_name} using {vectorizer_name}")
        results = pd.DataFrame(grid_search.cv_results_)
        results['RMSE'] = np.sqrt(-results['mean_test_score'])
        parameters = [col for col in results.columns if col.startswith('param_')]
        num_params = len(parameters)
        if num_params == 0:
            print(f"No hyperparameters to plot for {model_name}.")
            continue
        for param in parameters:
            subset = results.dropna(subset=[param]).sort_values(by=param)
            if plot_type == 'violin':
                fig = go.Figure()
                for val in subset[param].unique():
                    fig.add_trace(go.Violin(x=subset[subset[param] == val][param], y=subset[subset[param] == val]['RMSE'], name=str(val)))
                fig.update_layout(title=f'{model_name} with {vectorizer_name} - RMSE vs. {param.split("__")[-1]}',
                                  xaxis_title=param.split("__")[-1],
                                  yaxis_title='RMSE',
                                  xaxis=dict(tickangle=45),
                                  showlegend=True)
                graphs.append(dcc.Graph(figure=fig))
    return graphs



# ----------------------------------------------------
#
#                   Dashboard Layout 
#
# ----------------------------------------------------

# Initialize Dash app
app = dash.Dash(__name__)

# Layout for tab 1 (pyLDAvis)
tab1_layout = html.Div([
    html.H1("LDA Topic Visualization Dashboard"),
    
    html.Div([
        dcc.Graph(
            id='topic-count-chart',
            figure={
                'data': [
                    go.Bar(
                        x=topic_counts['Topic'],
                        y=topic_counts['Count'],
                        marker={'color': 'skyblue'}
                    )
                ],
                'layout': go.Layout(
                    title='Number of Documents in Each LDA Topic',
                    xaxis={'title': 'Topic'},
                    yaxis={'title': 'Document Count'}
                )
            }
        )
    ]),

    html.Div([
        html.Iframe(
            id='pyldavis-iframe',
            srcDoc=pyLDAvis.prepared_data_to_html(lda_vis_data),
            width='100%',
            height='600px',
            style={'border': 'none', 'flex': '1'}
        )
    ]),
    
    # Added code: Reset button to reset pyLDAvis
    html.Button('Reset pyLDAvis', id='reset-button', n_clicks=0)
])

# Layout for tab 2 (Heatmap)
tab2_layout = html.Div([
    html.H1("Heatmap"),
    html.Div([
        dcc.Graph(
            id='topic-similarity-heatmap',
            figure={
                'data': [
                    go.Heatmap(
                        z=similarity_matrix,
                        colorscale='Viridis'
                    )
                ],
                'layout': go.Layout(
                    title='Topic-Document Similarity Heatmap',
                    xaxis={'title': 'Document'},
                    yaxis={'title': 'Topic'}
                )
            }
        )
    ])
])

# Layout for tab 3 (Model evaluation)
tab3_layout = html.Div([
    html.H1("Graphs with Texts Selection"),
    html.Div([
        dcc.Dropdown(
            options=[{'label': prettyTextNames[dataset], 'value': dataset} for dataset in available_datasets],
            value=dropdown_options[0]["value"],  # Default value
            id='textSet-dropdown'
        )
    ]),
    html.Div(id='graph-container')
])
# Layout for tab 4 (Regression evaluation)
tab4_layout = html.Div([
    html.H1("Grid Search Results Visualization"),
    html.Div([
        dcc.Dropdown(
            options=["Random Forest Regression", "Gradient Boosting Regression", "SVR"],
            value="Random Forest Regression",  # Default value
            id='violin_dropdown'
        )
    ]),
    html.Div(id="violin_container" )
])

# Tab Structure
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='LDA Topics', children=tab1_layout),
        dcc.Tab(label='Heatmap', children=tab2_layout),
        dcc.Tab(label='Model Evaluation', children=[tab3_layout]),
        dcc.Tab(label='Regression', children=[tab4_layout]),
    ])
])

# -------------
# Callback for tab 1
# update pyLDAvis based on selected topic and reset pyLDAvis
@app.callback(
    Output('pyldavis-iframe', 'srcDoc'),
    [Input('topic-count-chart', 'clickData'),
     Input('reset-button', 'n_clicks')] # reset-button
)
def update_pyldavis(clickData, n_clicks):
    ctx = dash.callback_context
    if not ctx.triggered: # No input triggered the callback
        return pyLDAvis.prepared_data_to_html(lda_vis_data)
    else:
        prop_id = ctx.triggered[0]['prop_id']
        if 'reset-button' in prop_id:
            # Reset pyLDAvis to its original prepared data
            return pyLDAvis.prepared_data_to_html(lda_vis_data)
        elif clickData is None:
            return pyLDAvis.prepared_data_to_html(lda_vis_data)
        else:
            selected_topic = clickData['points'][0]['x']
            # Filter documents by selected topic
            filtered_corpus = [doc for doc, topic in zip(corpus_bow, predominant_topics) if topic == selected_topic]
            # Prepare new LDA visualization data based on filtered corpus
            filtered_lda_vis_data = gensimvis.prepare(lda_model, filtered_corpus, dictionary)
            return pyLDAvis.prepared_data_to_html(filtered_lda_vis_data)

# -------------
# Callback for tab 3
@app.callback(
Output('graph-container', 'children'),
[Input('textSet-dropdown', 'value')]
)   
def update_graphs(text):
    # Load data based on selected vectorizer
    svc_df = load_data(f"{text}_scv.csv", svc_path)
    knn_df = load_data(f"{text}_knn.csv", knn_path)
    # Create graph for SVC
    graph_svc = create_graph(svc_df, text, "SVC", "model__C")
    # Create graph for KNN
    graph_knn = create_graph(knn_df, text, "KNN", "model__n_neighbors")
    
    return html.Div([
        dcc.Graph(id='graph-svc', figure=graph_svc),
        dcc.Graph(id='graph-knn', figure=graph_knn)
    ])

# -------------
# Callback for tab 4
@app.callback(
Output('violin_container', 'children'),
[Input('violin_dropdown', 'value')]
)   
def update_graphs(value):
    if value == "Random Forest Regression":
        return html.Div([
                html.Div(extract_and_plot_grid_search_results(grid_searches, configurations)[:6],
                          style={'display': 'grid', 'grid-template-columns': 'repeat(auto-fit, minmax(500px, 1fr))', 'grid-gap': '20px', 'max-width': '1700px', 'margin': '0 auto'})
            ])
    elif value == "Gradient Boosting Regression":
        return html.Div([
                html.Div(extract_and_plot_grid_search_results(grid_searches, configurations)[6:15],
                          style={'display': 'grid', 'grid-template-columns': 'repeat(auto-fit, minmax(500px, 1fr))', 'grid-gap': '20px', 'max-width': '1700px', 'margin': '0 auto'})
            ])
    else:
        return html.Div([
                html.Div(extract_and_plot_grid_search_results(grid_searches, configurations)[15:],
                          style={'display': 'grid', 'grid-template-columns': 'repeat(auto-fit, minmax(500px, 1fr))', 'grid-gap': '20px', 'max-width': '1700px', 'margin': '0 auto'})
            ])


if __name__ == '__main__':
    app.run_server(debug=True)
