# Imports
import streamlit as st
import pandas as pd
import pickle
from IPython.display import HTML


# pandas options
pd.set_option('display.max_colwidth', None)

# setting the basic configuration of the web app. This is shown in the Tab
st.set_page_config(page_title = "#🎶Song Recommendations" 
                    ,page_icon = ":bar_chart:" 
                    )

# read the data 
df = pd.read_csv("spotify_data.zip")
df['song_url'] = 'https://open.spotify.com/track/' + df['id']


# Opening intro text
st.write("# 🎶Song Recommender✨")
st.write("#### Choose the parameters of the song:")

# Sliders for the values
# dance
dance_value = st.slider('Set the min level of dancebility?', min_value=0.0, max_value=1.0, value=0.54, step=0.1)

# acousticness
acoustic_value = st.slider('Set the min level of acousticness?', min_value=0.0, max_value=1.0, value=0.49, step=0.1)

# energy
energy_value = st.slider('Set the min level of energy?', min_value=0.0, max_value=1.0, value=0.48, step=0.1)

# instrumentalness
instrumental_value = st.slider('Set the min level of instrumentalness?', min_value=0.0, max_value=1.0, value=0.48, step=0.1)

# popularity
popularity_value = st.slider('Set the min level of popularity?', min_value=1.0, max_value=100.0, value=33.0, step=1.0)

# year
year_value = st.slider('Set the min level of year?', min_value=1921, max_value=2020, value=1999, step=1)

# filtering the data based on above criteria
df_filtered = df.query('danceability >= @dance_value & acousticness >= @acoustic_value & energy >= @energy_value &  instrumentalness >= @instrumental_value & popularity >= @popularity_value & year >= @year_value')

# Songs available in new 
total_unique_songs = df_filtered['name'].nunique()
st.write("Unique songs in the dataset mataching the above criteria: " + str(total_unique_songs))

# get the song list
song_name_list = df_filtered['name'].unique().tolist()

# Convert the names as a drop-down
option = st.selectbox(
    'Which is your fav song?',
    song_name_list)

st.write('You selected:', option)


# pass through the prediction
selection = df.query("name == @option ").select_dtypes(['int', 'float'])

# load the model
nn_model = pickle.load(open('nn_model_prediction.sav','rb'))


# Generate the neigbors
neighbor_list = nn_model.kneighbors(selection,  return_distance=False)[:,1:].tolist()[0] # Skip the first value 


# Final list of predictions
neighbors = df.loc[neighbor_list,['name', 'song_url']]

# cleaning operations
neighbors = neighbors.reset_index(drop= True).rename(columns = {'name': 'Song', 'song_url': 'Spotify Link'})

# render the links
HTML(neighbors.to_html(render_links=True, escape=False))

# use this to make table more clean https://github.com/softhints/Pandas-Tutorials/blob/master/styling/create-clickable-link-pandas-dataframe-jupyterlab.ipynb
# show the table
# st.dataframe(neighbors)

test = neighbors

# Reference: https://discuss.streamlit.io/t/display-urls-in-dataframe-column-as-a-clickable-hyperlink/743/4
# show the links
def make_clickable(link):
    # target _blank to open new window
    # extract clickable text to display for your link
    text = link.split('=')[0]
    return f'<a target="_blank" href="{link}">{text}</a>'

st.write("#### The song🎶 recommendations💽 are: ")


# link is the column with hyperlinks
test['Spotify Link'] = test['Spotify Link'].apply(make_clickable)
test = test.to_html(escape=False)

st.write(test, unsafe_allow_html=True)
# st.table(test)

st.write("#### Get your headphones🎧 and enjoy😎")

