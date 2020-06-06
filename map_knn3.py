import dash
import dash_core_components as dcc
import dash_html_components as html
import os, json
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import datetime
from datetime import timedelta
from sklearn.cluster import KMeans
import seaborn as sn
import plotly.graph_objects as go

##########################################################################
##########################################################################

path_to_json = 'departements-france'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

x = datetime.datetime(2020, 3, 1)
df = pd.DataFrame()
for i in sorted(json_files):
    df_temp = gpd.read_file('departements-france/'+i)
    df_temp["date"] = x
    x+= timedelta(days=1)
    frames = [df_temp, df]
    df = pd.concat(frames)

##########################################################################
##########################################################################
df1 = pd.DataFrame(df)
df1["lon"] = df.geometry.x
df1["lat"] = df.geometry.y

DeserializableColumns = ['Population', 'Beds']

for DeserializableColumn in DeserializableColumns:

  #Normalize Json Format
  jsonDf = pd.json_normalize(df1[DeserializableColumn])

  #Adding normalised json data into Df
  df1 = df1.join(jsonDf, rsuffix='' + DeserializableColumn)

#Drop Json Data
df1 = df1.drop(DeserializableColumns, axis=1)

df1= df1.drop(['geometry', 'Emergencies', 'MedicalTests','MedicalActs'], axis=1)
df1["Confirmed"].fillna(0.0, inplace = True)
df1["Deaths"].fillna(0.0, inplace = True)
df1["Recovered"].fillna(0.0, inplace = True)
df1["Severe"].fillna(0.0, inplace = True)
df1["Critical"].fillna(0.0, inplace = True)
##########################################################################
##########################################################################


X2 = pd.DataFrame()
X2["lon"]=df1["lon"]
X2["lat"]=df1["lat"]
X2["Deaths"]=df1["Deaths"]
X2["Recovered"]=df1["Recovered"]
X2["Severe"]=df1["Severe"]
X2["Critical"]=df1["Critical"]
X2["Confirmed"]=df1["Confirmed"]
X2["Total"]=df1["Total"]
X2["Under19"]=df1["Under19"]
X2["Under39"]=df1["Under39"]
X2["Under59"]=df1["Under59"]
X2["Under74"]=df1["Under74"]
X2["Over75"]=df1["Over75"]
X2["Resuscitation"]=df1["Resuscitation"]
X2["IntensiveCare"]=df1["IntensiveCare"]
X2["TotalBeds"]=df1["TotalBeds"]
X2["date"]=df1["date"]
X2["Province/State"]=df1["Province/State"]
X2["date"]= X2["date"].apply(lambda x: x.strftime('%Y-%m-%d'))
X3 = X2.sort_values(by='date')

##########################################################################
##########################################################################
kmeans = KMeans(n_clusters = 3, init ='k-means++')
kmeans.fit(X3[X2.columns[0:7]]) # Compute k-means clustering.
X3['cluster_label'] = kmeans.fit_predict(X2[X2.columns[0:7]])
centers = kmeans.cluster_centers_ # Coordinates of cluster centers.
labels = kmeans.predict(X3[X2.columns[0:7]]) # Labels of each point



##########################################################################
##########################################################################
fig = go.Figure()
fig = px.scatter_mapbox(X3, lat="lat", lon="lon",
                        color="cluster_label", zoom=3, size="Deaths",animation_frame="date", hover_name="Province/State")
fig.update_layout(mapbox_style="stamen-terrain")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

app.run_server(debug=True, port=8040)  # Turn off reloader if inside Jupyter
