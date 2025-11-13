import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

##lecture du fichier
df=pd.read_csv("vesuvius_survival_dataset.csv")

##survie selon ce qu'il y a dans var
var=['DistanceFromV','WealthIndex','ShelterAccess','Status']
for i in var:
    print('Survival depending on :', i)
    df2=df.groupby(i)['Survived'].mean().reset_index()
    print(df2)
    print('-'*10, '\n')

fig = go.Figure() ##affichage d'un graphe

##figure de la survie selon l'age
fig.add_trace(go.Histogram(x=df[df['Survived']==0]['Age'], name='Not Survived', opacity=0.5))
fig.add_trace(go.Histogram(x=df[df['Survived']==1]['Age'], name='Survived', opacity=0.5))

fig.update_layout(
    title='Age Distribution by Survival',
    xaxis_title='Age',
    yaxis_title='Density',
    barmode='overlay',  
    bargap=0.1, 
)


fig.show()
