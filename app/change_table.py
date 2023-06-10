import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import plotly.offline as pyo
import sqlite3
from flask import Flask, render_template, g
import sqlite3
import json


app = Flask(__name__)

#for connecting to databse
def connect_db():
    return sqlite3.connect('database.db')

@app.before_request
def before_request():
    g.db = connect_db()

@app.teardown_request
def teardown_request(exception):
    db = getattr(g, 'db', None)
    if db is not None:
        db.close()


@app.route('/country/<country_id>/<year_id>/')
def year_country(country_id,year_id):
    cur = g.db.cursor()
    cur.execute("SELECT * FROM suicides WHERE country=? AND year=?",(country_id, year_id,))
    temp_data = data.loc[(data.country == country_id) & (data.year == year_id)]
    suma = temp_data['suicides_no'].sum()


    country_to_fetch = temp_data.country.unique()[0]
    fig11 = fig1(country_to_fetch, year_id)
    fig22 = fig2(country_to_fetch, year_id)

    graphJSON1 = pyo.plot(fig11, output_type='div')
    graphJSON2 = pyo.plot(fig22, output_type='div')

    return render_template('page.html', vsetko = cur.fetchall(),suma=suma,
        graphJSON1=graphJSON1,graphJSON2=graphJSON2)


@app.route('/country/<random_id>/')
def country(random_id):
    cursor = g.db.cursor()
    cursor.execute("SELECT DISTINCT year,country FROM suicides WHERE country_id=? ORDER BY year ASC",(random_id,))
    country_to_fetch = data.loc[data.country_id == int(random_id)].country.unique()[0]
    fig11 = fig1(country_to_fetch)
    fig22 = fig2(country_to_fetch)
    fig44 = fig4(country_to_fetch)
    fig55 = fig5(country_to_fetch)
    graphJSON1 = pyo.plot(fig11, output_type='div')
    graphJSON2 = pyo.plot(fig22, output_type='div')
    graphJSON3 = pyo.plot(fig44, output_type='div')
    graphJSON4 = pyo.plot(fig55, output_type='div')

    return render_template('country.html', years=cursor.fetchall(), graphJSON1=graphJSON1,
        graphJSON2=graphJSON2,graphJSON3=graphJSON3,graphJSON4=graphJSON4)


def fig1(country = None,year=None):
    title = "Number of suicides "
    data_to_push = [go.Line(y=suicide_rate['suicides_no'], x=suicide_rate['year'])]
    if(country != None and year == None):
        country_rate = data.loc[data['country'] == country][['year','suicides_no']].groupby(['year']).sum().reset_index()
        title += f'per year in {country}'
        data_to_push = [go.Bar(y=country_rate['suicides_no'], x=country_rate['year'])]
    elif(year != None):
        country_rate = data.loc[(data.country == country) & (data.year == year)][['age','suicides_no']].groupby(['age']).sum().reset_index()
        title += f'by age in {country} in {year}'
        data_to_push = [go.Bar(y=country_rate['suicides_no'], x=country_rate['age'])]

    fig = go.Figure(
    data=data_to_push,
    layout=go.Layout(
        title={
            'text':title,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        yaxis=dict(
            title=title
            )
        )
    )
    return fig

def fig2(country = None,year=None):
    data_to_push = sex_suicide
    add = ""
    if(country != None and year == None):
        add += f'in {country}'
        data_to_push = data.loc[data['country'] == country][['gender', 'suicides_no', 'age']].groupby(['gender', 'age']).sum().reset_index()
    elif(year != None):
        add += f'in {country} in {year}'
        data_to_push = data.loc[(data['country'] == country) & (data.year == year)][['gender', 'suicides_no', 'age']].groupby(['gender', 'age']).sum().reset_index()
    
    genders = go.Pie(labels=data_to_push['gender'], values=data_to_push['suicides_no'])
    male = go.Pie(labels=data_to_push['age'][data_to_push['gender']=='male'], 
                values=data_to_push['suicides_no'][data_to_push['gender']=='male'])
    female = go.Pie(labels=data_to_push['age'][data_to_push['gender']=='female'], 
                values=data_to_push['suicides_no'][data_to_push['gender']=='female'])

    fig = go.Figure()

    fig.add_trace(genders)
    fig.add_trace(male)
    fig.add_trace(female)

    fig.data[0].visible = True
    fig.data[1].visible = False
    fig.data[2].visible = False


    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                buttons=[
                    dict(
                        label="All",
                        method="update",
                        args=[{"visible": [True, False, False]}]
                    ),
                    dict(
                        label="Male",
                        method="update",
                        args=[{"visible": [False, True, False]}]
                    ),
                    dict(
                        label="Female",
                        method="update",
                        args=[{"visible": [False, False, True]}]
                    )
                ],
                direction="down",
                pad={"r": 1, "t": 1},
                showactive=True,
                x=0,
                xanchor="left",
                y=1.2,
                yanchor="top"
        )],
        title={
            'text': "Suicide number by gender and age "+add,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    fig.update_traces(marker_colors=['setosa','virginica'])
    return fig

def fig3():
    choropleth_map = go.Figure()
    years = mean_vals['year'].unique()
    years.sort()

    visible = [False] * len(years)
    visible[0] = True

    for i, year in enumerate(years):
        trace = go.Choropleth(
            locations=mean_vals.loc[mean_vals['year'] == year].country,
            locationmode='country names',
            colorscale='Portland',
            z=mean_vals.loc[mean_vals['year'] == year].suicides_rate,
            colorbar={'title': 'Suicide number (higher=worse)'},
            marker={
                'line': {
                    'color': 'rgb(255,255,255)',
                    'width': 2
                }
            },
            visible=visible[i]
        )
        choropleth_map.add_trace(trace)

    dropdown_menu = []

    for i, year in enumerate(years):
        dropdown_menu.append(
            dict(
                args=[{'visible': [y == year for y in years]}],
                label=str(year),
                method='update'
            )
        )

    choropleth_map.update_layout(
        title_text='Number of suicides per 100k for years',
        updatemenus=[
            dict(
                active=0,
                buttons=dropdown_menu,
                direction='down',
                showactive=True,
                x=0,
                y=1
            )
        ],
        geo={
            'projection':{
              'type':'orthographic'  # default is 'equirectangular'
            },
            'scope': 'world'
        }
    )
    return choropleth_map

def fig4(country0):
    dunno = mean_vals.loc[mean_vals['country'] == country0]
    fig = sp.make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=dunno['year'], y=dunno['GdpPerCapita'],
        name="GdpPerCapita"), secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=dunno['year'], y=dunno['suicides_rate'], name="Suicide rate"),
        secondary_y=True,
    )

    fig.update_layout(title={
        'text':"Suicides rate comapred to GdpPerCapita in "+country0,
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}
    )


    fig.update_xaxes(title_text="year")


    fig.update_yaxes(
        title_text="<b>primary</b> Suicide_rate/100k", 
        secondary_y=False)
    fig.update_yaxes(
        title_text="<b>secondary</b> GdpPerCapita", 
        secondary_y=True)
    return fig

def fig5(country):
    temp_data = data.loc[(data.country == country)][['suicides_no','year','GdpPerCapita','gender']].groupby(['year','gender']).agg({'suicides_no':'sum','GdpPerCapita':'first'}).reset_index()
    fig = px.scatter(temp_data, x="GdpPerCapita", y="suicides_no", color="gender", trendline="ols")
    fig.update_layout(
        title={
            'text': "Correlation of suicides and GDP with respect to genders",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    return fig

@app.route('/')
def home():
    cursor = g.db.cursor()
    cursor.execute("SELECT DISTINCT country,country_id FROM suicides")

    fig11 = fig1()
    fig22 = fig2()
    fig33 = fig3()
    graphJSON1 = pyo.plot(fig11, output_type='div')
    graphJSON2 = pyo.plot(fig22, output_type='div')
    graphJSON3 = pyo.plot(fig33, output_type='div')
    return render_template('main.html', countries=cursor.fetchall(),graphJSON1=graphJSON1,
    graphJSON2=graphJSON2,graphJSON3=graphJSON3,best = best_of_all,second=second)


#loading and preprocessing data
data = pd.read_csv("master.csv").drop(['HDI for year','country-year'],axis=1)
data =data.dropna(subset=['suicides_no'])
data['all_population'] = data.groupby(['country','year'])['population'].transform(sum)
data['all_population'] = np.where(data.year > 2016, data.population, data.all_population)
data.rename(columns={'sex':'gender',' gdp_for_year ($) ':'GdpForYear','gdp_per_capita ($)':'GdpPerCapita'},inplace=True)
country_to_id = {}

for country in data['country'].unique():
    if country not in country_to_id:
        country_to_id[country] = len(country_to_id) + 1

data['country_id'] = data['country'].map(country_to_id)
data.year = data.year.astype('str')

#creation of auxiliary tables to reduce time
suicide_rate = data[['year', 'suicides_no']].groupby(['year']).sum().reset_index()
sex_suicide = data[['gender', 'suicides_no', 'age']].groupby(['gender', 'age']).sum().reset_index()
mean_vals = data[['country', 'year', 'suicides_no', 'all_population','GdpPerCapita']].groupby(['country', 'year']).agg({'suicides_no': 'sum', 'all_population': 'first', 'GdpPerCapita' : 'first'}).reset_index()
mean_vals['suicides_rate'] = mean_vals['suicides_no'] / (mean_vals['all_population']/100000)
best_of_all = mean_vals.loc[mean_vals.suicides_no == mean_vals.suicides_no.max()][['country','suicides_no','year']].to_numpy()
second = data.loc[data.suicides_no == data.suicides_no.max()][['gender','suicides_no','year','country']].to_numpy()

#storing of processed and updated data to new csv file
data.to_csv('~/project/master0.csv')

#storing data to database
connection = connect_db()
data.to_sql('suicides', connection, if_exists='replace', index=False)
connection.commit()
connection.close()