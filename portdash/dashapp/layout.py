from datetime import datetime

import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

from config import conf
from portdash import database as db

layout = html.Div(children=[
    html.Div([
        html.H4('Account(s) to display',
                style={'margin-bottom': 0, 'margin-top': 0}),
        dcc.Dropdown(id='portfolio-selector',
                     options=[{'label': n, 'value': n}
                              for n in db.get_account_names()] +
                             [{'label': 'All Accounts',
                               'value': 'All Accounts'}],
                     multi=True,
                     placeholder='Select an account',
                     value=['All Accounts']),
        ], style={'text-align': 'left'}),
    html.Div([
        html.H4('Value to chart',
                style={'margin-bottom': 0, 'margin-top': 0}),
        dcc.Dropdown(id='line-selector',
                     options=[{'label': v, 'value': k}
                              for k, v in conf('lines').items()],
                     multi=True,
                     value=['_total:1']),
        ], style={'text-align': 'left'}),
    html.Div([
        dcc.RadioItems(
            id='comparison-selector',
            className='six columns',
            options=[
                {'label': 'Sum accounts', 'value': 'sum'},
                {'label': 'Compare accounts', 'value': 'compare'},
                {'label': 'Counterfactual account', 'value': 'sim'},
            ],
            value='sum',
            labelStyle={'display': 'inline-block'},
            style={'text-align': 'left'},
        ),
        dcc.RadioItems(
            id='date-range',
            className='six columns',
            options=[
                {'label': 'Year-to-date', 'value': 'ytd'},
                {'label': 'Previous month', 'value': '1month'},
                {'label': 'Previous year', 'value': '1year'},
                {'label': 'Five years', 'value': '5year'},
                {'label': 'Maximum', 'value': 'max'},
            ],
            value='ytd',
            labelStyle={'display': 'inline-block'},
            style={'text-align': 'right'},
        ),
    ], className='row', style={'columnCount': 2}),
    html.Div(["Input symbol for counterfactual:",
              dcc.Input(id='sim-input', type='text'),
              html.Button(id='sim-submit', children='Submit', n_clicks=0)],
             id='sim-selector-container', hidden=True,
             style={'text-align': 'left'}),
    html.Hr(style={'margin-bottom': 0, 'margin-top': 0}),
    html.Div([
        html.Div([dcc.DatePickerRange(id='date-selector',
                                      start_date=datetime.today(),
                                      end_date=datetime.today(),
                                      display_format='MMM Do, YYYY'),
                  html.Div(id='value', children='Value box')],
                 className='six columns',
                 style={'text-align': 'left'}),
        html.Div(id='rate-of-return', children='RoR box',
                 className='six columns',
                 style={'text-align': 'right'})
    ],
        className='row', style={'margin-top': 0, 'columnCount': 2}),
    html.Hr(style={'margin-top': 0}),
    dcc.Graph(id='portfolio-value'),
    html.Div(id='cache', style={'display': 'none'}),
    html.P(children=f'Quotes last updated {db.get_last_quote_date()}'),
    html.P(children=f'The most recent transaction was on '
                    f'{db.get_max_trans_date()}'),
]
)
