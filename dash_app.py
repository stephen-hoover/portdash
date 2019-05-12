import argparse
from datetime import timedelta
from functools import lru_cache
from typing import Tuple, Union

import numpy as np
import pandas as pd

from portdash.config import conf, load_config
from portdash.acemoney import create_simulated_portfolio
import portdash.database as db

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

# We need to read and set the config before defining the app layout,
# so all of the argparse code needs to be at module level.
parser = argparse.ArgumentParser(
    description="Run the Portfolio Dashboard application")
parser.add_argument('-c', '--conf', required=True,
                    help="Configuration file in YAML format")
parser.add_argument('--debug', action='store_true',
                    help="Run the Dash server in debug mode")
args = parser.parse_args()
load_config(args.conf)

app = dash.Dash(name="portfolio_dashboard")
app.title = "Portfolio Dashboard"

app.layout = html.Div(children=[
    dcc.Dropdown(id='portfolio-selector',
                 options=[{'label': n, 'value': n}
                          for n in db.get_account_names()] +
                         [{'label': 'All Accounts', 'value': 'All Accounts'}],
                 multi=True,
                 placeholder='Select an account',
                 value=['All Accounts']),
    dcc.Dropdown(id='line-selector',
                 options=[{'label': v, 'value': k}
                          for k, v in conf('lines').items()],
                 multi=True,
                 value=['_total:1']),
    html.Div(["Input symbol for counterfactual:",
              dcc.Input(id='sim-input', type='text'),
              html.Button(id='sim-submit', children='Submit', n_clicks=0)]),
    html.Div([
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
        ),
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
            style={'text-align': 'right'},
        ),
    ], className='row'),
    html.Hr(style={'margin-bottom': 0, 'margin-top': 0}),
    html.Div([
        html.Div([dcc.DatePickerRange(id='date-selector',
                                      start_date=pd.datetime.today(),
                                      end_date=pd.datetime.today(),
                                      display_format='MMM Do, YYYY'),
                  html.Div(id='value', children='Value box')],
                 className='six columns',
                 style={'text-align': 'left'}),
        html.Div(id='rate-of-return', children='RoR box',
                 className='six columns',
                 style={'text-align': 'right'})
    ],
        className='row', style={'margin-top': 0}),
    html.Hr(style={'margin-top': 0}),
    dcc.Graph(id='portfolio-value'),
    html.Div(id='cache', style={'display': 'none'}),
    html.P(children=f'Quotes last updated {db.get_last_quote_date()}'),
    html.P(children=f'The most recent transaction was on '
                    f'{db.get_max_trans_date()}'),
]
)


class NoUpdate(Exception):
    """Raise this exception to prevent component updates"""
    pass


def calc_n_days(start_time, stop_time):
    return int((stop_time - start_time).total_seconds() // 86400)


def relative_gain(account, n_days, annualize=False, ignore_zeros=False):
    """For an account, what's the return for a rolling window?

    Find the total return for an account, averaged over a given time
    period. Ignore contributions and withdrawals. Calculate the percentage
    change each day and multiply those together to find the
    change for a given time period.
    """
    change = (account['_total'].diff() -
              account['contributions'].diff() +
              account['withdrawals'].diff())
    # shift_tot: Total at the beginning of period -- use a floor to avoid nans
    shift_tot = account['_total'].shift(1)
    if ignore_zeros:
        shift_tot = shift_tot.replace(0, 1e10)
    avg_pct_change = ((1 + (change / shift_tot))
                      .rolling(f'{n_days}d').apply(np.prod))
    if annualize:
        avg_pct_change = avg_pct_change ** (365 / n_days) - 1
    else:
        avg_pct_change = avg_pct_change - 1
    avg_pct_change *= 100  # Convert fraction to percentage
    return avg_pct_change


def get_start_time(date_range: str, max_range: pd.datetime) -> pd.datetime:
    today = pd.datetime.today()
    if date_range == 'max':
        return max_range
    elif date_range == 'ytd':
        return pd.datetime(year=today.year, month=1, day=1)
    elif date_range.endswith('month'):
        val = int(date_range[:-5])
        return today - pd.Timedelta(val * 30, 'D')
    elif date_range.endswith('year'):
        val = int(date_range[:-4])
        return today - pd.Timedelta(val * 365, 'D')


def get_stop_time(date_range: str,
                  max_range: pd.datetime) -> Union[pd.datetime, None]:
    if date_range == 'max':
        # Show the day of the last withdrawal, but no further.
        return max_range + pd.Timedelta(1, 'D')
    else:
        return None


def get_nonzero_range(trace: pd.Series) -> Tuple[pd.datetime, pd.datetime]:
    trace_gt0 = trace[trace > 0.02]
    return trace_gt0.index.min(), trace_gt0.index.max()


def total_rate_of_return(account_names, start_date, stop_date, sim_symbol=None):
    return _total_ror(tuple(sorted(account_names)), start_date, stop_date, sim_symbol)


@lru_cache(1028)
def _total_ror(account_names, start_date, stop_date, sim_symbol=None):
    acct = db.sum_accounts(account_names)
    first_date, last_date = get_nonzero_range(acct['_total'])
    start_date = max([first_date, start_date])
    stop_date = max([min([last_date, stop_date]),
                     start_date + timedelta(days=1)])
    n_days = calc_n_days(start_date, stop_date)
    if sim_symbol:
        acct = create_simulated_portfolio(acct.loc[start_date:], sim_symbol)
    y = relative_gain(acct.loc[:stop_date], n_days,
                      annualize=True, ignore_zeros=True)
    return y.iloc[-1]


def calculate_traces(account_names, start_time=None, stop_time=None,
                     date_range=None):
    if not account_names:
        raise NoUpdate()
    if 'All Accounts' in account_names:
        account_names = db.get_account_names(simulated=False)
    account_names = tuple(sorted(account_names))
    all_traces, max_range, latest_time = _calculate_traces(account_names)

    start_time = pd.to_datetime(start_time)
    start_time = start_time or get_start_time(date_range, max_range)
    stop_time = pd.to_datetime(stop_time)
    stop_time = stop_time or get_stop_time(date_range, latest_time)
    return all_traces.loc[start_time:stop_time]


def calculate_sim_traces(account_names: Tuple[str], sim_symbol: str,
                         start_time=None, stop_time=None,
                         date_range=None) -> pd.DataFrame:
    """As `calculate_traces`, but return a table of traces for a portfolio
    which has all of the same purchases and withdrawals as the input accounts,
    but in which all transactions are for the `sim_symbol` security.
    """
    if not account_names or not sim_symbol:
        raise NoUpdate()
    if 'All Accounts' in account_names:
        account_names = db.get_account_names(simulated=False)

    # First get the real account, and cut it to the desired time window.
    account_names = tuple(sorted(account_names))
    _, max_range, latest_time = _calculate_traces(account_names)
    acct = db.sum_accounts(account_names)

    start_time = pd.to_datetime(start_time)
    start_time = start_time or get_start_time(date_range, max_range)
    stop_time = pd.to_datetime(stop_time)
    stop_time = stop_time or get_stop_time(date_range, latest_time)
    acct = acct[start_time:stop_time]

    # Now convert the read portfolio to a simulated portfolio
    sim_acct = create_simulated_portfolio(acct, sim_symbol=sim_symbol)
    all_traces, _, _ = _portfolio_to_traces(sim_acct)

    return all_traces.loc[start_time:stop_time]


@lru_cache(20)
def _calculate_traces(account_names: Tuple[str]):
    acct = db.sum_accounts(account_names)
    return _portfolio_to_traces(acct)


def _portfolio_to_traces(acct: pd.DataFrame):
    all_traces = pd.DataFrame(index=acct.index)
    max_range = all_traces.index.max()
    latest_time = all_traces.index.min()
    for line in conf('lines').keys():
        if line.startswith('netcont:'):
            y = acct['contributions'] - acct['withdrawals']
        elif line.startswith('netgain:'):
            y = acct['_total'] - acct['contributions'] + acct['withdrawals']
        elif line in ['28:6', '365:7']:
            y = relative_gain(acct, int(line[:-2]), annualize=False)
        elif line in ['730:8']:
            y = relative_gain(acct, int(line[:-2]), annualize=True)
        else:
            y = acct[line.split(':')[0]]
        all_traces[line.split(':')[0]] = y

        y_gt0 = y[y > 0]
        if len(y_gt0) > 0:
            max_range = min((y_gt0.index.min() - timedelta(days=1), max_range))
            if line.split(':')[0] == '_total':
                latest_time = max((y_gt0.index.max(), latest_time))
    return all_traces, max_range, latest_time


def _standardize_lines(lines):
    lines = sorted(lines, key=lambda x: x.split(':')[-1])
    return lines


def min_max_date(acct_names, date_range, graph_relayout):
    all_traces = calculate_traces(acct_names, date_range=date_range)

    if 'xaxis.range[0]' in (graph_relayout or {}):
        min_date = pd.to_datetime(graph_relayout['xaxis.range[0]'].split()[0])
        max_date = pd.to_datetime(graph_relayout['xaxis.range[1]'].split()[0])
    elif 'yaxis.range[0]' in (graph_relayout or {}):
        # Raise an exception to keep the existing value in the component
        raise NoUpdate()
    else:
        min_date = all_traces.index.min()
        max_date = all_traces.index.max()
    return min_date, max_date


@app.callback(dash.dependencies.Output('portfolio-value', 'relayoutData'),
              [dash.dependencies.Input('date-range', 'value')])
def reset_zoom(date_range):
    """Clear manual zooms on the graph if users select a new date range"""
    return None


@app.callback(dash.dependencies.Output('date-selector', 'start_date'),
              [dash.dependencies.Input('date-range', 'value'),
               dash.dependencies.Input('portfolio-value', 'relayoutData')],
              [dash.dependencies.State('portfolio-selector', 'value')])
def update_date_sel_min(date_rng, graph_rng, acct_names):
    md, _ = min_max_date(acct_names, date_rng, graph_rng)
    return md


@app.callback(dash.dependencies.Output('date-selector', 'end_date'),
              [dash.dependencies.Input('date-range', 'value'),
               dash.dependencies.Input('portfolio-value', 'relayoutData')],
              [dash.dependencies.State('portfolio-selector', 'value')])
def update_date_sel_max(date_rng, graph_rng, acct_names):
    _, md = min_max_date(acct_names, date_rng, graph_rng)
    return md


@app.callback(dash.dependencies.Output('value', 'children'),
              [dash.dependencies.Input('portfolio-selector', 'value'),
               dash.dependencies.Input('date-selector', 'start_date'),
               dash.dependencies.Input('date-selector', 'end_date')])
def update_value_box(acct_names, min_date, max_date):
    if not acct_names:
        return "..."
    min_date, max_date = pd.to_datetime(min_date), pd.to_datetime(max_date)

    all_traces = calculate_traces(acct_names, min_date, max_date)

    min_val = np.round(all_traces.loc[min_date, '_total'], 2)
    max_val = np.round(all_traces.loc[max_date, '_total'], 2)

    change = np.round(max_val - min_val, 2)
    sign = "+" if change > 0 else "-"

    chg_color = "green" if change > 0 else "red"
    components = [
        html.Span(f"${min_val:,.2f} \u2192 ${max_val:,.2f} ("),
        html.Span(f"{sign}${abs(change):,.2f}", style={'color': chg_color}),
    ]
    if min_val > 0:
        # Calculate an annualized change iff we have a finite percent change
        pct_change = np.abs(change / min_val)
        n_years = (max_date - min_date).total_seconds() / (365 * 86400)
        ann_chg = np.power(1 + pct_change, 1 / n_years) - 1
        components.extend([
            html.Span(", "),
            html.Span(f"{sign}{ann_chg:.2%}", style={'color': chg_color}),
            html.Span(" annualized"),
        ])
    components.append(html.Span(")"))
    return components


def ror_component(acct_names, min_date, max_date, name=None, sim_symbol=None):
    ror = total_rate_of_return(acct_names, min_date, max_date, sim_symbol)
    db_t = np.log(2) / np.log(1 + ror / 100)
    txt_color = "green" if ror > 0 else "red"
    name_span = [html.Span(f"{name}: ")] if name else []
    return html.Div(name_span +
                    [html.Span(f"{ror:.2f}%", style={'color': txt_color}),
                     html.Span(f" (Doubling time: {db_t:.1f} years)")])


@app.callback(dash.dependencies.Output('rate-of-return', 'children'),
              [dash.dependencies.Input('portfolio-selector', 'value'),
               dash.dependencies.Input('comparison-selector', 'value'),
               dash.dependencies.Input('date-selector', 'start_date'),
               dash.dependencies.Input('date-selector', 'end_date'),
               dash.dependencies.Input('sim-submit', 'n_clicks'),
               ],
              [dash.dependencies.State('sim-input', 'value')])
def update_ror_box(acct_names, comp_type, min_date, max_date, n_clicks, sim_symbol):
    if not acct_names:
        return "..."
    min_date, max_date = pd.to_datetime(min_date), pd.to_datetime(max_date)

    components = [html.H4("Annualized rate of return: ")]
    if comp_type == 'compare':
        for name in acct_names:
            components.append(ror_component([name], min_date, max_date, name))
    elif comp_type == 'sim':
        components.append(ror_component(acct_names, min_date, max_date, 'Actual portfolio'))
        components.append(ror_component(acct_names, min_date, max_date, 'Counterfactual portfolio', sim_symbol))
    else:
        components.append(ror_component(acct_names, min_date, max_date))

    return components


def get_layout(axes_used, lines):
    kwargs = {'xaxis': {'title': 'Date'}}
    ynames = ['yaxis2', 'yaxis']
    if '$' in axes_used:
        kwargs[ynames.pop()] = {'title': 'Dollars', 'gridcolor': '#bdbdbd'}
    if 'pct' in axes_used:
        which_y = ynames.pop()
        kwargs[which_y] = {'title': 'Percent'}
        if which_y == 'yaxis2':
            kwargs[which_y].update({'side': 'right', 'overlaying': 'y',
                                    'tickfont': {'color': 'green'},
                                    'titlefont': {'color': 'green'},
                                    #'gridcolor': 'purple'
                                   })
    if len(lines) == 1:
        kwargs['title'] = lines[0]
    return go.Layout(**kwargs)


@app.callback(dash.dependencies.Output('line-selector', 'multi'),
              [dash.dependencies.Input('comparison-selector', 'value')])
def set_line_selector_multi(comp_type):
    if comp_type in ['compare', 'sim']:
        return False
    else:
        return True


@app.callback(dash.dependencies.Output('portfolio-value', 'figure'),
              [dash.dependencies.Input('portfolio-selector', 'value'),
               dash.dependencies.Input('line-selector', 'value'),
               dash.dependencies.Input('comparison-selector', 'value'),
               dash.dependencies.Input('date-selector', 'start_date'),
               dash.dependencies.Input('date-selector', 'end_date'),
               dash.dependencies.Input('sim-submit', 'n_clicks'),
               ],
              [dash.dependencies.State('sim-input', 'value')])
def make_graph(account_names, lines, comp_type, min_date, max_date, n_clicks, sim_symbol):
    lines = np.atleast_1d(lines)
    if comp_type == 'compare':
        return _make_comparison_graph(account_names, lines, min_date, max_date)
    elif comp_type == 'sim':
        return _make_counterfactual_graph(account_names, sim_symbol, lines,
                                          min_date, max_date)
    else:
        return _make_summed_graph(account_names, lines, min_date, max_date)


def _make_summed_graph(account_names, lines, min_date, max_date):
    if not account_names:
        return None
    all_traces = calculate_traces(account_names, min_date, max_date)

    lines = _standardize_lines(lines)
    plots = []
    axes_used = {conf('which_axis')[l] for l in lines}
    lines_on_y2 = list(filter(lambda x: conf('which_axis')[x] == 'pct',
                              conf('which_axis')))
    for line in lines:
        if line in lines_on_y2 and len(axes_used) > 1:
            which_y = 'y2'
        else:
            which_y = 'y1'
        y = all_traces[line.split(':')[0]]
        plots.append(go.Scatter(x=y.index, y=y, mode='lines',
                                name=conf('lines')[line], yaxis=which_y))
    return {
        'data': plots,
        'layout': get_layout(axes_used, [conf('lines')[l] for l in lines]),
    }


def _make_counterfactual_graph(account_names, sim_symbol, lines,
                               min_date, max_date):
    if not account_names:
        return None

    all_traces = calculate_traces(account_names, min_date, max_date)
    line = _standardize_lines(lines)[0]
    plots = []
    y = all_traces[line.split(':')[0]]
    plots.append(go.Scatter(x=y.index, y=y, mode='lines',
                            name='Actual Portfolio', yaxis='y1'))

    sim_traces = calculate_sim_traces(account_names, sim_symbol, min_date, max_date)
    y_sim = sim_traces[line.split(':')[0]]
    plots.append(go.Scatter(x=y_sim.index, y=y_sim, mode='lines',
                            name='Counterfactual Portfolio', yaxis='y1'))

    return {
        'data': plots,
        'layout': get_layout(conf('which_axis')[line], [conf('lines')[line]]),
    }


def _make_comparison_graph(account_names, lines, min_date, max_date):
    if not account_names:
        return None

    traces = {n: calculate_traces([n], min_date, max_date)
              for n in account_names}
    line = _standardize_lines(lines)[0]
    plots = []
    for name, trace in traces.items():
        y = trace[line.split(':')[0]]
        plots.append(go.Scatter(x=y.index, y=y, mode='lines',
                                name=name, yaxis='y1'))
    return {
        'data': plots,
        'layout': get_layout(conf('which_axis')[line], [conf('lines')[line]]),
    }


if __name__ == '__main__':
    app.run_server(debug=args.debug)
