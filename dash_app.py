from datetime import timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from portdash import config
import portdash.database as db

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go


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
                          for k, v in config.LINES.items()],
                 multi=True,
                 value=['_total:1']),
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


def get_start_time(date_range, max_range):
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


def get_stop_time(date_range, max_range):
    if date_range == 'max':
        # Show the day of the last withdrawal, but no further.
        return max_range + pd.Timedelta(1, 'D')
    else:
        return None


def get_nonzero_range(trace):
    trace_gt0 = trace[trace > 0.02]
    return trace_gt0.index.min(), trace_gt0.index.max()


def total_rate_of_return(account_names, start_date, stop_date):
    return _total_ror(tuple(sorted(account_names)), start_date, stop_date)


@lru_cache(1028)
def _total_ror(account_names, start_date, stop_date):
    acct = db.sum_accounts(account_names)
    first_date, last_date = get_nonzero_range(acct['_total'])
    start_date = max([first_date, start_date])
    stop_date = max([min([last_date, stop_date]),
                     start_date + timedelta(days=1)])
    n_days = calc_n_days(start_date, stop_date)
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


@lru_cache(20)
def _calculate_traces(account_names):
    acct = db.sum_accounts(account_names)
    all_traces = pd.DataFrame(index=acct.index)
    max_range = all_traces.index.max()
    latest_time = all_traces.index.min()
    for line in config.LINES.keys():
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

    pct_change = (np.abs(change / min_val) *
                  365 * 86400 / (max_date - min_date).total_seconds())

    chg_color = "green" if change > 0 else "red"
    components = [
        html.Span(f"${min_val:,.2f} \u2192 ${max_val:,.2f} ("),
        html.Span(f"{sign}${abs(change):,.2f}", style={'color': chg_color}),
        html.Span(", "),
        html.Span(f"{sign}{pct_change:.2%}", style={'color': chg_color}),
        html.Span(" annualized)"),
    ]
    return components


def ror_component(acct_names, min_date, max_date, name=None):
    ror = total_rate_of_return(acct_names, min_date, max_date)
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
               dash.dependencies.Input('date-selector', 'end_date')])
def update_ror_box(acct_names, comp_type, min_date, max_date):
    if not acct_names:
        return "..."
    min_date, max_date = pd.to_datetime(min_date), pd.to_datetime(max_date)

    components = [html.H4("Annualized rate of return: ")]
    if comp_type == 'compare':
        for name in acct_names:
            components.append(ror_component([name], min_date, max_date, name))
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
    if comp_type == 'compare':
        return False
    else:
        return True


@app.callback(dash.dependencies.Output('portfolio-value', 'figure'),
              [dash.dependencies.Input('portfolio-selector', 'value'),
               dash.dependencies.Input('line-selector', 'value'),
               dash.dependencies.Input('comparison-selector', 'value'),
               dash.dependencies.Input('date-selector', 'start_date'),
               dash.dependencies.Input('date-selector', 'end_date')])
def make_graph(account_names, lines, comp_type, min_date, max_date):
    lines = np.atleast_1d(lines)
    if comp_type == 'compare':
        return _make_comparison_graph(account_names, lines, min_date, max_date)
    else:
        return _make_summed_graph(account_names, lines, min_date, max_date)


def _make_summed_graph(account_names, lines, min_date, max_date):
    if not account_names:
        return None
    all_traces = calculate_traces(account_names, min_date, max_date)

    lines = _standardize_lines(lines)
    plots = []
    axes_used = {config.WHICH_AXIS[l] for l in lines}
    lines_on_y2 = list(filter(lambda x: config.WHICH_AXIS[x] == 'pct',
                              config.WHICH_AXIS))
    for line in lines:
        if line in lines_on_y2 and len(axes_used) > 1:
            which_y = 'y2'
        else:
            which_y = 'y1'
        y = all_traces[line.split(':')[0]]
        plots.append(go.Scatter(x=y.index, y=y, mode='lines',
                                name=config.LINES[line], yaxis=which_y))
    return {
        'data': plots,
        'layout': get_layout(axes_used, [config.LINES[l] for l in lines]),
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
        'layout': get_layout(config.WHICH_AXIS[line], [config.LINES[line]]),
    }


if __name__ == '__main__':
    app.run_server(debug=True)
