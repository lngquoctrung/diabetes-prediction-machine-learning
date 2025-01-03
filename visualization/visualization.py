from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import numpy as np

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = dbc.Container([
    
]);


if __name__ == "__main__":
    app.run_server(debug=True, port=8080);