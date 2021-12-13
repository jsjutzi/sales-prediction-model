import io
from flask import Flask, render_template, Response
from Create_Lr_model import print_thing
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():  # put application's code here
    fig = Figure()
    axis = fig.add_subplot(1,1,1)

if __name__ == '__main__':
    print_thing()
    app.run(debug=True)
