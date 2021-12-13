import io
import random
import base64
from flask import Flask, send_file, render_template, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns
from ReadCSV import get_game_sales_by_genre, get_game_sales_by_platform

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('landing.html')

@app.route('/byGenre')
def byGenre():
    fig, ax = plt.subplots(figsize=(11, 6))
    ax = sns.set_style(style="darkgrid")
    x = []
    y = []

    byGenre = get_game_sales_by_genre()
    for genre in byGenre:
        x.append(genre)
        y.append(byGenre[genre])

    sns.barplot(x, y)
    img=io.BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='img/png')

@app.route('/byPlatform')
def byPlatform():
    fig2, ax2 = plt.subplots(figsize=(11, 6))
    x2 = []
    y2 = []

    byPlatform = get_game_sales_by_platform()
    for platform in byPlatform:
        x2.append(platform)
        y2.append(byPlatform[platform])

    sns.barplot(x2, y2)
    img=io.BytesIO()
    fig2.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='img/png')

# @app.route('/plot.png')
# def plot_png():
#     fig = create_figure()
#     output = io.BytesIO()
#     FigureCanvas(fig).print_png(output)
#     return Response(output.getvalue(), mimetype='image/png')
#
# def create_figure():
#     fig = Figure()
#     axis = fig.add_subplot(1, 1, 1)
#     xs = range(100)
#     ys = [random.randint(1, 50) for x in xs]
#     axis.plot(xs, ys)
#     return fig

if __name__ == '__main__':
    app.run(debug=True)
