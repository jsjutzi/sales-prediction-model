from flask import Flask, render_template
from Create_Lr_Model import run_stuff

app = Flask(__name__)

@app.route('/')
def index():
    run_stuff()
    return render_template('landing.html')


if __name__ == '__main__':
    app.run(debug=True)
