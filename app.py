from flask import Flask, render_template
from ReadCSV import print_thing

app = Flask(__name__)

@app.route('/')
def index():  # put application's code here
    return render_template('landing.html')

if __name__ == '__main__':
    print_thing()
    app.run(debug=True)
