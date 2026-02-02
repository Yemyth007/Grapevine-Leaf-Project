from flask import Flask, render_template, request
import pyrebase
from firebase_configs import firebase_configs
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/home/cpedesign/mysite/static/detections/'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/object-counts')
def display_counts():
    selected_config_name = request.args.get('config', os.environ.get('FIREBASE_CONFIG', 'default'))
    config = firebase_configs[selected_config_name]
    firebase = pyrebase.initialize_app(config)
    db = firebase.database()

    ref = db.child('total_counts')
    latest_counts = ref.get().val()

    if not latest_counts:
        latest_counts = {}
    return render_template('counts_display.html', counts=latest_counts)

@app.route('/GUI')
def display_gui():
    selected_config_name = request.args.get('config', 'gui')
    try:
        config = firebase_configs[selected_config_name]
    except KeyError:
        return "Invalid Firebase configuration name", 400

    firebase = pyrebase.initialize_app(config)
    db = firebase.database()

    return render_template('GUI.html')

@app.route('/load_detection', methods=['POST'])
def handle_load_detection():
    config = firebase_configs['gui']
    firebase = pyrebase.initialize_app(config)
    db = firebase.database()

    db.child("load_detection").set(1)

    return "Load detection started!", 200

@app.route('/terminate', methods=['POST'])
def handle_termination():
    config = firebase_configs['gui']
    firebase = pyrebase.initialize_app(config)
    db = firebase.database()

    db.child("terminate").set(1)

    return "Program termination initiated!", 200



@app.route('/information')
def display_information():
    selected_config_name = request.args.get('config', 'gui')
    config = firebase_configs[selected_config_name]

    firebase = pyrebase.initialize_app(config)
    db = firebase.database()

    return render_template('information.html')




@app.route('/display')
def display_detections():
    selected_config_name = request.args.get('config', 'default')
    config = firebase_configs[selected_config_name]

    firebase = pyrebase.initialize_app(config)
    db = firebase.database()
    return render_template('display.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True) 
