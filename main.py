import joblib
import torch.nn as nn
import torch

# my model
from detection import Detection
from ultralytics import YOLO


# servidor
from flask import Flask, request, render_template, redirect, url_for, Response, jsonify, send_from_directory
import base64



D = Detection(vector_path="./models/vector.gz")
D.load_state_dict(torch.load('./models/control-calidad.pt'))


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/zbdhfioñsd-erghie5lidebvhulueis', methods=['POST', 'GET', 'PUT', 'PATCH'])
def procesar():
    if not request.method =="POST":
        return "ok"
    
    img = request.files.get('file')
    res = D.response(img.read(), byte=True)
    img , datos = res['img'], res['datos']
    img = base64.b64encode(img).decode('utf-8')
    labels = list(map(lambda x: x[0], datos))
    confs = list(map(lambda x: f"{x[1]:.4f}", datos))

    return jsonify({
        'label':" ".join(labels),
        'confidence':" ".join(confs),
        'recomendation':('-'),
        'image_base64':img
    })

@app.route('/zbdhfioñsd-erghie5lidebvhulueis/sdfrgbesdfbgdes<ar>')
def descarga(ar):
    if ar == "dataset_test":
        return send_from_directory(
            directory ="./datasets/",
            path="dataset_test.csv",
            as_attachment=True
        )
    elif ar == "dataset_train":
        return send_from_directory(
            directory ="./datasets/",
            path="dataset_train.csv",
            as_attachment=True
        )
    elif ar == "model":
        return send_from_directory(
            directory ="./models/",
            path="control-calidad.pt",
            as_attachment=True
        )
    elif ar == "encoder":
        return send_from_directory(
            directory ="./models/",
            path="vector.gz",
            as_attachment=True
        )


app.run(host="0.0.0.0")