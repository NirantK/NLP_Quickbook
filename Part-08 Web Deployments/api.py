import logging

import flask
import os
import numpy as np
from flask import Flask, jsonify, render_template, request
from scipy import misc
from sklearn.externals import joblib

app = Flask(__name__)

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(str(__name__) + ".log")
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


@app.route("/")
@app.route("/index")
def index():
    return flask.render_template("index.html", label=False)


@app.route("/status", methods=["GET"])
def get_status():
    return jsonify({"version": "0.0.1", "status": True})


@app.route("/predict", methods=["POST"])
def make_prediction():
    if request.method == "POST":
        # get uploaded file if it exists
        logger.debug(request.files)

        f = request.files["file"]
        f.save(f.filename)  # save file to disk
        logger.info(f"{f.filename} saved to disk")

        # read file from disk
        with open(f.filename, "r") as infile:
            text_content = infile.read()
        logger.info(f"Text Content from file read")

        prediction = model.predict([text_content])
        logger.info(f"prediction: {prediction}")
        prediction = "pos" if prediction[0] == 1 else "neg"
        os.remove(f.filename)
        return flask.render_template("index.html", label=prediction)


if __name__ == "__main__":
    # load ml model from disk
    model = joblib.load("model.pkl")
    # start api
    app.run(host="0.0.0.0", port=8000, debug=True)
