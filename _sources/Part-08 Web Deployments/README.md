# Deploying a ML model with Flask API

Based on Scikit-Learn stack

---

In this tutorial,

1. We build a small text classification model, and write it to disk in `model_train.py`
2. Reuse this model, in `model_predict.py`
3. Expose the model using Flask with `api.py`

Using Flask to create an API, we can deploy this model and create a simple web page to load and classify new movie reviews.

## To run locally

- Install pip and Python 3
- Clone this repository
- Navigate to the working directory
- Install the Python dependencies `pip install -r requirements.txt`
- Run the API `python api.py`
- Open a web browser and go to `http://localhost:8000`