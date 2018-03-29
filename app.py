import time
import os
import sys
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from PIL import Image
import util.util as util
from flask import Flask, send_file, request

OUTPUT_PATH = 'deblurred.png'
ARGS = '--dataroot ./mydata --model test --dataset_mode single \
    --learn_residual --resize_or_crop scale_width --gpu_ids -1 \
    --fineSize 360 --loadSizeX 360'

app = Flask(__name__)
#MODEL_PATH = ... # TODO: Insert path to the model to load

# def load_model(MODEL_PATH):
#     """Load the model"""
#     # TODO: INSERT CODE

# def data_preprocessing(data):
#     """Preprocess the request data to transform it to a format that the model understands"""
#     # TODO: INSERT CODE

def init_opts(args):
    opt = TestOptions().parse(args)
    return opt

def load_model(opt):
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    return create_model(opt)

def preprocess_data(opt):
    data_loader = CreateDataLoader(opt)
    return data_loader.load_data()

def deblur(model, dataset):
    for data in dataset:
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        #util.save_image(visuals['real_A'], 'blurred.png')
        util.save_image(visuals['fake_B'], OUTPUT_PATH)

# Every incoming POST request will run the `evaluate` method
# The request method is POST (this method enables your to send arbitrary data to the endpoint in the request body, including images, JSON, encoded-data, etc.)
@app.route('/<path:path>', methods=["POST"])
def evaluate(path):
    """Preprocessing the data and evaluate the model"""
    if request.method == "POST":
        # CODE FOR DATA PREPROCESSING
        dataset = preprocess_data(opt)
        deblur(model, dataset)
        # TODO: CODE FOR EVALUATION
        # TODO: RETURN THE PREDICTION
    return path 

# Load the model and run the server
if __name__ == "__main__":
    print("* Loading model and starting Flask server...")
    print("please wait until server has fully started")
    opt = init_opts(ARGS.split())
    model = load_model(opt)
    app.run(host='0.0.0.0')