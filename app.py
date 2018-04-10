import time
import os
import sys
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from data.base_dataset import get_transform
from models.models import create_model
from PIL import Image
import util.util as util
from flask import Flask, send_file, request
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['png', 'jpg'])
INPUT_DIR = 'input'
OUTPUT_DIR = 'output'
OUTPUT_FILENAME = 'generated.png'
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
ARGS = '--dataroot ./mydata --model test --dataset_mode single \
    --learn_residual --resize_or_crop scale_width_and_crop \
    --name pretrained --checkpoints_dir /checkpoints\
    --fineSize 512 --loadSizeX 512'

# ARGS = '--dataroot ./mydata --model test --dataset_mode single \
#     --learn_residual --resize_or_crop scale_width_and_crop --gpu_ids -1 \
#     --fineSize 256 --loadSizeX 256'

app = Flask(__name__)
# MODEL_PATH = ... # TODO: Insert path to the model to load

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


def preprocess_data(im, trans_opt):
    transform = get_transform(trans_opt)
    transformed = transform(im)
    transformed.unsqueeze_(0)
    return {'A':transformed, 'A_paths':''}


def deblur(model, data):
    model.set_input(data)
    model.test()
    return model.get_current_visuals()

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Every incoming POST request will run the `evaluate` method
# The request method is POST (this method enables your to send arbitrary data to the endpoint in the request body, including images, JSON, encoded-data, etc.)


@app.route('/<path:path>', methods=["POST"])
def evaluate(path):
    """Preprocessing the data and evaluate the model"""
    if request.method != "POST":
        return BadRequest("POST only")
    if 'file' not in request.files:
        return BadRequest("File not present in request")
    file = request.files['file']
    if file.filename == '':
        return BadRequest("File name is not present in request")
    if not allowed_file(file.filename):
        return BadRequest("Invalid file type")

    filename = secure_filename(file.filename)
    input_filepath = os.path.join(INPUT_DIR, filename)
    file.save(input_filepath)

    img = Image.open(input_filepath).convert('RGB')
    data = preprocess_data(img, opt)
    visuals = deblur(model, data)
    util.save_image(visuals['fake_B'], OUTPUT_PATH)
    return send_file(OUTPUT_PATH, mimetype='image/png')

# Load the model and run the server
if __name__ == "__main__":
    print("* Loading model and starting Flask server...")
    print("please wait until server has fully started")
    if not os.path.isdir(INPUT_DIR):
        os.mkdir(INPUT_DIR)
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    opt = init_opts(ARGS.split())
    prepocess = get_transform(opt)
    model = load_model(opt)
    app.run(host='0.0.0.0')
