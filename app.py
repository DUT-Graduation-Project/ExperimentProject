import os
import argparse
import pandas as pd
import requests
# from flask_cors import CORS
from flask import Flask, render_template, request, send_from_directory, Response

from src.model_predictor import ModelPredictor
from src.configs import get_args_parse
import os


parser = argparse.ArgumentParser(parents=[get_args_parse()])
parser.add_argument("--faiss-name", type = str, default = "faiss_vit_l_14_openai.bin")
parser.add_argument("--model-name", type=str, default="ViT-L-14")
parser.add_argument("--pretrain-model", type = str, default = "openai")
args = parser.parse_args()

app = Flask(__name__, template_folder="templates")
# CORS(app)
predictor = ModelPredictor(args=args)

IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "submission")

search_text_global = ""

@app.route("/")
def index():
    return render_template("index.html")


# Đường dẫn tới thư mục public
public_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "public")


#Route để phục vụ tệp tĩnh
#@app.route("/images/<path:filename>")
#def serve_image(filename):
#   return send_from_directory(os.path.join(IMAGE_DIR, "images"), filename)


@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        res = Response()
        res.headers['X-Content-Type-Options'] = '*'
        return res
    
@app.route("/image-search/", methods=["GET"])
def image_search():
    image_url = request.args.get("image-path")
   
    image_lst = predictor.image_search(image_url)

    return render_template("index.html", image_lst=image_lst)




@app.route("/text-search", methods=["POST"])
def text_search():
    search_text = request.form.get("search_text", "search_text")
    search_text_global = search_text
    image_lst = predictor.text_search(search_text)
    return render_template("index.html", image_lst=image_lst, search_text=search_text)


@app.route("/submit", methods= ["POST"])
def submit():
    submit_lst = [[title.split(" ")[0], int(float(title.split(" ")[1]))] for title in request.json]
    print(submit_lst)
    submit = pd.DataFrame(submit_lst)
    submit.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index = False, header = False)
    return render_template("index.html")

@app.route("/ocr-search/", methods = ["POST"])
def ocr_search():
    url_lst = [[title.split(" ")[0], int(float(title.split(" ")[1]))] for title in request.json]
    ocr_text = request.args.get("text", "text")
    image_lst = predictor.ocr_search(ocr_text, url_lst)
    return render_template("index.html", image_lst = image_lst)

@app.route("/object-detection/", methods=["POST"])
def object_detection():
    url_lst = [[title.split(" ")[0], int(float(title.split(" ")[1]))] for title in request.json]
    objects = request.args.get("object")
    image_lst = predictor.object_detection_search(url_results = url_lst, objects = objects)
    return render_template("index.html", image_lst = image_lst)

@app.route("/segment-search/", methods = ["GET"])
def segment_search():
    image_url = request.args.get("image-path")
    image_lst = predictor.segment_search(image_url)
    return render_template("index.html", image_lst = image_lst)


@app.route("/imgs/<path:filename>")
def serve_image(filename):
    return send_from_directory(".", filename)

if __name__ == "__main__":
    app.run(debug = True)
