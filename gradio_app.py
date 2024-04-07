import argparse
import gradio as gr
from PIL import Image

import utils
from src.configs import get_args_parse
from src.model_predictor import ModelPredictor

class GradioApp:
    def __init__(self, args):
        self.args = args
        self._initialize_predictor()

    def _initialize_predictor(self):
        self.predictor = ModelPredictor(
            args= args,
        )

    @utils.fcall
    def show_imgs(self, text_query):
        imgs_path = self.predictor.text_search(text_query)

        #imgs = [Image.open(img_path) for img_path in imgs_path]

        return imgs_path
    
def show_gradio_ui(args):

    process = GradioApp(args)

    with gr.Blocks() as DBP_Block:
        with gr.Row():
            text_query = gr.Textbox(label = "Input Query", scale = 3, lines= 5)
            
        with gr.Row():
            submit_btn = gr.Button("Submit")

        with gr.Row():
            output = gr.Gallery(label = "Answer", columns= args.columns)

        # Activities
        submit_btn.click(fn = process.show_imgs, 
                         inputs= [text_query], 
                         outputs= output, api_name = "DBProcess",)

    DBP_Block.launch(share = args.share)

    return DBP_Block

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Main Configuration", parents =[get_args_parse()])
    parser.add_argument("--model-name", type=str, default="ViT-B/32")
    parser.add_argument("--share", type = bool, default = False)

    # UI configs
    parser.add_argument("--columns", type = int, default = 4)

    args = parser.parse_args()

    utils.setup_logging()

    show_gradio_ui(args)