# HCM-AI-2023

## Dataset
- The data directory is in the following format:
```
    - data
        |
        - - keyframe
        |      |
        |      - - keyframes-{N}
        |              |
        |              - - video
        |                    |
        |                    - - images.png
        |
        | - clip-features
        |
        | - map-keyframes
        |
        | - objects
        |
        - - vectordb 
        |       |
        |       - - faiss_vectordb
        |       |
        |       - - mapping.json
        |
        - - weights
        |
        | - mapping.npy
    
```
## Quickstart
- Prepare the environment:
```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```
- Build the vector databse:
```bash
python src/clip_database.py
```
- Deploy the application (Gradio):
```bash
python src/gradio_app.py --share False
```