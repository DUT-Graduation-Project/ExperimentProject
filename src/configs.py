import argparse


def get_args_parse():
    parser = argparse.ArgumentParser("Main Configuration", add_help=False)

    # General configs
    parser.add_argument("--device", type=str, default="cpu")

    # Directory configs
    parser.add_argument("--image-dir", type=str, default="./data/mapping.npy")
    parser.add_argument("--mapping-dir", type=str, default="./data/mapping.npy")
    parser.add_argument("--vid-dir", type=str, default="./data/videos/")
    parser.add_argument("--output-dir", type=str, default="./data/vectordb/")
    parser.add_argument("--object-dir", type=str, default="./data/objects/")
    parser.add_argument("--ocr-dir", type=str, default="./data/ocr/")

    # Logger configs
    parser.add_argument("--project", type=str, default="hcmai")
    parser.add_argument("--enable-wandb", type=lambda x: str(x).lower(), default=False)

    # Vectorstore configs
    parser.add_argument("--vectordb", type = str, default="faiss")
    parser.add_argument("--index-method", type=str, default="cosine")
    parser.add_argument("--feature-shape", type=int, default=768)

    # Output configs
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--ninterval", type=int, default=3)
    parser.add_argument("--n-probe", type = int, default = 100)
    parser.add_argument("--rerank", type = str, default = "alpha_qe")
    parser.add_argument("--temporal-search", type = str, default = "brute_force")

    # model configs
    parser.add_argument(
        "--od-model",
        type=str,
        default="https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1",
    )
    return parser
