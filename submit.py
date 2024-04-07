import os
import requests

SESSION_ID = os.getenv("HCMAI_SESSION_KEY")
print(SESSION_ID)

def submit(video : str, frame_idx : int):
    video = str(video)
    frame_idx = str(frame_idx)
    assert isinstance(video, str) and isinstance(frame_idx, str), "Input variables must be strings"

    URL = "https://eventretrieval.one/api/v1/submit"
    PARAMS = {
        "item" : video,
        "frame" : frame_idx,
        "session" : SESSION_ID
    }
    response = requests.get(url = URL, params = PARAMS)
    print(response.content)

if __name__ == "__main__":
    video = "L09_V007"
    frame_idx = 27054
    submit(video = video, frame_idx = frame_idx)