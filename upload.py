import os
import cloudinary
import cloudinary.uploader
import cloudinary.api
from dotenv import load_dotenv

DIR = "./data/samples/keyframe"
NUM = 10

def check_url_exists(url: str) -> bool:
    """Checks if a URL exists on Cloudinary."""
    cloudinary_client = cloudinary.Cloudinary()
    return cloudinary_client.exists(url)


if __name__ == "__main__":
    dotenv_path = "./.env"
    load_dotenv(dotenv_path=dotenv_path)

    cloud_name = os.environ.get("CLOUD_NAME")
    secret_key = os.environ.get("API_SECRET")
    api_key = os.environ.get("API_KEY")

    config = cloudinary.config(
        cloud_name=cloud_name, api_key=api_key, api_secret=secret_key, secure=True
    )
    
    for directory in os.listdir(os.path.join(DIR,"keyframes-{}".format(NUM))):
        sub_directory_path = os.path.join(DIR, "keyframes-{}".format(NUM), directory)
        for image in os.listdir(sub_directory_path):
            image_path = os.path.join(sub_directory_path, image)
            #try:
            cloudinary.uploader.upload_image(
                file=image_path,
                public_id=f"keyframes/{directory}/{image.split('.')[0]}",
                overwrite=False,
            )
            print(image_path)
            #except:
            #    print("Error")
