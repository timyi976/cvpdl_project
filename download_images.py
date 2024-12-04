import os
import requests
import re
from PIL import Image

def parse_input_file(file_path):
    pattern = re.compile(r"^(\S+)\s+(\S+)$")
    parsed_data = []

    with open(file_path, "r") as file:
        for line in file:
            match = pattern.match(line.strip())
            if match:
                folder, url = match.groups()
                parsed_data.append((folder, url))
    return parsed_data

def download_images(image_urls, save_directory):
    os.makedirs(save_directory, exist_ok=True)
    for idx, (folder_path, url) in enumerate(image_urls):
        print(f"Image {idx + 1}/{len(image_urls)}: ", end="")
        if "jpg" not in url and "png" not in url:
            print(f"Not a JPG or PNG file.")
            continue
        ext = "jpg" if "jpg" in url else "png"
        try:
            response = requests.get(url, stream=True, timeout=5)
            response.raise_for_status()
            # file_name = os.path.join(save_directory, f"image_{idx + 1}.jpg")
            file_name = os.path.join(save_directory, f"{folder_path}.{ext}")

            with open(file_name, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)

            with Image.open(file_name) as img:
                width, height = img.size
                print(f"{width}×{height}")

        except requests.exceptions.Timeout:
            print(f"Download timeout. Skipping...")

        except requests.exceptions.RequestException:
            print(f"Failed to download")
        
        except Exception:
            print(f"Unknown error")
            if os.path.exists(file_name):
                os.remove(file_name)  # Remove file if it exists
                print(f"Removed incomplete file: {file_name}")

if __name__ == "__main__":
    image_urls = parse_input_file("laion-ocr-index-url.txt")
    print(f"Total {len(image_urls)} Images")
    download_images(image_urls, "images")
