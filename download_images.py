import os
import shutil
import requests
from PIL import Image
from tqdm import tqdm
from pathlib import Path

def parse_input_file(file_path):
    parsed_data = []
    with open(file_path, "r") as file, tqdm(unit=" images") as pbar:
        for line in file:
            if line.strip() == '': continue
            key, url = line.strip().split(' ', 1)
            parsed_data.append((key, url))
            pbar.update(1)
    return parsed_data

def download_image(url, save_dir: Path):
    if "jpg" not in url and "png" not in url:
        # print(f"Not a JPG or PNG file.")
        raise
    ext = "jpg" if ".jpg" in url else "png"
    response = requests.get(url, stream=True, timeout=5)
    response.raise_for_status()
    os.makedirs(save_dir, exist_ok=True)
    file_name = save_dir / f"image.{ext}"
    orig_file_name = save_dir / f"image_orig.{ext}"

    with open(orig_file_name, "wb") as file:
        for chunk in response.iter_content(1024):
            file.write(chunk)

    with Image.open(orig_file_name) as img:
        # width, height = img.size
        # print(f"{width}×{height}")
        new_img = scale_and_pad_to_512(img)
        new_img.save(file_name)

def scale_and_pad_to_512(img) -> Image.Image:
    width, height = img.size
    scale_factor = 512 / max(width, height)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    new_img = Image.new("RGB", (512, 512), (255, 255, 255))
    paste_x = (512 - new_width) // 2
    paste_y = (512 - new_height) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img

def try_download_image(url, save_dir: Path):
    try:
        download_image(url, save_dir)
        return True
    except requests.exceptions.Timeout:
        # print(f"Download timeout. Skipping...", save_dir)
        ...
    except requests.exceptions.RequestException:
        # print(f"Failed to download", save_dir)
        ...
    except Exception:
        # print(f"Unknown error", url)
        ...
    return False

def download_images(urls, save_dir: Path):
    success_images = []
    for key, url in tqdm(urls, desc=f"Download images", ncols=0, unit=" images"):
        if url == None: continue
        first, second = key.split("_")
        success = try_download_image(url, save_dir / first / second)
        if success: success_images.append(key)
    return success_images

if __name__ == "__main__":
    image_urls = parse_input_file("/nfs/nas-6.1/cvpdl_2024/laion-ocr-index-url.txt")
    image_dict = dict(image_urls)
    # key = "40530_405304116"
    # url = image_dict.get(key)
    # first, second = key.split("_")
    # save_dir = Path.cwd() / "images" / first / second
    # try_download_image(url, save_dir)
    # indeces = list(image_dict.keys())[:10]
    indeces = ["00000_000000012", "00000_000000036", "00000_000000044", "00000_000000061"]
    save_dir = Path.cwd() / "images"
    urls = [(key, image_dict.get(key)) for key in indeces]
    success_images = download_images(urls, save_dir)
    print(f"Total {len(success_images)} images")

    # TODO: move other files into image folder?
    source_folder = Path("/nfs/nas-6.1/cvpdl_2024/laion-ocr-all")
    for key in success_images:
        first, second = key.split("_")
        images_source: Path = source_folder / first / first / second
        images_dest: Path = save_dir / first / second
        for file in images_source.iterdir():
            if file.is_file(): # only files, skip directories
                shutil.copy(file, images_dest / file.name)

    output_file = Path.cwd() / "index.txt"
    with output_file.open("w") as file:
        for key in success_images:
            file.write(f"{key}\n")

