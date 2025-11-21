import os
import requests

def download_file(url, filename):
    if os.path.exists(filename):
        print(f"{filename} already exists, skipping download.")
        return

    print(f"Downloading {filename}...")
    r = requests.get(url, stream=True)
    with open(filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Downloaded: {filename}")

def download_all():
    github_base = "https://github.com/abdelac/image-classifier-saas/releases/download/v1.0"

    download_file(f"{github_base}/modeli.h5", "modeli.h5")
    download_file(f"{github_base}/gbdt_model.pkl", "gbdt_model.pkl")

if __name__ == "__main__":
    download_all()
