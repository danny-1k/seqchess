from urllib.request import urlopen
from bs4 import BeautifulSoup
from utils import download_from_url, unzip_file
import os

from tqdm import tqdm

#TODO: Add multi-threading to speed the fuck out of this shit


class Downloader:
    """Downloader class to scrape download links from https://www.pgnmentor.com.
        Downloads and unzips the files.
    """
    def __init__(self, num, save_dir) -> None:
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
            except:
                pass
        self.num = num
        self.save_dir = save_dir
        self.links = self.get_download_links()


    def get_download_links(self):
        links = []

        base_url = 'https://www.pgnmentor.com/'
        url = base_url + 'files.html'

        page = urlopen(url).read()
        soup = BeautifulSoup(page, 'html.parser')
        download_links = list(map((lambda x: x['href']), soup.find_all(
            'a', {'class': 'view'}, string='Download')))
        
        i = 0
        while i<self.num or i >= len(download_links):        
            url = base_url+download_links[i]
            if url.rsplit('/', 1)[1] not in os.listdir(self.save_dir):
                links.append(url)
                i += 1
            else:
                continue
            
        print(f"Gotten {len(links)} download links.")

        return links
    
    def download_files(self):
        for link in tqdm(self.links):
            try:
                download_from_url(link, self.save_dir)
            except:
                print(f"Could not download from {link}... Moving to next.")

    def unzip_files(self):
        for f in tqdm(os.listdir(self.save_dir)):
            if f.endswith(".zip"):
                f = os.path.join(self.save_dir, f)
                unzip_file(f, self.save_dir, remove=True)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--num", type=int, default=362)
    parser.add_argument("--save_dir", type=str, default="../data/raw")

    args = parser.parse_args()

    num = args.num
    save_dir = args.save_dir

    downloader = Downloader(num=num, save_dir=save_dir)
    print("Started downloading...")
    downloader.download_files()
    downloader.unzip_files()