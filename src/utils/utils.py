import os
import zipfile
import requests
import socket
import yaml
from datetime import datetime


def download_from_url(url,dir):
    r = requests.get(url,allow_redirects=True).content
    filename = os.path.join(dir,url.rsplit('/', 1)[1])
    f = open(filename,'wb')
    f.write(r)
    f.close()

def unzip_file(f,dir,remove=False):
    with zipfile.ZipFile(f,'r') as zipf:
        zipf.extractall(dir)
    if remove:
        os.remove(f)


def get_hostname_and_time_string():
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    hostname = socket.gethostname()
    string = f"{current_time}_{hostname}"
    return string


def read_yaml(f):
    return yaml.safe_load(open(f, "r"))