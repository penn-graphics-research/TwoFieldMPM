# -*- coding: utf-8 -*-
import os
import sys
import requests

current_path = os.path.dirname(os.path.realpath(__file__))

def download(url, filename):
    sys.stdout.write(f"Downloading {url}\n")
    with open(filename, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total/1000), 1024*1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50*downloaded/total)
                sys.stdout.write('\r[{}{}]'.format('█' * done, '.' * (50-done)))
                sys.stdout.flush()
    sys.stdout.write('\n')

def download_folder(folder_name, file_list):
    local_folder = os.path.join(current_path, "projects/data", folder_name)
    try:
        os.mkdir(local_folder)
    except:
        pass
    base_url = "https://github.com/bow-lib/Bow-data/raw/master/" + folder_name + "/"
    for filename in file_list:
        download(base_url + filename, os.path.join(local_folder, filename))

EIPC = ["Sharkey.ply", "sphere.ply"]

try:
    os.mkdir(os.path.join(current_path, "projects/data"))
except:
    pass

if len(sys.argv) == 1: # download all
    download_folder("EIPC", EIPC)
elif sys.argv[1] == 'EIPC': # download only one folder in Bow-data
    download_folder("EIPC", EIPC)


