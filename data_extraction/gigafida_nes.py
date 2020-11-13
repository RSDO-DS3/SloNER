import os
import shutil
import subprocess
import pandas as pd

from xml.etree import cElementTree


def list_dir(dirpath: str) -> (list, list):
    files, dirs = [], []
    for dpath, dnames, fnames in os.walk(dirpath,):
        files.extend(fnames)
        dirs.extend(dnames)
        break
    return dirs, files


def extract_data(fname: str, dir: str) -> int:
    extract_cmd = f"tar -I zstd -xvf {fname} -C {dir}".split()
    print(f"Extracting file: {fname}")
    child = subprocess.Popen(extract_cmd, stdout=subprocess.PIPE)
    status = child.wait()
    print(f"Extracting finished with return code: {status}.")
    return status


def delete_dir(dname: str) -> bool:
    if os.path.exists(dname) and os.path.isdir(dname):
        shutil.rmtree(dname)
        return True
    else:
        return False


def extract_nes(dname: str) -> pd.DataFrame:
    return pd.DataFrame()


def process_gigafida_chunk(chunk: str) -> pd.DataFrame:
    return pd.DataFrame()


def extract_gigafida_nes(dir: str):
    dnames, fnames = list_dir(dir)
    for fname in fnames:
        chunk_name = fname.split(".")[0]
        print(f"Chunk name: {chunk_name}")
        # extract_data(fname=f"{gigafida_dir}/{fname}", dir=gigafida_dir)


if __name__ == '__main__':
    print("Extracting Gigafida's NEs")
    gigafida_dir = './data/datasets/gigafida2.1'  # relative to workdir
    extract_gigafida_nes(gigafida_dir)
