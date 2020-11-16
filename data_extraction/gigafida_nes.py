import os
import sys
import shutil
import subprocess
import pandas as pd
import time

from tqdm import tqdm
from bs4 import BeautifulSoup


def list_dir(dirpath: str) -> (list, list):
    files, dirs = [], []
    for dpath, dnames, fnames in os.walk(dirpath,):
        files.extend(fnames)
        dirs.extend(dnames)
        break
    return sorted(dirs), sorted(files)


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
    nes = []
    with open(dname) as f:
        data = BeautifulSoup(f, 'lxml')
        segs = data.find_all("seg", {"type": "name"})
        for seg in segs:
            for i, w in enumerate(seg.find_all("w")):
                nes.append({
                    "word": w.getText(),
                    "lemma": w["lemma"],
                    "msd": w["msd"],
                    "type": f'{"B" if i == 0 else "I"}-{seg["subtype"]}'
                })
    return pd.DataFrame(nes)


def process_gigafida_chunk(in_dir: str, out_dir: str) -> None:
    _, files = list_dir(in_dir)
    for file in tqdm(files):
        o_file = f"{out_dir}/{file.split('.')[0]}.csv"
        if os.path.exists(o_file):
            continue
        file_nes = extract_nes(f"{in_dir}/{file}")
        file_nes.to_csv(o_file, index=False)


def extract_gigafida_nes(in_dir: str, out_dir: str):
    dnames, fnames = list_dir(in_dir)
    start_time = time.time()
    for fname in tqdm(fnames):
        chunk_time = time.time()
        chunk_name = fname.split(".")[0]
        chunk_dir = f"{in_dir}/{chunk_name}"
        print(f"Processing chunk: {chunk_name}")

        # skip chunk if NEs already extracted
        # assumes that all xmls are extracted
        _, ne_files = list_dir(f'{out_dir}/{chunk_name}')
        if ne_files:
            continue

        # extract if not already extracted
        extract_time = time.time()
        if not (os.path.exists(chunk_dir) and os.path.isdir(chunk_dir)):
            extract_data(fname=f"{gigafida_dir}/{fname}", dir=gigafida_dir)
        print(f"Finished extracting in: {time.time() - extract_time:.3f}")

        # create chunk output directory
        chunk_out_dir = f"{out_dir}/{chunk_name}"
        if not os.path.exists(chunk_out_dir):
            os.mkdir(chunk_out_dir)

        # process the chunk
        process_time = time.time()
        process_gigafida_chunk(f"{in_dir}/{chunk_name}", chunk_out_dir)
        print(f"Finished processing in: {time.time() - process_time:.3f}")

        # delete the extracted files
        del_time = time.time()
        delete_dir(chunk_dir)
        print(f"Finished deleting in: {time.time() - del_time:.3f}")
        print(f"Finished processing chunk {chunk_name} in: {time.time() - chunk_time:.3f}")
    print(f"Finished all processing in: {time.time() - start_time:.3f}")


def combine_chunk_csvs(in_dir: str):
    dnames, _ = list_dir(in_dir)
    start_time = time.time()
    # all_data = pd.DataFrame()
    for dname in tqdm(dnames):
        print(f"Merging chunk {dname}")
        dpath = f"{in_dir}/{dname}"
        _, files = list_dir(dpath)
        chunk_all = pd.DataFrame()
        chunk_csv = {
            "per": pd.DataFrame(),
            "deriv-per": pd.DataFrame(),
            "org": pd.DataFrame(),
            "loc": pd.DataFrame(),
            "misc": pd.DataFrame(),
        }
        for fname in files:
            f_path = f"{dpath}/{fname}"
            try:
                data = pd.read_csv(f_path)
                for ne_type in chunk_csv.keys():
                    ne = data.loc[(data["type"] == f"B-{ne_type}") | (data["type"] == f"I-{ne_type}")]
                    chunk_csv[ne_type] = pd.concat([chunk_csv[ne_type], ne], ignore_index=True)
                chunk_all = pd.concat([chunk_all, data], ignore_index=True)
            except Exception as e:
                # usually this is an empty file
                if "No columns" not in str(e):
                    print(f"Error: {str(e)}", file=sys.stderr)
        for ne_type, ne_data in chunk_csv.items():
            ne_data.to_csv(f"{dpath}-{ne_type}.csv", index=False)
        chunk_all.to_csv(f"{dpath}.csv", index=False)
    print(f"Finished merging CSVs in {time.time() - start_time:.3f}")
    # all_data.to_csv(f"{in_dir}.csv", index=False)


if __name__ == '__main__':
    print("Extracting Gigafida's NEs")
    gigafida_dir = './data/datasets/gigafida2.1'  # relative to workdir
    gigafida_out_dir = './data/ne/gigafida/'
    # extract_gigafida_nes(gigafida_dir, gigafida_out_dir)
    combine_chunk_csvs(gigafida_out_dir)

