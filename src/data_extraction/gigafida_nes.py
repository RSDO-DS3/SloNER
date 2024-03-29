import os
import sys
import shutil
import subprocess
import pandas as pd
import time

from tqdm import tqdm
from bs4 import BeautifulSoup
from lxml import etree
from itertools import groupby

parser = etree.XMLParser()

def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


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
    # Efficient XML parsing as per: https://www.ibm.com/developerworks/xml/library/x-hiperfparse/
    nes = []
    context = etree.iterparse(fr"{dname}", events=("end",), tag="{http://www.tei-c.org/ns/1.0}seg")
    for action, seg in context:
        subtype = seg.attrib['subtype']
        word = ""
        lemma = ""
        msd = ""
        for w in seg:
            if w.tag != "{http://www.tei-c.org/ns/1.0}w":
                continue
            word += f" {w.text}"
            lemma += f" {w.attrib['lemma']}"
            msd += f" {w.attrib['ana'].split(':')[1]}"
        seg.clear()
        for ancestor in seg.xpath('ancestor-or-self::*'):
            while ancestor.getprevious() is not None:
                del ancestor.getparent()[0]
        nes.append({
            "word": word.strip(),
            "lemma": lemma.strip(),
            "msd": msd.strip(),
            "type": subtype.strip()
        })
    del context
    fdata = pd.DataFrame(nes)
    if fdata.isnull().sum().sum():
        print(f"THERE IS A NULL IN {dname}")
    return fdata


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

        # delete the extracted file_names
        del_time = time.time()
        delete_dir(chunk_dir)
        print(f"Finished deleting in: {time.time() - del_time:.3f}")
        print(f"Finished processing chunk {chunk_name} in: {time.time() - chunk_time:.3f}")
    print(f"Finished all processing in: {time.time() - start_time:.3f}")


def combine_data(path: str, file_names: list) -> dict:
    chunk_csv = {
        "per": pd.DataFrame(),
        "deriv-per": pd.DataFrame(),
        "org": pd.DataFrame(),
        "loc": pd.DataFrame(),
        "misc": pd.DataFrame(),
    }
    for fname in file_names:
        f_path = f"{path}/{fname}"
        try:
            data = pd.read_csv(f_path)
            for ne_type in chunk_csv.keys():
                ne = data.loc[(data["type"] == f"B-{ne_type}") | (data["type"] == f"I-{ne_type}")]
                chunk_csv[ne_type] = pd.concat([chunk_csv[ne_type], ne], ignore_index=True)
        except Exception as e:
            # usually this is an empty file
            if "No columns" not in str(e):
                print(f"Error: {str(e)}", file=sys.stderr)
    for ne_type, ne_data in chunk_csv.items():
        ne_data.to_csv(f"{path}-{ne_type}.csv", index=False)
    return chunk_csv


def combine_chunk_csvs(in_dir: str):
    dnames, _ = list_dir(in_dir)
    start_time = time.time()
    # all_data = pd.DataFrame()
    for dname in tqdm(dnames):
        print(f"Merging chunk {dname}")
        dpath = f"{in_dir}/{dname}"
        _, files = list_dir(dpath)
        chunk_csv = combine_data(dpath, files)
        chunk_all = pd.DataFrame()
        for _, data in chunk_csv.items():
            chunk_all = pd.concat([chunk_all, data])
        chunk_all.to_csv(f"{dpath}.csv", index=False)
    print(f"Finished merging CSVs in {time.time() - start_time:.3f}")
    # all_data.to_csv(f"{in_dir}.csv", index=False)


def merge_rows(data: pd.DataFrame, indices: list) -> pd.Series:
    # print(f"Merging rows: {indices}")
    entity = []
    lemma = []
    msd = []
    type = []
    for i in indices:
        entity.append(str(data["word"].iloc[i]))
        lemma.append(str(data["lemma"].iloc[i]))
        msd.append(data["msd"].iloc[i])
        type.append(data["type"].iloc[i])
    entity = " ".join(entity)
    lemma = " ".join(lemma)
    type = type[0]
    # print(f"Named entity: {entity}")
    # print(f"Lemma: {lemma}")
    # print(f"Type: {type}")
    msd = msd[0] if all_equal(msd) else ";".join(msd)
    # print(f"Msd: {msd}, {all_equal(msd)}")
    return pd.Series({"word": entity, "lemma": lemma, "msd": msd, "type": type})


def merge_nes(in_dir: str):
    _, fnames = list_dir(in_dir)
    fnames = [fname for fname in fnames if "-" not in fname]
    for fname in tqdm(fnames):
        fpath = f"{in_dir}/{fname}"
        print(f"Working on: {fpath}")
        merged_path = ".".join(fpath.split(".")[:-1]) + "-merged.csv"
        if os.path.exists(merged_path) and os.path.isfile(merged_path):
            # file already exists
            continue
        data = pd.read_csv(fpath)
        merged_data = []
        merge_indices = []
        merged_entities = 0
        print(f"Shape of data: {data.shape}")
        for i in tqdm(range(len(data))):
            row = data.iloc[i]
            row_type = row["type"]
            row["type"] = row["type"][2:]
            if row_type[:2] == "I-":
                if merge_indices:
                    merge_indices.append(i)
                else:
                    merge_indices = [i - 1, i]
                continue
            if merge_indices:
                prev_row = merge_rows(data, merge_indices)
                merged_data[-1] = prev_row
                merged_entities += 1
                merge_indices = []
            merged_data.append(row)
        print(f"Number of merged entities: {merged_entities}")
        print(f"Size of orig {data.shape[0]},\nSize of merged: {len(merged_data)}")
        diff = data.shape[0] - len(merged_data) == merged_entities
        print(f"Diff is: {diff}")
        print(f"Writing merged data to: {merged_path}")
        pd.DataFrame(merged_data).to_csv(merged_path, index=False)


if __name__ == '__main__':
    print("Extracting Gigafida's NEs")
    gigafida_dir = './data/datasets/gigafida2.1'  # relative to workdir
    gigafida_out_dir = './data/ne/gigafida/'
    extract_gigafida_nes(gigafida_dir, gigafida_out_dir)
    # combine_chunk_csvs(gigafida_out_dir)
    # merge_nes(gigafida_out_dir)

