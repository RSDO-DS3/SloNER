import os
import json


def list_dir(dirpath: str) -> (list, list):
    files, dirs = [], []
    for dpath, dnames, fnames in os.walk(dirpath,):
        files.extend(fnames)
        dirs.extend(dnames)
        break
    return sorted(dirs), sorted(files)


if __name__ == '__main__':
    _, files = list_dir('data/models/cro-slo-eng-bert-ssj500k')
    print(json.dumps(files, indent=4))
