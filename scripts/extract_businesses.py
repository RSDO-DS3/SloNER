import pandas as pd
import csv

from xml.etree import ElementTree


def build_entry(tag):
    entry = {}
    entry["popolnoIme"] = tag.find("PopolnoIme").text
    entry["oblika"] = tag.find("Oblika").text
    entry["organ"] = tag.find("Organ").text
    entry["maticnaStevilka"] = tag.attrib['ma']
    address = tag.find("N")
    entry["upravnaEnota"] = address.find("UpravnaEnota").text
    entry["ulica"] = address.find("Ulica").text
    try:
        entry["hisnaStevilka"] = address.attrib["hs"]
    except:
        entry["hisnaStevilka"] = None
        print("Can't extract attribute `hs`")
    entry["naselje"] = address.find("Naselje").text
    entry["posta"] = address.find("Posta").text
    try:
        entry["postnaStevilka"] = address.attrib["po"]
    except:
        entry["postnaStevilka"] = None
        print("Can't extract attribute `po`")
    entry["obcina"] = address.find("Obcina").text
    entry["regija"] = address.find("Regija").text
    return entry


if __name__ == '__main__':
    tree = ElementTree.parse('../data/Prs.xml')
    root = tree.getroot()
    data = []
    for business in root.findall("PS"):
        try:
            data.append(build_entry(business))
        except:
            print(f"Failed to parse {ElementTree.dump(business)}")
    df = pd.DataFrame(data)
    df.to_csv("../data/businesses.csv", index=False, quoting=csv.QUOTE_ALL)
