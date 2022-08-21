from pandas import DataFrame
from Bio import Entrez

import json
from pathlib import Path

from utils import separate_header


# returns a list of tuples containing a header and sequence for each sequence in a FASTA file
def process_fasta(file_text):
    sequences = file_text.split(">")[1:]
    sequences = [sequence.split("\n") for sequence in sequences]
    return [separate_header(sequence) for sequence in sequences]


# returns a dataframe of sequences and headers from a FASTA file
def make_dataframe_from_fasta(fasta):
    return DataFrame(process_fasta(fasta), columns=["header", "sequence"])


def clean_dataframe_header(df, replacements):
    df.columns = df.columns.str.strip().str.lower()
    for key, value in replacements.items():
        df.columns = df.columns.str.replace(key, value, regex=False)
    return df


def make_query(pdb_code):
    return f"{pdb_code}[All Fields] AND pdb[filter]"


def get_search_results(pdb_code):
    query = make_query(pdb_code)
    search_handle = Entrez.esearch(
        db="protein", term=query, idtype="acc", usehistory="y", retmax=50
    )
    search_results = Entrez.read(search_handle)
    search_handle.close()
    return search_results


def make_fasta_id_list(pdb_codes):
    id_list = []
    n_codes = len(pdb_codes)
    for i, pdb_code in enumerate(pdb_codes):
        search_results = get_search_results(pdb_code)
        for _id in search_results["IdList"]:
            id_list.append(_id)
        print(f"PDB code: {pdb_code}: {i+1} / {n_codes}")
    print(f"Search for {n_codes} proteins is complete")
    return id_list


def get_fasta_id_list(pdb_codes):
    print(f"Searching for {len(pdb_codes)} PDB entries")
    id_list_path = Path("Data/Cache/")
    id_list_path.mkdir(parents=True, exist_ok=True)
    id_list_filepath = id_list_path / Path("fasta_id_list_cache.txt")
    if not id_list_filepath.is_file():
        id_list = make_fasta_id_list(pdb_codes)
        with open(id_list_filepath, "w") as f:
            f.write(json.dumps(id_list))
        return id_list
    with open(id_list_filepath, "r") as f:
        return json.loads(f.read())


def get_ncbi_search_results(pdb_codes):
    id_list = get_fasta_id_list(pdb_codes)
    print(f"Number of IDs (chains) found: {len(id_list)}")
    search_handle = Entrez.epost(db="protein", id=",".join(map(str, id_list)))
    search_results = Entrez.read(search_handle, validate=True)
    search_handle.close()
    return search_results, id_list


def fetch_fasta(search_results, id_count):
    fetch_handle = Entrez.efetch(
        db="protein",
        rettype="fasta",
        retmode="text",
        retstart=0,
        retmax=id_count,
        webenv=search_results["WebEnv"],
        query_key=search_results["QueryKey"],
        idtype="acc",
    )
    data = fetch_handle.read()
    fetch_handle.close()
    return data


def get_fasta_from_ncbi_query(pdb_codes, email, api_key):
    Entrez.email = email
    Entrez.api_key = api_key
    list_path = Path("Data/Cache/")
    list_path.mkdir(parents=True, exist_ok=True)
    list_filepath = list_path / Path("fasta_cache.fa")
    if not list_filepath.is_file():
        search_results, id_list = get_ncbi_search_results(pdb_codes)
        id_count = len(id_list)
        print(f"Downloading {id_count} records")
        fasta = fetch_fasta(search_results, id_count)
        with open(list_filepath, "w") as f:
            f.write(json.dumps(fasta))
        return fasta
    with open(list_filepath, "r") as f:
        return json.loads(f.read())


def write_fasta_from_ncbi_query(filepath, pdb_codes, email, api_key):
    data = get_fasta_from_ncbi_query(pdb_codes, email, api_key)
    with open(filepath, "w") as out_handle:
        out_handle.write(data)


def save_data(df, path, filename, postfix):
    path.mkdir(parents=True, exist_ok=True)
    prefix = filename.split(".")[0]
    df.to_csv(f"{path}/{prefix}_{postfix}.csv", index=False)
