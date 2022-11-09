from pandas import DataFrame
from Bio import Entrez

import json
from pathlib import Path


def separate_header(sequence: str) -> tuple[str]:
    """Returns a tuble containing the sequence header and the sequence itself"""
    return sequence[0], "".join(sequence[1:])


def process_fasta(file_text: str) -> tuple[str]:
    """Returns a list of tuples containing a header and sequence for each sequence in a FASTA file"""
    # TODO check if ">" is the first character of a new line
    sequences = file_text.split(">")[1:]
    sequences = [sequence.split("\n") for sequence in sequences]
    return [separate_header(sequence) for sequence in sequences]


def make_dataframe_from_fasta(fasta: str) -> DataFrame:
    """Returns a dataframe of sequences and headers from a FASTA file"""
    return DataFrame(process_fasta(fasta), columns=["header", "sequence"])


def make_query(pdb_code: str) -> str:
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


def collapse_on_column(df: DataFrame, column_name: str) -> DataFrame:
    """Collapse rows that differ only by specified column"""
    columns = [name for name in df.columns if name != column_name]
    return (
        df.groupby(columns)[column_name]
        .apply(lambda lst: "".join(sorted(lst)))
        .reset_index()
    )


class FastaParser:
    def __init__(self):
        self.column_names = [
            "pdb_code",
            "chain",
            "description",
            "sequence",
        ]

    def process_header(self, df):
        header = df["header"].str.split("|")
        df["pdb_code"] = header.str[1]
        description = header.str[2]
        df["chain"] = description.str[0]
        df["description"] = description.str.split(", ").str[-1]
        return df.drop("header", axis=1)

    def organize_dataframe(self, df):
        return df[self.column_names].sort_values(
            ["pdb_code", "chain"], ignore_index=True
        )

    def run_pipeline(self, df):
        return (
            df.pipe(self.process_header)
            .pipe(collapse_on_column, "chain")
            .pipe(self.organize_dataframe)
        )
