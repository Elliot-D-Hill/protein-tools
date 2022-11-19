from pathlib import Path
from typing import Iterable
from pandas import DataFrame
from Bio import Entrez
from json import load, dump


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


# TODO return type hint
def get_search_results(pdb_code: str):
    query = make_query(pdb_code)
    search_handle = Entrez.esearch(
        db="protein", term=query, idtype="acc", usehistory="y", retmax=10_000
    )
    search_results = Entrez.read(search_handle)
    search_handle.close()
    return search_results


def make_fasta_id_list(pdb_codes: Iterable) -> list:
    id_list = []
    n_codes = len(pdb_codes)
    for i, pdb_code in enumerate(pdb_codes):
        search_results = get_search_results(pdb_code)
        for id_ in search_results["IdList"]:
            id_list.append(id_)
        print(f"PDB code: {pdb_code}: {i+1} / {n_codes}")
    print(f"Search for {n_codes} proteins is complete")
    return id_list


# TODO type hints tuple[?, list]
def get_ncbi_search_results(pdb_codes: Iterable, id_list_filepath: Path) -> tuple:
    print(f"Searching for {len(pdb_codes)} PDB entries")
    if id_list_filepath.is_file():
        with open(id_list_filepath, "r") as f:
            id_list = load(f)
    else:
        id_list = make_fasta_id_list(pdb_codes)
        with open(id_list_filepath, "w") as f:
            dump(id_list, f)
    print(f"Number of IDs (chains) found: {len(id_list)}")
    search_handle = Entrez.epost(db="protein", id=",".join(map(str, id_list)))
    search_results = Entrez.read(search_handle)
    search_handle.close()
    return search_results, id_list


# TODO type hints
def fetch_fasta(search_results, id_list: list) -> str:
    id_count = len(id_list)
    print(f"Downloading {id_count} records")
    batch_size = 500
    fetch_handle = Entrez.efetch(
        db="protein",
        id=id_list,
        rettype="fasta",
        retmode="text",
        retstart=0,
        retmax=id_count * 10,
        webenv=search_results["WebEnv"],
        query_key=search_results["QueryKey"],
        idtype="acc",
        batchsize=batch_size,
    )
    data = fetch_handle.read()
    fetch_handle.close()
    return data


def get_fasta_from_ncbi_query(
    pdb_codes: Iterable,
    email: str,
    api_key: str,
    id_list_filepath: Path,
    fasta_filepath: Path,
) -> str:
    Entrez.email = email
    Entrez.api_key = api_key
    if fasta_filepath.is_file():
        with open(fasta_filepath, "r") as f:
            return load(f)
    search_results, id_list = get_ncbi_search_results(pdb_codes, id_list_filepath)
    fasta = fetch_fasta(search_results, id_list)
    with open(fasta_filepath, "w") as f:
        dump(fasta, f)
    return fasta


def collapse_on_column(df: DataFrame, column_name: str) -> DataFrame:
    """Collapse rows that differ only by specified column"""
    columns = [name for name in df.columns if name != column_name]
    return (
        df.groupby(columns)[column_name]
        .apply(lambda lst: "".join(sorted(lst)))
        .reset_index()
    )


class FastaParser:
    def __init__(self) -> None:
        self.column_names = [
            "pdb_code",
            "chain",
            "description",
            "sequence",
        ]

    def process_header(self, df: DataFrame) -> DataFrame:
        header = df["header"].str.split("|")
        df["pdb_code"] = header.str[1]
        description = header.str[2]
        df["chain"] = description.str[0]
        df["description"] = description.str.split(", ").str[-1]
        return df.drop("header", axis=1)

    def organize_dataframe(self, df: DataFrame) -> DataFrame:
        return df[self.column_names].sort_values(
            ["pdb_code", "chain"], ignore_index=True
        )

    def run_pipeline(self, df: DataFrame) -> DataFrame:
        return (
            df.pipe(self.process_header)
            .pipe(collapse_on_column, "chain")
            .pipe(self.organize_dataframe)
        )
