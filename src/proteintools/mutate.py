from pandas import DataFrame, Series
from typing import Callable


def split_mutations(df: DataFrame) -> DataFrame:
    return df.assign(mutation=df["mutation"].str.split(","))


def parse_mutation(df: DataFrame, parser: Callable) -> Series:
    return df["mutation"].apply(
        lambda mutations: [parser(mutation) for mutation in mutations]
    )


def assign_mutations(df: DataFrame) -> DataFrame:
    return df.assign(
        old_residue=parse_mutation(df, lambda mutation: mutation[0]),
        chain=parse_mutation(df, lambda mutation: mutation[1]),
        position=parse_mutation(df, lambda mutation: int(mutation[2:-1])),
        new_residue=parse_mutation(df, lambda mutation: mutation[-1]),
    )


def mutate(sequence: str, positions: list[int], new_characters: list[str]) -> str:
    sequence_list = list(sequence)
    for position, new_character in zip(positions, new_characters):
        sequence_list[int(position) - 1] = new_character
    return "".join(sequence_list)


def apply_mutation(df: DataFrame) -> DataFrame:
    return df.assign(
        variant=df.apply(
            lambda x: mutate(x["sequence"], x["position"], x["new_residue"]), axis=1
        )
    )


def map_mutations(df: DataFrame) -> DataFrame:
    return df.pipe(split_mutations).pipe(assign_mutations).pipe(apply_mutation)
