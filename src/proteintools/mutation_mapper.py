from pandas import DataFrame, merge

from dataclasses import dataclass, field
from typing import List, Dict
from copy import deepcopy


@dataclass
class Chain:
    chain_type: str
    sequence: str

    def apply_substitution(self, position, new_character):
        sequence_list = list(self.sequence)
        sequence_list[position] = new_character
        self.sequence = "".join(sequence_list)


@dataclass
class Protein:
    chains: Dict[str, Chain] = field(repr=False)


@dataclass
class ReferenceProtein(Protein):
    pdb_code: str


@dataclass
class Mutation:
    old_residue: str
    chain: str
    position: str
    new_residue: str


@dataclass
class VariantProtein(Protein):
    chains: Dict[str, Chain] = field(init=False)
    _id: int
    reference: ReferenceProtein
    mutations: List[Mutation] = field(repr=False)

    def __post_init__(self):
        self.chains = deepcopy(self.reference.chains)
        self.apply_mutations()

    def apply_mutations(self) -> None:
        for mutation in self.mutations:
            chain_key = next(key for key in self.chains.keys() if mutation.chain in key)
            self.chains[chain_key].apply_substitution(
                mutation.position, mutation.new_residue
            )

    def to_dict(self, chain_name, variant_chain):
        return {
            "pdb_code": self.reference.pdb_code,
            "variant_id": self._id,
            "chain_name": chain_name,
            "chain_type": variant_chain.chain_type,
            "variant_chain": variant_chain.sequence,
            "reference_chain": self.reference.chains[chain_name].sequence,
        }

    def append_dict_to_list(self, lst):
        for (chain_name, variant_chain) in self.chains.items():
            lst.append(self.to_dict(chain_name, variant_chain))
        return lst


def make_chains(group):
    chain = zip(group["chain"], group["chain_type"], group["sequence"])
    return {
        chain_name: Chain(chain_type, sequence)
        for (chain_name, chain_type, sequence) in chain
    }


def make_reference_dict(df):
    return {
        pdb_code: ReferenceProtein(make_chains(group), pdb_code)
        for pdb_code, group in df.groupby("pdb_code")
    }


def make_mutations(group):
    return [Mutation(**kwargs) for kwargs in group.to_dict(orient="records")]


def make_variant(variant_id, reference, mutation_group):
    mutations = make_mutations(mutation_group)
    return VariantProtein(variant_id, reference, mutations)


def make_variant_list(reference_df, mutation_df):
    variants = []
    references = make_reference_dict(reference_df)
    for pdb_code, pdb_group in mutation_df.groupby("pdb_code"):
        reference = references[pdb_code]
        for _id, variant_group in pdb_group.groupby("variant_id"):
            variants.append(make_variant(_id, reference, variant_group))
    return variants


class MutationMapper:
    def __init__(
        self,
    ):
        self.column_names = [
            "pdb_code",
            "variant_id",
            "chain_name",
            "chain_type",
            "mutation",
            "variant_kd",
            "reference_kd",
            "variant_chain",
            "reference_chain",
        ]

    def make_dataframe(self, variant_list):
        list_of_variant_dicts = []
        for variant in variant_list:
            list_of_variant_dicts = variant.append_dict_to_list(list_of_variant_dicts)
        return DataFrame(list_of_variant_dicts)

    def map_mutations(self, raw, references, mutations):
        merge_on = ["pdb_code", "variant_id"]
        variant_list = make_variant_list(references, mutations.set_index(merge_on))
        chains_df = self.make_dataframe(variant_list)
        print(chains_df.head(10))
        print(raw.head(10))
        return merge(
            chains_df, raw, how="left", left_on=merge_on, right_on=merge_on
        ).filter(items=self.column_names)
