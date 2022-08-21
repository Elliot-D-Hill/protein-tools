from abc import abstractmethod
from pandas import merge, concat
from numpy import select

from fastatools import clean_dataframe_header


class DatasetMaker:
    def format_dataframe(self, df):
        return df

    def filter_dataframe(self, df):
        return df

    def transform_dataframe(self, df):
        return df

    def organize_dataframe(self, df):
        return df

    def make_dataset(self, df):
        return (
            df.pipe(self.format_dataframe)
            .pipe(self.filter_dataframe)
            .pipe(self.transform_dataframe)
            .pipe(self.organize_dataframe)
        )


class FastaDatasetMaker(DatasetMaker):
    def __init__(self):
        self.column_names = [
            "pdb_code",
            "chain",
            "chain_type",
            "description",
            "sequence",
        ]

    @abstractmethod
    def assign_chain_type(self, df):
        pass

    def process_header(self, df):
        header = df["header"]
        header = header.str.split("|")
        df["pdb_code"] = header.str[1]
        description = header.str[2]
        df["chain"] = description.str[0]
        df["description"] = description.str.split(", ").str[-1]
        return df.drop("header", axis=1)

    def sort_then_join(self, lst):
        return "".join(sorted(lst))

    # collapse rows that differ only by specified column
    def collapse_rows(self, df, column_to_collapse):
        columns = [n for n in df.columns if n != column_to_collapse]
        return (
            df.groupby(columns)[column_to_collapse]
            .apply(self.sort_then_join)
            .reset_index()
        )

    def organize_dataframe(self, df):
        return df[self.column_names].sort_values(
            ["pdb_code", "chain"], ignore_index=True
        )

    def transform_dataframe(self, df):
        return (
            df.pipe(self.process_header)
            .pipe(self.collapse_rows, "chain")
            .pipe(self.assign_chain_type)
        )


class SabdabDatasetMaker(FastaDatasetMaker):
    def __init__(self, chain_df):
        self.chain_df = chain_df
        self.column_names = [
            "pdb_code",
            "chain",
            "chain_type",
            "description",
            "sequence",
        ]

    def assign_chain_type(self, df):
        def is_chain_subset(df, column):
            return [
                any([i in str(a) for i in str(b)])
                for a, b in zip(df[column], df["chain"])
            ]

        df = merge(self.chain_df, df, on="pdb_code", how="inner")
        conditions = [
            is_chain_subset(df, "heavy"),
            is_chain_subset(df, "light"),
            is_chain_subset(df, "antigen"),
        ]
        choices = [df["heavy"].name, df["light"].name, df["antigen"].name]
        df["chain_type"] = select(conditions, choices, default="antigen")
        return df


class ChainDatasetMaker(DatasetMaker):
    def format_dataframe(self, df):
        old_columns = ["pdb", "Hchain", "Lchain", "antigen_chain"]
        new_columns = ["pdb_code", "heavy", "light", "antigen"]
        columns = dict(zip(old_columns, new_columns))
        return (
            df.filter(items=old_columns)
            .assign(pdb=df["pdb"].str.upper())
            .assign(
                antigen_chain=df["antigen_chain"].str.replace(" | ", "", regex=False)
            )
            .rename(columns=columns)
        )

    def filter_dataframe(self, df):
        return df[df["heavy"] != df["light"]]

    def transform_dataframe(self, df):
        return df.groupby("pdb_code").sum()


class MutationDatasetMaker(DatasetMaker):
    def __init__(self):
        self.column_names = [
            "pdb_code",
            "variant_id",
            "chain",
            "old_residue",
            "position",
            "new_residue",
        ]

    def filter_dataframe(self, df):
        return df[["pdb_code", "mutation"]]

    def transform_dataframe(self, df):
        return (
            df.assign(mutation=df["mutation"].str.split(","))
            .explode(["mutation"], ignore_index=False)
            .pipe(self.parse_mutation)
            .pipe(self.assign_variant_id)
        )

    def parse_mutation(self, df):
        return (
            df.pipe(lambda x: x.assign(old_residue=x["mutation"].str[0]))
            .pipe(lambda x: x.assign(chain=x["mutation"].str[1]))
            .pipe(
                lambda x: x.assign(position=(x["mutation"].str[2:-1].astype(int) - 1))
            )
            .pipe(lambda x: x.assign(new_residue=x["mutation"].str[-1]))
            .reset_index()
        )

    def assign_variant_id(self, df):
        counts = [
            group["index"] - (group["index"].min() - 1)
            for _, group in df.groupby("pdb_code")
        ]
        return df.assign(variant_id=concat(counts)).drop("index", axis=1)

    def organize_dataframe(self, df):
        return df.reindex(columns=self.column_names)


class ReferenceDatasetMaker(FastaDatasetMaker):
    def __init__(self, model):
        self.model = model

    def assign_chain_type(self, df):
        conditions = [
            df["description"].str.lower().str.contains("heavy chain"),
            df["description"].str.lower().str.contains("light chain"),
        ]
        choices = ["heavy", "light"]
        df["chain_type"] = select(conditions, choices, default="unknown")
        self.model.eval()
        # FIXME make_dataloader does not exist; need to build it
        dataloader = make_dataloader()
        predictions = [self.model(text, offsets) for _, text, offsets in dataloader]
        df["chain_type"] = predictions
        return df


class SkempiDatasetMaker(DatasetMaker):
    def __init__(self):
        self.new_column_names = ["pdb_code", "mutation", "variant_kd", "reference_kd"]
        self.old_column_names = [
            "pdb",
            "mutations_cleaned",
            "affinity_mut_parsed",
            "affinity_wt_parsed",
        ]

    def format_dataframe(self, df):
        header_replacements = {" ": "_", "-": "_", "(": "", ")": "", "#": ""}
        columns = dict(zip(self.old_column_names, self.new_column_names))
        return clean_dataframe_header(df, header_replacements).rename(columns=columns)

    def filter_rows(self, df):
        antibodies = ["AB/AG", "AB/AG,Pr/PI"]
        is_antibody = df["hold_out_type"].isin(antibodies)
        return df[is_antibody]

    def filter_columns(self, df):
        return df[self.new_column_names]

    def filter_dataframe(self, df):
        return df.pipe(self.filter_rows).pipe(self.filter_columns)

    def assign_variant_id(self, df):
        groupby_pdb_code = df.groupby("pdb_code")
        groups = [
            group.assign(variant_id=range(1, group.shape[0] + 1))
            for _, group in groupby_pdb_code
        ]
        return concat(groups)

    def transform_dataframe(self, df):
        return (
            df.assign(pdb_code=df["pdb_code"].str[0:4])
            .pipe(self.assign_variant_id)
            .reset_index(drop=True)
        )

    def organize_dataframe(self, df):
        columns = self.new_column_names[:]
        columns.insert(1, "variant_id")
        return df[columns]
