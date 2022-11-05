from pandas import DataFrame

map_mutation_input = DataFrame(
    {
        "mutation": {0: "BC10A", 1: "BC1A", 2: "BC10A,BC1A"},
        "sequence": {0: "AAAAAAAAAB", 1: "BAAA", 2: "BAAAAAAAAB"},
    }
)
map_mutation_output = DataFrame(
    {
        "mutation": {0: ["BC10A"], 1: ["BC1A"], 2: ["BC10A", "BC1A"]},
        "sequence": {0: "AAAAAAAAAB", 1: "BAAA", 2: "BAAAAAAAAB"},
        "old_residue": {0: ["B"], 1: ["B"], 2: ["B", "B"]},
        "chain": {0: ["C"], 1: ["C"], 2: ["C", "C"]},
        "position": {0: [10], 1: [1], 2: [10, 1]},
        "new_residue": {0: ["A"], 1: ["A"], 2: ["A", "A"]},
        "variant": {0: "AAAAAAAAAA", 1: "AAAA", 2: "AAAAAAAAAA"},
    }
)
