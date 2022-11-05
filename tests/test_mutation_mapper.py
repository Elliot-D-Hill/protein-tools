from pandas.tests import assert_frame_equal

import cases
from proteintools.mutate import map_mutations


def test_map_mutations():
    mapped_mutations = map_mutations(cases.map_mutation_input)
    assert_frame_equal(cases.map_mutation_output, mapped_mutations)
