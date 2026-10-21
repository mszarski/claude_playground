"""Compositional random parts, hard variant: the random_mix machinery biased to 2-4 bodies and 3-7 finishing
operations (holes / counterbores / countersinks, patterns, pockets, slots, fillets, chamfers, shells), every one
applied only where it is geometrically valid. See random_mix.py."""
from datagen.families.random_mix import generate_mix


def generate(rng):
    return generate_mix(rng, hard=True)
