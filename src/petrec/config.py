"""Configuration values used by recommendation scoring."""

# FEDIAF adult complete-food minimums on a dry-matter basis (g/100 g DM).
# Cat values use the 100 kcal/kg BW^0.67 table; dog values use 110 kcal/kg BW^0.75.
FEDIAF_MINIMUMS_DRY_MATTER = {
    "cat": {"protein": 25.0, "fat": 9.0},
    "dog": {"protein": 18.0, "fat": 5.5},
}
FEDIAF_MIN_PROTEIN_DRY_MATTER = FEDIAF_MINIMUMS_DRY_MATTER["dog"]["protein"]
FEDIAF_MIN_FAT_DRY_MATTER = FEDIAF_MINIMUMS_DRY_MATTER["dog"]["fat"]

# Composition score parameters.
COMPOSITION_DECAY = 0.85
COMPOSITION_NUTRIENT_WEIGHT = 0.5
