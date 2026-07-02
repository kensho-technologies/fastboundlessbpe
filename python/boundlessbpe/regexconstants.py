# Copyright 2026-present Kensho Technologies, LLC.
# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_REGEX = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

GPT4_REGEX = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"

# GPT4o regex
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py#L101-L111
# https://github.com/openai/tiktoken/blob/4560a889/tiktoken_ext/openai_public.py#L101-L114

# This regex could be made more efficient. If I was the one working on this encoding, I would
# have done a few other things differently too, e.g. I think you can allocate tokens more
# efficiently across languages.
# NOTE: See GPT4O_REGEX_PARTS below for the list form
GPT4O_SPLIT_PATTERN = "|".join(
    [
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)

# Craig's version of a pattern
FULL_PATTERN         = r"'(?i:[sdmt]|ll|ve|re)| ?[\p{Lu}]+(?=[\p{Lu}][\p{Ll}])| ?[\p{Lu}]?[\p{Ll}]+| ?[\p{Lu}]+| ?[\p{Lt}\p{Lm}\p{Lo}]+|\p{N}{1,3}(?=(?:\p{N}{3})*(?:\P{N}|$))| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+"

ULTIMATE_PATTERN_PARTS = [
r" ?(?:\p{L}\p{M}*)+['\u2019](?:\p{L}\p{M}*)+",             # contraction, allow curly apostrophe
r"_(?:\p{Ll}\p{M}*)+",                                      # snake_case, with underscore at front TODO: support __init__?
r" ?(?:\p{Lu}\p{M}*)+(?=(?:\p{Lu}\p{M}*)(?:\p{Ll}\p{M}*))", # optional space, uppercase followed by upper and then lower case letter, i.e. the XML in XMLHttpRequest
r" ?(?:\p{Lu}\p{M}*)?(?:\p{Ll}\p{M}*)+",                    # optional space, optional uppercase, one or more lowercase i.e. Http or http
r" ?(?:\p{Lu}\p{M}*)+",                                     # all uppercase acronym CONSTANT
r" ?(?:[\p{Lt}\p{Lm}\p{Lo}]\p{M}*)+",                       # titlecase, modifier, or other (those without case) letters
r"(?:\p{N}\p{M}*){1,3}(?=(?:(?:\p{N}\p{M}*){3})*(?:(?:\P{N}\p{M}*)|$))",  # numbers
r'(?:[\p{P}\p{S}]\p{M}*)+',                   # punctuation and symbols
r"[^\S\r\n]*[\n\r]+|[^\S\r\n]+",               # whitespace, Is this what there originally going for?
r"(?:[\p{Z}\p{C}]\p{M}*)+",                    # separator or control with combining marks, note that \s includes \p{Z} plus \r and \n from \p{C}, so put \p{C} after \s ones
r"\p{M}+",                                     # marks shoud be attatched to some \P{M}, just for incorrect utf-8
".+"                                          # left over, should be empty or there is a regex bug
]

ULTIMATE_PATTERN = "|".join(ULTIMATE_PATTERN_PARTS)

# is it all whitespace bytes
ALL_WHITESPACE_BYTES = rb"^\s+$"


# which bits should be allowed to be merged in a supermerge
 # TODO: is this unused?
SUPERWORD_PARTS = [
rb" ?(?:\p{L}\p{M}*)+['\u2019](?:\p{L}\p{M}*)+",             # contraction, allow curly apostrophe
rb"_(?:\p{Ll}\p{M}*)+",                                      # snake_case, with underscore at front TODO: support __init__?
rb" ?(?:\p{Lu}\p{M}*)?(?:\p{Ll}\p{M}*)+",                    # optional space, optional uppercase, one or more lowercase i.e. Http or http
rb" ?(?:\p{Lu}\p{M}*)+",                                     # all uppercase acronym CONSTANT
rb" ?(?:[\p{Lt}\p{Lm}\p{Lo}]\p{M}*)+",                       # titlecase, modifier, or other (those without case) letters
]

SUPERWORD_PATTERN = b"^(" + b"|".join(SUPERWORD_PARTS) + b")$"


# what I used for the COLM submission
# note it is a bytes regex
# and note that the final * is a bug!!!, no it isn't
ORIGINAL_MERGE_PATTERN = rb"^[ _'a-zA-Z]*[a-zA-Z][ _'a-zA-Z]*$"

# the reviwer didn't like the a-zA-Z
# to do that, we'll need to cast back to a string before use
# - match the entire string
# - lookahead ensures there is at least one letter
# - otherwise can have spaces, underscores, apostrophes, or curly apostrophes
# for reference, this is exactly what was used in the ablation training
IMPROVED_MERGE_PATTERN = r"^(?=.+\p{L})(?:\p{L}\p{M}*|[ _'\u2019])+$"

# suggestion from reviewer, which is the same if you don't have leading \p{M}
# IMPROVED_MERGE_PATTERN = r"^(?=.*\p{L})[\p{L}\p{M} _'\u2019]+$"

# Simple merge eligibility: token contains at least one Unicode letter.
# Uses search (not full match) so any letter anywhere makes the token merge-eligible.
# This works with GPT4O regex where pretokens can have a leading punctuation char
# from [^\r\n\p{L}\p{N}]? — those pretokens are still word-like and should merge.
SIMPLE_MERGE_PATTERN = r"\p{L}"

# ============================================================================
# Pretokenization patterns
# ============================================================================

# Default scripts that require character-by-character tokenization
DEFAULT_SCRIPT_SPECIFIC_SCRIPTS = [
    'Han', 'Hiragana', 'Katakana', 'Thai',
    'Myanmar', 'Khmer', 'Lao'
]

# Common regex parts shared by all tokenization patterns
# Handles emoji, punctuation, numbers, whitespace, etc.
COMMON_REGEX_PARTS = [
    # Keycap sequences (0-9, #, *) - BEFORE general emoji
    r"[0-9#*]\uFE0F?\u20E3",

    # Regional Indicator emoji pairs (flags)
    r"[\U0001F1E6-\U0001F1FF]{2}",

    # Emoji with modifiers (skin tones, variation selectors, ZWJ)
    r"\p{Extended_Pictographic}[\U0001F3FB-\U0001F3FF]?[\uFE0F\uFE0E]?(?:\u200d\p{Extended_Pictographic}[\U0001F3FB-\U0001F3FF]?[\uFE0F\uFE0E]?)*",

    # Other symbols and punctuation with optional leading space
    # limit to runs of 16
    r' ?(?:[\p{S}\p{P}]\p{M}*){1,16}',

    # Horizontal whitespace with following line endings (max 16 chars each)
    r"[^\S\r\n]{0,16}[\r\n]{1,16}",

    # Remaining whitespace (max 16 chars)
    r"[^\S\r\n]{1,16}(?![^\s\p{N}\p{C}\p{M}])",

    # Cleanup remaining whitespace
    r"\s{1,16}",

    # Numbers (1–3 digits, grouping by threes right to left)
    r"\p{N}{1,3}(?=(?:\p{N}{3})*(?:\P{N}|$))",

    # Separator, control, or combining marks
    r"[\p{Z}\p{C}\p{M}]+"
]

# Pattern parts for word-level tokenization (main/default for most scripts)
WORD_LEVEL_REGEX_PARTS = [
    # Words with lowercase and optional contractions
    r" ?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}\u200d]*[\p{Ll}\p{Lm}\p{Lo}\p{M}\u200d]+(?:['\u2019][\p{Ll}\p{Lm}\p{Lo}\p{M}]+)?",

    # Words with uppercase and optional contractions
    r" ?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}\u200d]+[\p{Ll}\p{Lm}\p{Lo}\p{M}\u200d]*(?:['\u2019][\p{Ll}\p{Lm}\p{Lo}\p{M}]+)?",
] + COMMON_REGEX_PARTS

# Joined version for convenience
WORD_LEVEL_REGEX = "|".join(WORD_LEVEL_REGEX_PARTS)

# Pattern parts for character-level tokenization (script-specific scripts like CJK)
SCRIPT_SPECIFIC_REGEX_PARTS = [
    # Single letter with optional space and combining marks
    r" ?\p{L}\p{M}*",
] + COMMON_REGEX_PARTS

# Joined version for convenience
SCRIPT_SPECIFIC_REGEX = "|".join(SCRIPT_SPECIFIC_REGEX_PARTS)

# GPT4O pattern parts as a list for reuse
GPT4O_REGEX_PARTS = [
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"\p{N}{1,3}",
    r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
    r"\s*[\r\n]+",
    r"\s+(?!\S)",
    r"\s+",
]

# Joined version for convenience
GPT4O_REGEX = "|".join(GPT4O_REGEX_PARTS)

# GPT4O pattern parts as a list for reuse
SCRIPT_SPECIFIC_GPT4O_REGEX_PARTS = [
    r" ?\p{L}\p{M}*",
    r"\p{N}{1,3}",
    r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
    r"\s*[\r\n]+",
    r"\s+(?!\S)",
    r"\s+",
]

# Joined version for convenience
SCRIPT_SPECIFIC_GPT4O_REGEX = "|".join(SCRIPT_SPECIFIC_GPT4O_REGEX_PARTS)


