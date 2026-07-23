# Copyright 2026-present Kensho Technologies, LLC.
"""Tests for the Pretokenizer class."""

import regex
import pytest
from boundlessbpe.pretokenize import Pretokenizer
from boundlessbpe.regexconstants import (
    WORD_LEVEL_REGEX, SCRIPT_SPECIFIC_REGEX, DEFAULT_SCRIPT_SPECIFIC_SCRIPTS,
    GPT4O_EXPORT_REGEX, GPT4O_COARSE_REGEX,
)


# Test cases: (input_string, expected_tokens)
PRETOKENIZE_TESTS = [
    # Letters : \p{L}
    ('Hello world!', ['Hello', ' world', '!']),
    ('Hello мир 你好 world!',['Hello', ' мир', ' 你', '好', ' world', '!']),
    ('Helloмир你好world!',['Hello', 'мир', '你', '好', 'world', '!']),  # split on scripts
    ('café   мир',['café', '  ', ' мир']),
    ('caféмир',['café', 'мир']),  # split on scripts

    # Asian-script
    ('漢字カタカナ ABC', ['漢','字','カ','タ','カ','ナ', ' ABC']),
    # with optional space
    (' 漢字カタカナ ABC', [' 漢','字','カ','タ','カ','ナ', ' ABC']),
    # Chinese with Myanmar character at end
    ("漢字カ,タカナไทยမြန်မာ", ['漢', '字', 'カ', ',', 'タ', 'カ', 'ナ', 'ไ', 'ท', 'ย', 'မြ', 'န်', 'မာ']),
    # Khmer with combining marks
    ('ភាសាខ្មែរ', ['ភា', 'សា', 'ខ្', 'មែ', 'រ']),

    # Lao with tone marks that are Lo don't join
    ('ພາສາລາວ', ['ພ', 'າ', 'ສ', 'າ', 'ລ', 'າ', 'ວ']),

    # Arabic with diacritics (right-to-left)
    ('مَرْحَبًا', ['مَرْحَبًا']),  # Should keep as one word with marks

    # Hebrew with niqqud (vowel points)
    ('שָׁלוֹם', ['שָׁלוֹם']),  # Should keep as one word

    # Mixed RTL and LTR
    ('Hello שלום world', ['Hello', ' שלום', ' world']),

    # Multiple combining marks on one base
    ('e\u0301\u0308', ['e\u0301\u0308']),  # e with acute and diaeresis

    # join-control Devanagari conjunct
    ("क्\u200dष",["क्\u200dष"]),
    # join-control extended conjunct
    ("क्\u200dष\u200dत्र", ["क्\u200dष\u200dत्र"]),
    # Latin ZWJ sequence is rare, so we'll split
    ("a\u200db", ['a\u200db']),
    # ZWNJ sequence - creates a boundary, should split
    ("स\u200cल", ["स", "\u200c", "ल"]),
    # Contraction, with curly quote, not at end
    ("we're can't you've cows' it's It's", ["we're"," can't"," you've", " cows", "'", " it's", " It's"]),
    # snake_case now handled in supermerges
    ('_init _a_b x_y_z', ['_', 'init', ' _', 'a', '_', 'b', ' x', '_', 'y', '_', 'z']),
    # Mixed_case_VAR
    (' Mixed_case_VAR', [' Mixed', '_', 'case', '_', 'VAR']),
    # fancy CamelCaps, simplified version
    ('XMLHttpRequest', ['XMLHttp', 'Request']),
    # fancy CamelCaps leading space, simplified version
    (' XMLHttpRequest', [' XMLHttp', 'Request']),
    # words
    ('Http http HttpRequest', ['Http', ' http', ' Http', 'Request']),
    # all uppercase
    ('fooNASA CPU GPU', ['foo', 'NASA',' CPU',' GPU']),
    # this now relies on the fallback non-latin case
    ('ǅǄ Łódź Ωmega midΩmega', ['ǅǄ', ' Łódź', ' Ω', 'mega', ' mid', 'Ω', 'mega']),

    # Multiple apostrophes/contractions
    ("'twas", ["'", "twas"]),  # Don't match at start
    ("twas'", ["twas", "'"]),  # or at end
    ("'", ["'"]),  # Lone apostrophe

    # Hyphenated words
    ('well-known', ['well', '-', 'known']),  # Should split

    # Titlecase digraphs
    ('ǲ', ['ǲ']),  # Latin Small Letter Dz

    # Ligatures
    ('ﬁ', ['ﬁ']),  # fi ligature

    # Combining enclosing marks
    ('a⃝', ['a⃝']),  # a with combining enclosing circle

    # Numbers
    (' 1 ', [' ', '1', ' ']),
    (' 12 ', [' ', '12', ' ']),
    (' 123 ', [' ', '123', ' ']),
    (' 1234 ', [' ', '1', '234', ' ']),
    (' 12345 ', [' ', '12', '345', ' ']),
    (' 123456 ', [' ', '123', '456', ' ']),
    (' 1234567 ', [' ', '1', '234', '567', ' ']),
    (' 1', [' ', '1']),
    (' 12', [' ', '12']),
    (' 123', [' ', '123']),
    (' 1234', [' ', '1', '234']),
    (' 12345', [' ', '12', '345']),
    (' 123456', [' ', '123', '456']),
    (' 1234567', [' ', '1', '234', '567']),
    ('1 ', ['1', ' ']),
    ('12 ', ['12', ' ']),
    ('123 ', ['123', ' ']),
    ('1234 ', ['1', '234', ' ']),
    ('12345 ', ['12', '345', ' ']),
    ('123456 ', ['123', '456', ' ']),
    ('1234567 ', ['1', '234', '567', ' ']),
    ('1', ['1']),
    ('12', ['12']),
    ('123', ['123']),
    ('1234', ['1', '234']),
    ('12345', ['12', '345']),
    ('123456', ['123', '456']),
    ('1234567', ['1', '234', '567']),
    ('1234 567 89012', ['1', '234', ' ', '567', ' ', '89', '012']),

    # Devanagari digits
    ('१२३४', ['१', '२३४']),

    # Myanmar digits
    ('၁၂၃၄', ['၁', '၂၃၄']),

    # Mixed Common and Myanmar digits
    ('12၃၄56', ['12၃', '၄56']),

    # Whitespace
    ('        123', ['        ', '123']),
    ('\u00a0 \u200e', ['\u00a0 ', '\u200e']),
    ('        . Hello', ['       ', ' .', ' Hello']),
    ('        Hello', ['       ', ' Hello']),
    ('        $10', ['       ', ' $', '10']),
    ('  \t \n\n  ', ['  \t \n\n', '  ']),
    ('Hello     world    \n\n \n ', ['Hello', '    ', ' world', '    \n\n', ' \n', ' ']),

    # Multiple spaces edge case (>16 chars)
    ('a' + ' ' * 20 + 'b', ['a', ' ' * 16, ' ' * 3, ' b']),

    # Tab variations
    ('\t\t\t', ['\t\t\t']),

    # Mixed whitespace
    (' \t\n', [' \t\n']),

    # Punctuation and symbols
    (', a,b !?.!?', [',',' a',',','b', ' !?.!?']),
    ('~$A', ['~$','A']),
    ('$10+$12==$22', ['$', '10', '+$', '12','==$','22']),

    # Ellipsis variations
    ('...', ['...']),
    ('…', ['…']),

    # Control/Mark
    ('\u0301\u0300', ['\u0301\u0300']),

    # Emoji
    ('Á👍🏽👩‍👩‍👧‍👦🇺🇳🏳️‍🌈', ['Á', '👍🏽', '👩\u200d👩\u200d👧\u200d👦', '🇺🇳', '🏳️\u200d🌈']),
    ("🇺🇸🇬🇧", ['🇺🇸', '🇬🇧']),
    ('1️⃣2️⃣3️⃣', ['1️⃣', '2️⃣', '3️⃣']),
    ('☺︎☺️', ['☺︎', '☺️']),

    # Combinations
    ('iOS17 macOS14', ['i', 'OS', '17', ' mac', 'OS', '14']),
    ('U.S.A.', ['U', '.', 'S', '.', 'A', '.']),
    ('$123,456', ['$', '123', ',', '456']),
    ('99%', ['99', '%']),
    ('test@example.com', ['test', '@', 'example', '.', 'com']),

    # Zero-width characters
    ('a\u200bb', ['a', '\u200b', 'b']),

    # Soft hyphen
    ('in\u00ADvisible', ['in', '\u00AD', 'visible']),

    # Superscript/subscript
    ('x²y₃', ['x', '²', 'y', '₃']),
]


@pytest.fixture
def pretokenizer() -> Pretokenizer:
    """Create a pretokenizer with script-aware mode for tests."""
    return Pretokenizer(
        main_regex=WORD_LEVEL_REGEX,
        script_specific_regex=SCRIPT_SPECIFIC_REGEX,
        script_specific_scripts=DEFAULT_SCRIPT_SPECIFIC_SCRIPTS
    )


@pytest.mark.parametrize("input_str,expected", PRETOKENIZE_TESTS)
def test_pretokenize(pretokenizer: Pretokenizer, input_str: str, expected: list[str]) -> None:
    """Test pretokenization produces expected tokens."""
    tokens = pretokenizer.pretokenize(input_str)
    assert tokens == expected, f"For input {input_str!r}: got {tokens}, expected {expected}"


@pytest.mark.parametrize("input_str,expected", PRETOKENIZE_TESTS)
def test_pretokenize_reversible(pretokenizer: Pretokenizer, input_str: str, expected: list[str]) -> None:
    """Test that pretokenization is reversible (tokens join back to original)."""
    tokens = pretokenizer.pretokenize(input_str)
    back = "".join(tokens)
    assert input_str == back, f"Not reversible: {input_str!r} -> {tokens} -> {back!r}"


# ---------------------------------------------------------------------------
# Export regexes: GPT4O_EXPORT_REGEX (train with) and GPT4O_COARSE_REGEX
# (export with). Key properties: apostrophes are word-internal ONLY (leading /
# trailing ones split off as punctuation), and a leading run of N spaces splits
# as (N-1) spaces + a single space attached to the following word.
# ---------------------------------------------------------------------------

# (input, fine expected, coarse expected)
EXPORT_REGEX_TESTS = [
    # A trailing/leading apostrophe must NOT attach to the word.
    ("'hello'", ["'", "hello", "'"], ["'", "hello", "'"]),
    ("aides'", ["aides", "'"], ["aides", "'"]),
    # Internal apostrophes stay attached (contractions, names).
    ("isn't", ["isn't"], ["isn't"]),
    ("O'Brien", ["O'Brien"], ["O'Brien"]),
    ("y'all", ["y'all"], ["y'all"]),
    ("cafe's", ["cafe's"], ["cafe's"]),
    # Leading spaces: N spaces -> (N-1) space token + 1 space on the word.
    ("      hello", ["     ", " hello"], ["     ", " hello"]),
    (" hello", [" hello"], [" hello"]),
    ("hello", ["hello"], ["hello"]),
    # Coarse groups a run of words into one pretoken; fine keeps them separate.
    ("   hello world", ["  ", " hello", " world"], ["  ", " hello world"]),
    ("don't stop", ["don't", " stop"], ["don't stop"]),
]


@pytest.mark.parametrize("text,fine,coarse", EXPORT_REGEX_TESTS)
def test_export_regexes(text: str, fine: list[str], coarse: list[str]) -> None:
    """GPT4O_EXPORT_REGEX / GPT4O_COARSE_REGEX apostrophe and leading-space behavior."""
    assert regex.findall(GPT4O_EXPORT_REGEX, text) == fine, f"fine {text!r}"
    assert regex.findall(GPT4O_COARSE_REGEX, text) == coarse, f"coarse {text!r}"
    # Both partitions must be reversible.
    assert "".join(fine) == text
    assert "".join(coarse) == text
