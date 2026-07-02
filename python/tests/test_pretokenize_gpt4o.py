# Copyright 2026-present Kensho Technologies, LLC.
"""Tests for the Pretokenizer class with GPT4O regex defaults."""

import pytest
from boundlessbpe.pretokenize import Pretokenizer
from boundlessbpe.regexconstants import GPT4O_REGEX, SCRIPT_SPECIFIC_GPT4O_REGEX, DEFAULT_SCRIPT_SPECIFIC_SCRIPTS


# Test cases: (input_string, expected_tokens)
# These use GPT4O_REGEX as main and SCRIPT_SPECIFIC_GPT4O_REGEX for CJK etc.
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

    # join-control Devanagari conjunct -- GPT4O splits on ZWJ
    ("क्\u200dष", ["क्", "\u200dष"]),
    # join-control extended conjunct -- GPT4O splits on ZWJ
    ("क्\u200dष\u200dत्र", ["क्", "\u200dष", "\u200dत्र"]),
    # Latin ZWJ sequence -- ZWJ absorbed into following token
    ("a\u200db", ['a', '\u200db']),
    # ZWNJ sequence -- ZWNJ absorbed into following token
    ("स\u200cल", ["स", "\u200cल"]),
    # Contraction, with curly quote, not at end
    ("we're can't you've cows' it's It's", ["we're"," can't"," you've", " cows", "'", " it's", " It's"]),
    # snake_case -- GPT4O absorbs underscore into following word
    ('_init _a_b x_y_z', ['_init', ' _', 'a', '_b', ' x', '_y', '_z']),
    # Mixed_case_VAR -- underscore absorbed into following word
    (' Mixed_case_VAR', [' Mixed', '_case', '_VAR']),
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
    ("'twas", ["'twas"]),  # GPT4O matches as contraction
    ("twas'", ["twas", "'"]),  # or at end
    ("'", ["'"]),  # Lone apostrophe

    # Hyphenated words -- GPT4O absorbs hyphen into following word
    ('well-known', ['well', '-known']),

    # Titlecase digraphs
    ('ǲ', ['ǲ']),  # Latin Small Letter Dz

    # Ligatures
    ('ﬁ', ['ﬁ']),  # fi ligature

    # Combining enclosing marks
    ('a⃝', ['a⃝']),  # a with combining enclosing circle

    # Numbers -- GPT4O uses simple \p{N}{1,3} (left-to-right grouping, no lookahead)
    (' 1 ', [' ', '1', ' ']),
    (' 12 ', [' ', '12', ' ']),
    (' 123 ', [' ', '123', ' ']),
    (' 1234 ', [' ', '123', '4', ' ']),
    (' 12345 ', [' ', '123', '45', ' ']),
    (' 123456 ', [' ', '123', '456', ' ']),
    (' 1234567 ', [' ', '123', '456', '7', ' ']),
    (' 1', [' ', '1']),
    (' 12', [' ', '12']),
    (' 123', [' ', '123']),
    (' 1234', [' ', '123', '4']),
    (' 12345', [' ', '123', '45']),
    (' 123456', [' ', '123', '456']),
    (' 1234567', [' ', '123', '456', '7']),
    ('1 ', ['1', ' ']),
    ('12 ', ['12', ' ']),
    ('123 ', ['123', ' ']),
    ('1234 ', ['123', '4', ' ']),
    ('12345 ', ['123', '45', ' ']),
    ('123456 ', ['123', '456', ' ']),
    ('1234567 ', ['123', '456', '7', ' ']),
    ('1', ['1']),
    ('12', ['12']),
    ('123', ['123']),
    ('1234', ['123', '4']),
    ('12345', ['123', '45']),
    ('123456', ['123', '456']),
    ('1234567', ['123', '456', '7']),
    ('1234 567 89012', ['123', '4', ' ', '567', ' ', '890', '12']),

    # Devanagari digits -- left-to-right grouping
    ('१२३४', ['१२३', '४']),

    # Myanmar digits -- left-to-right grouping
    ('၁၂၃၄', ['၁၂၃', '၄']),

    # Mixed Common and Myanmar digits
    ('12၃၄56', ['12၃', '၄56']),

    # Whitespace
    ('        123', ['       ', ' ', '123']),
    ('\u00a0 \u200e', ['\u00a0', ' \u200e']),
    ('        . Hello', ['       ', ' .', ' Hello']),
    ('        Hello', ['       ', ' Hello']),
    ('        $10', ['       ', ' $', '10']),
    ('  \t \n\n  ', ['  \t \n\n', '  ']),
    ('Hello     world    \n\n \n ', ['Hello', '    ', ' world', '    \n\n \n', ' ']),

    # Multiple spaces edge case (>16 chars) -- GPT4O has no 16-char limit
    ('a' + ' ' * 20 + 'b', ['a', ' ' * 19, ' b']),

    # Tab variations
    ('\t\t\t', ['\t\t\t']),

    # Mixed whitespace
    (' \t\n', [' \t\n']),

    # Punctuation and symbols -- GPT4O absorbs comma into following word
    (', a,b !?.!?', [',', ' a', ',b', ' !?.!?']),
    ('~$A', ['~$','A']),
    ('$10+$12==$22', ['$', '10', '+$', '12','==$','22']),

    # Ellipsis variations
    ('...', ['...']),
    ('…', ['…']),

    # Control/Mark
    ('\u0301\u0300', ['\u0301\u0300']),

    # Emoji -- GPT4O doesn't have special keycap/flag/ZWJ handlers
    ('Á👍🏽👩‍👩‍👧‍👦🇺🇳🏳️‍🌈', ['Á', '👍🏽👩\u200d👩\u200d👧\u200d👦🇺🇳🏳️\u200d🌈']),
    ("🇺🇸🇬🇧", ['🇺🇸🇬🇧']),
    ('1️⃣2️⃣3️⃣', ['1', '️⃣', '2', '️⃣', '3', '️⃣']),
    ('☺︎☺️', ['☺︎', '☺️']),

    # Combinations
    ('iOS17 macOS14', ['i', 'OS', '17', ' mac', 'OS', '14']),
    ('U.S.A.', ['U', '.S', '.A', '.']),
    ('$123,456', ['$', '123', ',', '456']),
    ('99%', ['99', '%']),
    ('test@example.com', ['test', '@example', '.com']),

    # Zero-width characters -- ZWSP absorbed into following token
    ('a\u200bb', ['a', '\u200bb']),

    # Soft hyphen -- absorbed into following token
    ('in\u00ADvisible', ['in', '\u00ADvisible']),

    # Superscript/subscript
    ('x²y₃', ['x', '²', 'y', '₃']),

    # Mixed English, punctuation, CJK, and Latin
    ('To be, or not to be: that is the question: 德國HYDAC電磁球閥 Hi',
     ['To', ' be', ',', ' or', ' not', ' to', ' be', ':', ' that', ' is', ' the', ' question', ':', ' 德', '國', 'HYDAC', '電', '磁', '球', '閥', ' Hi']),
]


@pytest.fixture
def pretokenizer() -> Pretokenizer:
    """Create a pretokenizer with GPT4O regex and script-aware mode for tests."""
    return Pretokenizer(
        main_regex=GPT4O_REGEX,
        script_specific_regex=SCRIPT_SPECIFIC_GPT4O_REGEX,
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
