# Copyright 2026-present Kensho Technologies, LLC.
from pathlib import Path
import os
import random
from typing import TextIO, Optional, Any

# do the same encoding as Huggingface bytelevel pretokenization
# see byte_level.rs for the original

# fn bytes_char() -> HashMap<u8, char> {
#     let mut bs: Vec<u8> = vec![];
#     bs.extend(b'!'..=b'~');
#     bs.extend(b'\xA1'..=b'\xAC');
#     bs.extend(b'\xAE'..=b'\xFF');

#     let mut cs: Vec<u32> = bs.iter().map(|i| *i as u32).collect();
#     let mut n = 0;

#     for b in 0..=255u8 {
#         if !bs.contains(&b) {
#             bs.push(b);
#             cs.push(u32::pow(2, 8) + n);
#             n += 1;
#         }
#     }

#     bs.into_iter()
#         .zip(cs)
#         .map(|(f, t)| (f, unsafe { std::char::from_u32_unchecked(t) }))
#         .collect()
# }

def bytes_char() -> tuple[dict[bytes, str], list[tuple[bytes, str]]]:
    bs: list[int] = []
    bs.extend(range(ord('!'), ord('~') + 1))
    bs.extend(range(0xA1, 0xAC + 1))
    bs.extend(range(0xAE, 0xFF + 1))

    # these map to the same character
    cs: list[int] = [b for b in bs]
    n = 0

    # which are invalid chars and have to use a mapping
    added: list[tuple[bytes, str]] = []

    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            added.append((bytes([b]), chr(2 ** 8 + n)))
            n += 1

    result = {bytes([f]): chr(t) for f, t in zip(bs, cs)}

    return result, added

byte_map, added = bytes_char()

inv_byte_map = { v : k for k, v in byte_map.items() }

def tobytes(s : str) -> bytes:
    return b"".join([inv_byte_map[c] for c in s])

# encode a bytestring
def frombytes(bs : bytes) -> str:
        return "".join([byte_map[bytes([b])] for b in bs])

# convert from a hex string to bytes,
# like in a .vocab file
def fromhex(hex : str) -> bytes:
    return bytes.fromhex(hex)

# given a bytes object b, get a hex encoded string
def tohex(b : bytes) -> str:
    return b.hex()

def make_dir_if_not_exists(directory_path : str) -> None:

    # Create a Path object for the directory
    directory = Path(directory_path)

    # Check if the directory exists, and if not, create it
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

# TODO: update these

# dump the vocab to a file, encoded as characters here
# no special tokens are added
# are saved in same order by index, so should preserve order
def write_vocab(vocab : dict[bytes, int],
                filename : str) -> None:
    vocab_size = len(vocab)

    # write these in increasing index order
    # so same as any previous order
    byindex = sorted([(idx,token) for token,idx in vocab.items()])

    with open(filename, 'w') as f:
        for _, token in byindex:
            f.write(token.hex() + '\n')

# read our hex formatted vocab file
# return a list of bytes objects
# input file has one vocab word per line,
# each hex encoded
def load_vocab(vocab_filepath : str) -> list[bytes]:

    if not os.path.exists(vocab_filepath):
        raise FileNotFoundError(f'Missing vocab file: {vocab_filepath}')

    with open(vocab_filepath) as vocab_file:
        # fromhex ignores whitespace from \n at end
        initial_vocab = [bytes.fromhex(token) for token in vocab_file.readlines()]

    return initial_vocab


def fix_random_seed(random_seed : int) -> None:
    random.seed(random_seed)

def create_initial_vocab() -> list[bytes]:
    """Create the initial vocabulary of valid single bytes.

    Returns list of 243 valid byte values that can appear in utf-8
    (excluding 0xC0, 0xC1, and 0xF5-0xFF).
    """
    valid_byte_values = list(range(0, 192)) + list(range(194, 245))  # 192 + 51 = 243 bytes
    return [bytes([idx]) for idx in valid_byte_values]


def verify_all_bytes(vocab : dict[bytes, int]) -> None:

    # Check all valid single bytes are in vocab
    for b in create_initial_vocab():
        if b not in vocab:
            print("missing byte", b)
        assert b in vocab


# are the dicts equal?
# if not print some diagnostics before dying
def verify_dicts(d1 : dict[Any, Any], d2 : dict[Any, Any]) -> None:

    if d1 != d2:
        print(len(d1), len(d2))
        joint_keys = set(d1.keys()) | set(d2.keys())
        for tok in joint_keys:
            cnt = d1.get(tok, None)
            sc = d2.get(tok, None)
            if sc != cnt:
                print("verify_dicts:", tok, cnt, sc)
        assert False


def is_all_digits(byte_data : bytes) -> bool:
    return all(b in b'0123456789' for b in byte_data)

# allow ' _ space and letters
def can_supermerge(byte_data : bytes) -> bool:
    return all(b in b" _'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" for b in byte_data)

def frombytespair(pair: tuple[bytes, bytes]) -> tuple[str, str]:
    left, right = pair
    return (frombytes(left), frombytes(right))



# write the dict, first the number of them, then each one
# the keys are either bytes, or a pair (bytes,bytes), depending
# on ispair
# this is now backwards as they values aren't always going to be unique
# but the indices are
def _write_sorted_dict_intkey(d : dict[Any, Any], f : TextIO, ispair : bool, isstr : bool) -> None:

    # write the size, so we don't need to care about the indices being continuous
    f.write(f"{len(d)}\n")
    sortedd = [(idx, val) for idx, val  in d.items()]
    if ispair:
        for idx, (tok1,tok2) in sortedd:
            if isstr:
                f.write(f"{idx} {tok1} {tok2}\n")
            else:
                f.write(f"{idx} {frombytes(tok1)} {frombytes(tok2)}\n")
    else:
        for idx, tok in sortedd:
            if isstr:
                f.write(f"{idx} {tok}\n")
            else:
                f.write(f"{idx} {frombytes(tok)}\n")


# v2 version: write merges with counts
# d is dict[int, tuple[tuple[bytes, bytes], int]] mapping idx -> ((left, right), count)
def _write_sorted_dict_intkey_with_counts(d : dict[Any, Any], f : TextIO, ispair: bool, isstr : bool) -> None:
    # write the size
    f.write(f"{len(d)}\n")
    # order by index, tok may be a pair or not
    if ispair:
        sorted_pairs = [(idx, tok, cnt, unlocked_flag) for idx, (tok, cnt, unlocked_flag) in d.items()]
        for idx, (tok1, tok2), cnt, unlocked_flag in sorted_pairs:
            if isstr:
                f.write(f"{idx} {tok1} {tok2} {cnt} {unlocked_flag}\n")
            else:
                f.write(f"{idx} {frombytes(tok1)} {frombytes(tok2)} {cnt} {unlocked_flag}\n")
    else:
        sorted_singles = [(idx, tok, cnt) for idx, (tok, cnt) in d.items()]
        for idx, tok, cnt in sorted_singles:
            if isstr:
                f.write(f"{idx} {tok} {cnt}\n")
            else:
                f.write(f"{idx} {frombytes(tok)} {cnt}\n")

# read the dict
def _read_sorted_dict_intkey(f : TextIO, ispair : bool, isstr : bool) -> dict[int, Any]:

    # read the size
    n = int(f.readline().rstrip("\n") )

    d: dict[int, Any] = {}
    if ispair:
        for i in range(n):
            line = f.readline().rstrip("\n").split(" ")
            assert len(line) == 3, f"expected 3 fields: {line} on line {i} of {n}"
            idx_str, tok1_str, tok2_str = line
            idx = int(idx_str)
            tok1: bytes | str
            tok2: bytes | str
            if not isstr:
                tok1 = tobytes(tok1_str)
                tok2 = tobytes(tok2_str)
            else:
                tok1 = tok1_str
                tok2 = tok2_str
            d[idx] = (tok1,tok2)
    else:
        for i in range(n):
            line = f.readline().rstrip("\n").split(" ")
            assert len(line) == 2, f"expected 2 fields: {line} on line {i} of {n}"
            idx_str, tok_str = line
            idx = int(idx_str)
            tok: bytes | str
            if not isstr:
                tok = tobytes(tok_str)
            else:
                tok = tok_str
            d[idx] = tok

    return d


# v2 version: read merges with counts
# Returns:
#   If ispair=True: (dict[int, tuple[tuple[bytes, bytes], int, int]], {})
#       mapping idx -> ((left, right), count, unlocked_flag)
#   If ispair=False: (dict[int, bytes], dict[bytes, int])
#       mapping idx -> token, and token -> count
def _read_sorted_dict_intkey_with_counts(f : TextIO, ispair: bool, isstr : bool) -> tuple[dict[int, Any], dict[Any, int]]:
    # read the size
    n = int(f.readline().rstrip("\n"))

    d: dict[int, Any] = {}
    counts: dict[Any, int] = {}
    if ispair:
        for i in range(n):
            line = f.readline().rstrip("\n").split(" ")
            assert len(line) == 5, f"expected 5 fields: {line} on line {i} of {n}"
            idx_str, tok1_str, tok2_str, cnt_str, unlocked_flag_str = line
            idx = int(idx_str)
            cnt = int(cnt_str)
            unlocked_flag = int(unlocked_flag_str)
            tok1: bytes | str
            tok2: bytes | str
            if not isstr:
                tok1 = tobytes(tok1_str)
                tok2 = tobytes(tok2_str)
            else:
                tok1 = tok1_str
                tok2 = tok2_str
            d[idx] = ((tok1, tok2), cnt, unlocked_flag)
    else:
        for i in range(n):
            line = f.readline().rstrip("\n").split(" ")
            assert len(line) == 3, f"expected 3 fields: {line} on line {i} of {n}"
            idx_str, tok_str, cnt_str = line
            idx = int(idx_str)
            cnt = int(cnt_str)
            tok: bytes | str
            if not isstr:
                tok = tobytes(tok_str)
            else:
                tok = tok_str
            d[idx] = tok
            counts[tok] = cnt

    return d, counts 


# find all occurences of bad_token in the list, and replace with the individual bytes
# retuns the new tokens, and the number of deletions (can be 0)
# replacement_pair should combine to form bad_token 

# TODO: now will need to track initial words in superwords

def blow_up(lst : list[bytes], bad_token : bytes, parts : list[bytes]) -> tuple[list[bytes], int]:

    new_tokens = []
    deletions = 0

    for tok in lst:
        if tok == bad_token:
            # either single bytes or a pair of bytes
            new_tokens.extend(parts) 
            deletions += 1
        else:
            new_tokens.append(tok)
    return new_tokens, deletions

# delete the token from out list
# asserts bad_token was in the list at least once
# since otherwise we should have skipped tokens
def delete(tokens : list[bytes], bad_token : bytes) -> list[bytes]:
    before = len(tokens)
    # delete all occurences
    tokens = [t for t in tokens if t != bad_token]
    # should be in here
    if len(tokens) == before:
        print("debug 1:", frombytes(bad_token), before)

    assert len(tokens) < before
    return tokens


# -----------------------------------------------------------------------------
# Merge function for BPE tokenization
# -----------------------------------------------------------------------------

def merge(tokens: list[bytes], pair: tuple[bytes, bytes]) -> tuple[list[bytes], int]:
    """
    In the list of tokens, replace all consecutive occurrences
    of pair (t1,t2) with the combined token t1+t2
    Example: tokens=[b'a', b'b', b'c', b'a', b'b'],
    pair=(b'a', b'b') -> [b'ab', b'c', b'ab']
    will have max_count merges found,
    unless there are the pair elements are runs of the same
    """
    newtokens = []
    i = 0
    merge_cnt = 0
    left, right = pair
    merged = left + right
    while i < len(tokens):
        # if not at the very last position AND the pair matches, replace it
        if tokens[i] == left and i < len(tokens) - 1 and tokens[i+1] == right:
            newtokens.append(merged)
            merge_cnt += 1
            i += 2
        else:
            newtokens.append(tokens[i])
            i += 1

    # Note: merge_cnt may be 0 if both tokens exist in the chunk but aren't adjacent.
    # This can happen with the token_locations optimization which filters by token
    # presence, not adjacency. Callers should check merge_cnt before using results.
    # Also note that for pairs like (b'2', b'2') and '222222', we may count more
    # potential merges than we can execute since we only do non-overlapping merges.
    # TODO: is this ^^^ still true? 

    return newtokens, merge_cnt


# -----------------------------------------------------------------------------
# Dictionary I/O helper functions
# -----------------------------------------------------------------------------

# write the dict, first the number of them, then each one
# the keys are either bytes, or a pair (bytes,bytes), depending
# on ispair
def _write_sorted_dict(d: dict[Any, Any], f: TextIO, ispair: bool, isstr: bool) -> None:

    # write the size, so we don't need to care about the indices being continuous
    f.write(f"{len(d)}\n")
    sortedd = [(idx, k) for k, idx in d.items()]
    if ispair:
        for idx, (tok1,tok2) in sortedd:
            if isstr:
                f.write(f"{idx} {tok1} {tok2}\n")
            else:
                f.write(f"{idx} {frombytes(tok1)} {frombytes(tok2)}\n")
    else:
        for idx, tok in sortedd:
            if isstr:
                f.write(f"{idx} {tok}\n")
            else:
                f.write(f"{idx} {frombytes(tok)}\n")


# read the dict
def _read_sorted_dict(f: TextIO, ispair: bool, isstr: bool) -> dict[Any, int]:

    # read the size
    n = int(f.readline().rstrip("\n") )

    d: dict[Any, int] = {}
    if ispair:
        for i in range(n):
            line = f.readline().rstrip("\n").split(" ")
            assert len(line) == 3, f"expected 3 fields: {line} on line {i} of {n}"
            idx_str, tok1_str, tok2_str = line
            idx = int(idx_str)
            tok1: bytes | str
            tok2: bytes | str
            if not isstr:
                tok1 = tobytes(tok1_str)
                tok2 = tobytes(tok2_str)
            else:
                tok1 = tok1_str
                tok2 = tok2_str
            d[(tok1,tok2)] = idx
    else:
        for i in range(n):
            line = f.readline().rstrip("\n").split(" ")
            assert len(line) == 2, f"expected 2 fields: {line} on line {i} of {n}"
            idx_str, tok_str = line
            idx = int(idx_str)
            tok: bytes | str
            if not isstr:
                tok = tobytes(tok_str)
            else:
                tok = tok_str
            d[tok] = idx

    return d