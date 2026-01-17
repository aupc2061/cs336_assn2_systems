import os
import regex as re
import pickle
from typing import Iterable, Iterator
from collections import Counter
import multiprocessing
from typing import BinaryIO

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_chunk(args: tuple[str, int, int, list[str]]) -> Counter:
    """Process a single chunk and return pre-tokens for it"""
    fp, start, end, special_tokens = args
    with open(fp, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8")
        # Normalize Windows line endings to Unix
        chunk = chunk.replace("\r\n", "\n")
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # Split on special tokens before actual split
        if special_tokens:
            pattern = "|".join(re.escape(tok) for tok in special_tokens)
            segments = re.split(pattern, chunk)
        else:
            segments = [chunk]
        tokens = []
        for seg in segments:
            tokens.extend(re.findall(PAT, seg))
        return Counter(tokens)


def parallel_pretokenize(filepath: str, num_processes: int = 4, special_tokens: list[str] = None) -> Counter:
    """Run pre-tokenization in parallel across file chunks."""
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]
    
    # Check file size - use single process for small files to avoid multiprocessing overhead
    file_size = os.path.getsize(filepath)
    if file_size < 1_000_000:  # Less than 1MB, use single process
        return process_chunk((filepath, 0, file_size, special_tokens))
    
    with open(filepath, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    
    # Create (filepath, start, end) tuples for each chunk
    chunk_args = [
        (filepath, start, end, special_tokens) 
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    
    # Process chunks in parallel
    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(process_chunk, chunk_args)
    
    # Merge all counters
    total_counts = Counter()
    for result in results:
        total_counts.update(result)
    
    return total_counts


def train_bpe(filepath: str | os.PathLike, vocab_size: int = 512, special_tokens: list[str] = None) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a BPE tokenizer on a given text file.
       Inputs:
           filepath: path to the text file to train on
           vocab_size: desired vocabulary size (including initial byte vocab)
           special_tokens: list of special tokens to include in the vocabulary
       Returns:
           vocab: dictionary mapping token IDs to byte sequences
           merges: list of byte pair merges performed during training
    """
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]
    special_tokens = sorted(special_tokens)

    # Get pre-token counts
    counts = parallel_pretokenize(filepath, num_processes=4, special_tokens=special_tokens)
    
    # Initialize encoding with byte pairs
    enc = {}
    for tok, freq in counts.items():
        bt = tuple(bytes([b]) for b in tok.encode("utf-8"))
        enc[bt] = enc.get(bt, 0) + freq
    
    # Perform BPE merges
    merges = []
    d = {}
    for e in enc:
        for i, j in zip(e[:-1], e[1:]):
            d[(i, j)] = d.get((i, j), 0) + enc[e]
    
    num_merges = vocab_size - 256  - len(special_tokens) # Subtract initial byte vocabulary and special tokens
    
    for _ in range(num_merges):
        if not d:
            break
        s = max(d, key=lambda x: (d[x], x))
        del d[s]
        merges.append(s)
        
        new_enc = {}
        for tok, freq in enc.items():
            new_tok = []
            i = 0
            while i < len(tok):
                if i < len(tok) - 1 and (tok[i], tok[i + 1]) == s:
                    merged = tok[i] + tok[i + 1]

                    if new_tok: # left neighbour, so decrement count for that and increase for new merged
                        old_leftn = (new_tok[-1], tok[i])
                        d[old_leftn] = d.get(old_leftn, 0) - freq
                        if d[old_leftn] <= 0:
                            d.pop(old_leftn, None)
                        new_ln = (new_tok[-1], merged)
                        d[new_ln] = d.get(new_ln, 0) + freq

                    if i + 2 < len(tok): # right neighbour, so decrement count for that and increase for new merged
                        old_rightn = (tok[i + 1], tok[i + 2])
                        d[old_rightn] = d.get(old_rightn, 0) - freq
                        if d[old_rightn] <= 0:
                            d.pop(old_rightn, None)
                        new_rn = (merged, tok[i + 2])
                        d[new_rn] = d.get(new_rn, 0) + freq
                    new_tok.append(merged)
                    i += 2
                else:
                    new_tok.append(tok[i])
                    i += 1
            new_enc[tuple(new_tok)] = new_enc.get(tuple(new_tok), 0) + freq
        enc = new_enc
        
    # Build vocabulary
    vocab = {i: bytes([i]) for i in range(256)}
    for i, tok in enumerate(special_tokens):
        vocab[256 + i] = tok.encode("utf-8")
    for i, (a, b) in enumerate(merges):
        vocab[i + 256 + len(special_tokens)] = a + b
    
    return vocab, merges

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.merges = merges
        self.special_tokens = special_tokens
        self.tok2id = {v: k for k, v in self.vocab.items()}
        self.merge_ranks = {p: i for i, p in enumerate(self.merges)}

    def encode_bpe(self, text: str) -> list[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        encs = [tuple(bytes([b]) for b in k.group(0).encode("utf-8")) for k in re.finditer(PAT, text)]
        for i, enc in enumerate(encs):
            while True:
                pairs = [(enc[j], enc[j + 1]) for j in range(len(enc) - 1)]
                rankp = [(self.merge_ranks[p], j, p) for j, p in enumerate(pairs) if p in self.merge_ranks]
                if not rankp:
                    break
                _, j, _ = min(rankp)
                enc = enc[:j] + (enc[j] + enc[j + 1],) + enc[j + 2:]
            encs[i] = enc
        encoded = []
        for i in range(len(encs)):
            encoded.extend([self.tok2id[x] for x in encs[i]])
        return encoded

    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            return self.encode_bpe(text)
        # Sort special tokens by length (longest first) to handle overlapping tokens
        sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "(" + "|".join(re.escape(tok) for tok in sorted_specials) + ")"
        segments = re.split(pattern, text)
        encoded = []
        for seg in segments:
            if not seg:
                continue
            if seg in self.special_tokens:
                encoded.append(self.tok2id[seg.encode("utf-8")])
            else:
                encoded.extend(self.encode_bpe(seg))
        return encoded
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    def decode(self, ids: list[int]) -> str:
       s = b"".join([self.vocab[k] for k in ids]).decode("utf-8", errors="replace")
       return s 
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            enc = self.encode(s)
            for i in enc:
                yield i