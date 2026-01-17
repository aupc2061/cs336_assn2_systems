import os
from typing import BinaryIO
import multiprocessing
import regex as re
from collections import Counter
import pickle
from typing import Iterable, Iterator

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


## Usage
# with open("data/TinyStoriesV2-GPT4-train.txt", "rb") as f:
#     num_processes = 4
#     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

#     # The following is a serial implementation, but you can parallelize this
#     # by sending each start/end pair to a set of processes.
#     for start, end in zip(boundaries[:-1], boundaries[1:]):
#         f.seek(start)
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")
#         # Run pre-tokenization on your chunk and store the counts for each pre-token
        
if __name__ == "__main__":
    filepath = r"D:\CS336\assignment1-basics\data\TinyStoriesV2-GPT4-valid.txt"
    special_tokens = ["<|endoftext|>"]
    counts = parallel_pretokenize(filepath, num_processes=4, special_tokens=special_tokens)
    with open("pretokenized_counts.pkl", "wb") as f:
        pickle.dump(counts, f)