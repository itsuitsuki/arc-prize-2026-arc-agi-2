"""A safetensors file's header without reading tensor data: 8-byte little-endian length, then the JSON header."""
from contextlib import nullcontext
import json
from pathlib import Path
import struct
from typing import BinaryIO


def safetensors_header(source: Path | BinaryIO) -> tuple[dict, int]:
    """(name -> {"dtype", "shape", "data_offsets"} without __metadata__, byte offset where the data section starts) of a
    file path or of a binary stream positioned at the file's start (e.g. a remote file opened with HfFileSystem)."""
    with (open(source, "rb") if isinstance(source, (str, Path)) else nullcontext(source)) as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < size < 100_000_000:
            raise ValueError(f"{source}: invalid safetensors header length {size}")
        entries = json.loads(stream.read(size))
    entries.pop("__metadata__", None)
    return entries, 8 + size
