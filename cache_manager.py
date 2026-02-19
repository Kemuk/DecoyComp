#!/usr/bin/env python3
"""
Joblib-based caching for expensive computations.

This module provides persistent caching for molecular descriptor calculations
using joblib's Memory class. Descriptors are computed once per SMILES and
cached to disk, surviving across program restarts.
"""
from pathlib import Path
from joblib import Memory

# Default cache directory
DEFAULT_CACHE_DIR = Path(".cache/descriptors")


def get_memory(cache_dir: Path | str | None = None, verbose: int = 0) -> Memory:
    """
    Get a joblib Memory instance for caching.

    Args:
        cache_dir: Directory for cache storage. Defaults to .cache/descriptors
        verbose: Verbosity level (0=silent, 1=info, 2=debug)

    Returns:
        Configured joblib Memory instance
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    return Memory(location=str(cache_path), verbose=verbose)


# Global memory instance for descriptor caching
_descriptor_memory: Memory | None = None


def get_descriptor_memory() -> Memory:
    """Get the global descriptor cache Memory instance."""
    global _descriptor_memory
    if _descriptor_memory is None:
        _descriptor_memory = get_memory()
    return _descriptor_memory


def clear_descriptor_cache():
    """Clear all cached descriptors."""
    memory = get_descriptor_memory()
    memory.clear(warn=False)
    print("[INFO] Descriptor cache cleared")


def get_cache_info() -> dict:
    """Get information about the current cache state."""
    cache_path = DEFAULT_CACHE_DIR

    if not cache_path.exists():
        return {"exists": False, "size_mb": 0, "path": str(cache_path)}

    # Calculate total size
    total_size = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
    size_mb = total_size / (1024 * 1024)

    return {
        "exists": True,
        "size_mb": round(size_mb, 2),
        "path": str(cache_path.absolute()),
    }
