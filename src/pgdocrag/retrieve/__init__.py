"""Retrieval strategies layered over the stored chunks.

The ingestion pipeline is held constant across everything in here: same
extraction, same normalisation, same chunks, same vectors. What varies is only
how candidates are found and ordered, which is what makes the comparison between
dense, hybrid and reranked retrieval mean anything.
"""
