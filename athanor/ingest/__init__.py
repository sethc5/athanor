"""athanor.ingest — literature fetching and text preparation."""
from athanor.ingest.arxiv_client import ArxivClient, Paper
from athanor.ingest.parser import parse_papers
from athanor.ingest.semantic_scholar import SemanticScholarClient
from athanor.ingest.pdf import enrich_papers_with_fulltext

__all__ = ["ArxivClient", "Paper", "parse_papers", "SemanticScholarClient", "enrich_papers_with_fulltext"]
