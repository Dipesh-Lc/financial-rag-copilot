"""
filters.py
Helpers to build Chroma metadata filter dicts.
"""

from __future__ import annotations


def ticker_filter(ticker: str) -> dict:
    return {"ticker": ticker.upper()}


def form_type_filter(form_type: str) -> dict:
    return {"form_type": form_type}


def section_filter(section_name: str) -> dict:
    return {"section_name": section_name}


def filing_date_filter(filing_date: str) -> dict:
    return {"filing_date": filing_date}


def combined_filter(**kwargs) -> dict | None:
    """
    Build a Chroma $and filter from keyword args.
    E.g. combined_filter(ticker="AAPL", form_type="10-K")
    """
    conditions = [{k: v} for k, v in kwargs.items() if v is not None]
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


__all__ = ["ticker_filter", "form_type_filter", "section_filter", "filing_date_filter", "combined_filter"]
