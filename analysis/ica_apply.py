try:
    from handwriting_analysis.ica.ica_apply import ICAApplicator
except ModuleNotFoundError as exc:
    if exc.name != "handwriting_analysis":
        raise

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from handwriting_analysis.ica.ica_apply import ICAApplicator

__all__ = ["ICAApplicator"]
