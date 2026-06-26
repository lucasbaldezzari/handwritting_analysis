"""
Reporte PDF multipágina de figuras y DataFrames.

La clase PdfReport agrupa todas las figuras (matplotlib o browsers no-matplotlib,
p. ej. el browser Qt de MNE) y los DataFrames generados por un análisis en un único
PDF, una página por elemento.

Uso típico
----------
    from analysis.pdf_report import PdfReport

    with PdfReport("reporte.pdf") as report:
        report.add_dataframe(df, title="Resumen")
        report.add_figure(fig_erp)
        report.add_figure(fig_browser)   # browser MNE → se rasteriza

O sin context manager:

    report = PdfReport("reporte.pdf")
    report.add_figure(fig)
    report.close()
"""

import io

import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _fmt(value):
    """Formatea una celda de DataFrame para la tabla del PDF."""
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return ""
        return f"{value:.3f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


class PdfReport:
    """Acumula figuras y DataFrames en un PDF multipágina."""

    def __init__(self, filepath, metadata=None):
        self.filepath = filepath
        self._pdf = PdfPages(filepath, metadata=metadata)
        self.n_pages = 0

    # ── Context manager ───────────────────────────────────────────────────────
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ── Figuras ───────────────────────────────────────────────────────────────
    def add_figure(self, fig, dpi=200):
        """Agrega una figura al PDF.

        Las figuras matplotlib se guardan como vector; cualquier otra (p. ej. el
        browser Qt de MNE) se rasteriza a PNG y se coloca en una página.
        """
        if isinstance(fig, matplotlib.figure.Figure):
            try:
                self._pdf.savefig(fig, bbox_inches="tight")
                self.n_pages += 1
                return
            except Exception:
                # Si falla el guardado vectorial, se intenta rasterizar.
                pass
        self._add_rasterized(fig, dpi)

    def _add_rasterized(self, fig, dpi):
        """Rasteriza una figura no-matplotlib a una página del PDF."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        img = plt.imread(buf)
        buf.close()

        height, width = img.shape[:2]
        page = plt.figure(figsize=(width / dpi, height / dpi))
        ax = page.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.imshow(img)
        self._pdf.savefig(page)
        self.n_pages += 1
        plt.close(page)

    # ── DataFrames ────────────────────────────────────────────────────────────
    def add_dataframe(self, df, title=None, fontsize=8, figsize=(8.27, 11.69)):
        """Agrega un DataFrame como tabla en una página (A4 vertical por defecto)."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=12, pad=20)

        cell_text = [[_fmt(v) for v in row] for row in df.to_numpy()]
        col_labels = [str(c) for c in df.columns]
        row_labels = [str(i) for i in df.index]

        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            rowLabels=row_labels,
            loc="center",
            cellLoc="center",
            rowLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)
        table.scale(1, 1.4)

        self._pdf.savefig(fig, bbox_inches="tight")
        self.n_pages += 1
        plt.close(fig)

    # ── Cierre ────────────────────────────────────────────────────────────────
    def close(self):
        self._pdf.close()
