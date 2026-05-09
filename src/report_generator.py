from __future__ import annotations

import base64
import html
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


class ReportGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def _img_to_data_uri(self, data: bytes) -> str:
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def _plot_data_uri(self, series: pd.Series, kind: str) -> str | None:
        clean_series = series.dropna()
        if clean_series.empty:
            return None

        plt.figure(figsize=(10, 6) if kind == "hist" else (8, 6))
        if kind == "hist":
            plt.hist(clean_series, bins=30)
            plt.xlabel(series.name or "")
            plt.ylabel("Frequency")
            plt.title(f"Distribution of {series.name}")
        else:
            plt.boxplot(clean_series, vert=True)
            plt.ylabel(series.name or "")
            plt.title(f"Boxplot of {series.name}")

        buff = BytesIO()
        plt.savefig(buff, bbox_inches="tight", format="png")
        plt.close()
        return self._img_to_data_uri(buff.getvalue())

    def _col_stats_table_html(self, summary: pd.DataFrame, column: str) -> str:
        if column not in summary.index:
            return "<div class='muted'>No stats available.</div>"

        s = summary.loc[column]
        # Keep a stable, readable order
        preferred = [
            "count",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
            "missing_values",
            "missing_pct",
            "unique",
            "unique_pct",
            "zeros",
            "zeros_pct",
            "variance",
            "skew",
            "kurtosis",
            "iqr",
            "outlier_count",
            "outlier_pct",
        ]
        keys = [k for k in preferred if k in s.index] + [k for k in s.index if k not in preferred]

        rows = []
        for k in keys:
            v = s.get(k)
            if pd.isna(v):
                v_str = ""
            elif isinstance(v, (float, int)):
                v_str = f"{v:.6g}"
            else:
                v_str = str(v)
            rows.append(
                f"<tr><td class='k'>{html.escape(str(k))}</td><td class='v'>{html.escape(v_str)}</td></tr>"
            )
        return "<table class='stats'><tbody>" + "".join(rows) + "</tbody></table>"

    def _corr_table_html(self, corr: pd.DataFrame, max_cols: int = 30) -> str:
        if corr.empty:
            return "<div class='muted'>No correlations (no numerical columns).</div>"

        # Large matrices get unreadable fast; cap for UI safety.
        if corr.shape[1] > max_cols:
            corr = corr.iloc[:max_cols, :max_cols]
            note = f"<div class='muted'>Showing first {max_cols} columns only.</div>"
        else:
            note = ""

        # Render as simple HTML table (no heavy JS).
        header = "<tr><th></th>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in corr.columns) + "</tr>"
        body_rows = []
        for idx, row in corr.iterrows():
            tds = "".join(f"<td>{'' if pd.isna(x) else f'{x:.3f}'}</td>" for x in row.values)
            body_rows.append(f"<tr><th>{html.escape(str(idx))}</th>{tds}</tr>")
        return note + "<div class='table-scroll'><table class='corr'>" + header + "".join(body_rows) + "</table></div>"

    def _categorical_stats_table_html(self, df: pd.DataFrame, column: str) -> str:
        s = df[column]
        n_rows = len(s)
        missing = int(s.isna().sum())
        missing_pct = (missing / n_rows * 100.0) if n_rows else 0.0
        unique = int(s.nunique(dropna=True))
        unique_pct = (unique / n_rows * 100.0) if n_rows else 0.0
        mode_series = s.mode(dropna=True)
        mode_value = mode_series.iloc[0] if not mode_series.empty else ""

        rows = [
            ("count", int(s.notna().sum())),
            ("missing_values", missing),
            ("missing_pct", f"{missing_pct:.6g}"),
            ("unique", unique),
            ("unique_pct", f"{unique_pct:.6g}"),
            ("mode", mode_value),
        ]

        top_counts = s.value_counts(dropna=True).head(5)
        for idx, (val, cnt) in enumerate(top_counts.items(), start=1):
            rows.append((f"top_{idx}_value", val))
            rows.append((f"top_{idx}_count", int(cnt)))

        html_rows = []
        for k, v in rows:
            html_rows.append(
                f"<tr><td class='k'>{html.escape(str(k))}</td><td class='v'>{html.escape(str(v))}</td></tr>"
            )
        return "<table class='stats'><tbody>" + "".join(html_rows) + "</tbody></table>"

    def write_html_report(
        self,
        *,
        df: pd.DataFrame,
        numerical_columns: list[str],
        categorical_columns: list[str],
        summary: pd.DataFrame,
        corr_pearson: pd.DataFrame | None = None,
        corr_spearman: pd.DataFrame | None = None,
        report_name: str = "report.html",
    ) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        title = "EDA"
        parts: list[str] = []

        parts.append(
            f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #0b0f17;
      --panel: #111827;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --border: #243044;
      --accent: #60a5fa;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    .container {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px;
    }}
    .header {{
      display: flex;
      gap: 16px;
      align-items: baseline;
      justify-content: space-between;
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px;
      margin-bottom: 16px;
    }}
    .header h1 {{
      font-size: 22px;
      margin: 0;
      letter-spacing: 0.2px;
    }}
    .header-actions {{
      display: flex;
      gap: 8px;
      align-items: center;
    }}
    .save-btn {{
      border: 1px solid var(--border);
      background: #1f2937;
      color: var(--text);
      border-radius: 8px;
      padding: 8px 12px;
      font-size: 13px;
      cursor: pointer;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
    }}
    .section {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      margin: 16px 0;
    }}
    .section h2 {{
      font-size: 16px;
      margin: 0 0 12px 0;
      color: var(--text);
    }}
    .muted {{ color: var(--muted); }}
    .cols {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
    }}
    .col-card {{
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      background: rgba(255,255,255,0.02);
    }}
    .col-title {{
      font-size: 15px;
      margin: 0 0 10px 0;
      color: var(--accent);
    }}
    .plots {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      align-items: start;
    }}
    .plot {{
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px;
      background: rgba(0,0,0,0.15);
    }}
    .plot img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 8px;
    }}
    table.stats {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 13px;
    }}
    table.stats td {{
      border-top: 1px solid var(--border);
      padding: 8px 10px;
      vertical-align: top;
    }}
    table.stats td.k {{
      width: 220px;
      color: var(--muted);
    }}
    .table-scroll {{
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 12px;
    }}
    .comment-box {{
      margin-top: 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .comment-box label {{
      color: var(--muted);
      font-size: 13px;
    }}
    .comment-box textarea {{
      width: 100%;
      min-height: 90px;
      padding: 10px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: rgba(0,0,0,0.2);
      color: var(--text);
      font-size: 13px;
      font-family: inherit;
      resize: vertical;
      box-sizing: border-box;
    }}
    table.corr {{
      width: max-content;
      border-collapse: collapse;
      font-size: 12px;
      background: rgba(0,0,0,0.12);
    }}
    table.corr th, table.corr td {{
      border: 1px solid var(--border);
      padding: 6px 8px;
      text-align: right;
      white-space: nowrap;
    }}
    table.corr th {{
      position: sticky;
      top: 0;
      background: rgba(17,24,39,0.95);
      text-align: left;
    }}
    @media (max-width: 900px) {{
      .plots {{ grid-template-columns: 1fr; }}
      table.stats td.k {{ width: 160px; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>{html.escape(title)}</h1>
      <div class="header-actions">
        <button class="save-btn" onclick="saveReport()">Zapisz raport</button>
        <div class="meta">Rows: {len(df)} &nbsp;|&nbsp; Numerical columns: {len(numerical_columns)} &nbsp;|&nbsp; Categorical columns: {len(categorical_columns)}</div>
      </div>
    </div>
"""
        )

        parts.append("<div class='section'><h2>Numerical Analysis</h2><div class='cols'>")

        for col in numerical_columns:
            hist_uri = self._plot_data_uri(df[col], kind="hist")
            box_uri = self._plot_data_uri(df[col], kind="box")

            hist_html = (
                f"<img alt='Histogram {html.escape(col)}' src='{hist_uri}' />"
                if hist_uri
                else "<div class='muted'>Histogram not found.</div>"
            )
            box_html = (
                f"<img alt='Boxplot {html.escape(col)}' src='{box_uri}' />"
                if box_uri
                else "<div class='muted'>Boxplot not found.</div>"
            )

            parts.append(
                f"""
      <div class="col-card" id="{html.escape(col)}">
        <div class="col-title">{html.escape(col)}</div>
        <div class="plots">
          <div class="plot">{hist_html}</div>
          <div class="plot">{box_html}</div>
        </div>
        {self._col_stats_table_html(summary, col)}
        <div class="comment-box">
          <label for="comment-{html.escape(col)}">Komentarz użytkownika</label>
          <textarea id="comment-{html.escape(col)}" placeholder="Dodaj swoje wnioski lub uwagi do kolumny {html.escape(col)}..."></textarea>
        </div>
      </div>
"""
            )

        parts.append("</div></div>")

        parts.append("<div class='section'><h2>Categorical Analysis</h2><div class='cols'>")
        for col in categorical_columns:
            parts.append(
                f"""
      <div class="col-card" id="{html.escape(col)}">
        <div class="col-title">{html.escape(col)}</div>
        {self._categorical_stats_table_html(df, col)}
        <div class="comment-box">
          <label for="comment-{html.escape(col)}">Komentarz użytkownika</label>
          <textarea id="comment-{html.escape(col)}" placeholder="Dodaj swoje wnioski lub uwagi do kolumny {html.escape(col)}..."></textarea>
        </div>
      </div>
"""
            )
        parts.append("</div></div>")

        if corr_pearson is not None or corr_spearman is not None:
            parts.append("<div class='section'><h2>Correlations</h2>")
            if corr_pearson is not None:
                parts.append("<div class='muted'>Pearson</div>")
                parts.append(self._corr_table_html(corr_pearson))
            if corr_spearman is not None:
                parts.append("<div style='height:12px'></div>")
                parts.append("<div class='muted'>Spearman</div>")
                parts.append(self._corr_table_html(corr_spearman))
            parts.append("</div>")

        parts.append(
            """
<script>
function saveReport() {
  // textarea values are not reflected in outerHTML; write them back before saving
  const docClone = document.documentElement.cloneNode(true);
  const srcTextareas = document.querySelectorAll("textarea");
  const dstTextareas = docClone.querySelectorAll("textarea");
  const n = Math.min(srcTextareas.length, dstTextareas.length);
  for (let i = 0; i < n; i++) {
    const val = srcTextareas[i].value ?? "";
    dstTextareas[i].textContent = val;
    dstTextareas[i].setAttribute("data-saved-value", val);
  }

  const html = "<!doctype html>\\n" + docClone.outerHTML;
  const blob = new Blob([html], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "EDA_report.html";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
</script>
"""
        )
        parts.append("</div></body></html>")

        report_path = self.output_dir / report_name
        report_path.write_text("".join(parts), encoding="utf-8")
        return report_path

