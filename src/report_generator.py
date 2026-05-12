from __future__ import annotations

import base64
import html
from io import BytesIO
from pathlib import Path

import numpy as np
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
            plt.title(str(series.name or "column"))
        else:
            plt.boxplot(clean_series, vert=True)
            plt.ylabel(series.name or "")
            plt.title(str(series.name or "column"))

        buff = BytesIO()
        plt.savefig(buff, bbox_inches="tight", format="png")
        plt.close()
        return self._img_to_data_uri(buff.getvalue())

    def _heatmap_data_uri(
        self,
        data: np.ndarray,
        *,
        title: str,
        xlabel: str = "",
        ylabel: str = "",
        xticklabels: list[str] | None = None,
        yticklabels: list[str] | None = None,
        cmap: str = "viridis",
        figsize: tuple[int, int] = (11, 4),
    ) -> str:
        plt.figure(figsize=figsize)
        plt.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap)
        plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if xticklabels is not None:
            plt.xticks(range(len(xticklabels)), xticklabels, rotation=75, ha="right", fontsize=8)
        if yticklabels is not None:
            plt.yticks(range(len(yticklabels)), yticklabels, fontsize=8)
        plt.tight_layout()
        buff = BytesIO()
        plt.savefig(buff, bbox_inches="tight", format="png")
        plt.close()
        return self._img_to_data_uri(buff.getvalue())

    def _dataset_overview_html(self, df: pd.DataFrame) -> str:
        rows = len(df)
        cols = df.shape[1]
        mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        dup_rows = int(df.duplicated().sum())
        missing_cells = int(df.isna().sum().sum())
        items = [
            ("rows", rows),
            ("columns", cols),
            ("memory usage (MB)", f"{mem_mb:.2f}"),
            ("duplicate rows", dup_rows),
            ("missing cells", missing_cells),
        ]
        trs = "".join(
            f"<tr><td class='k'>{html.escape(str(k))}</td><td class='v'>{html.escape(str(v))}</td></tr>"
            for k, v in items
        )
        return "<table class='stats'><tbody>" + trs + "</tbody></table>"

    def _top_missing_columns_html(self, column_profile: pd.DataFrame, top_n: int = 10) -> str:
        if column_profile is None or column_profile.empty:
            return "<div class='muted'>—</div>"
        top = (
            column_profile.sort_values(["missing_pct", "missing_count"], ascending=False)
            .head(top_n)
            .loc[:, ["column", "missing_count", "missing_pct", "role", "dtype"]]
        )
        header = "<tr><th>column</th><th>missing_count</th><th>missing_%</th><th>role</th><th>dtype</th></tr>"
        body = []
        for _, r in top.iterrows():
            body.append(
                "<tr>"
                f"<td>{html.escape(str(r['column']))}</td>"
                f"<td>{int(r['missing_count'])}</td>"
                f"<td>{float(r['missing_pct']):.2f}</td>"
                f"<td>{html.escape(str(r['role']))}</td>"
                f"<td>{html.escape(str(r['dtype']))}</td>"
                "</tr>"
            )
        return "<div class='table-scroll'><table class='corr'>" + header + "".join(body) + "</table></div>"

    def _missingness_heatmap_uri(self, df: pd.DataFrame, max_rows: int = 500) -> str | None:
        if df is None or df.empty:
            return None
        sample = df.head(max_rows)
        miss = sample.isna().to_numpy(dtype=float)
        return self._heatmap_data_uri(
            miss,
            title=f"Missing values ({len(sample)} rows)",
            xlabel="columns",
            ylabel="rows",
            xticklabels=[str(c) for c in sample.columns],
            yticklabels=None,
            cmap="magma",
            figsize=(11, 5),
        )

    def _column_overview_html(self, column_profile: pd.DataFrame) -> str:
        if column_profile is None or column_profile.empty:
            return "<div class='muted'>—</div>"
        header = (
            "<tr>"
            "<th>Column name <button class='mini-btn' onclick='sortOverview(0)'>Sort</button> <button class='mini-btn' onclick='filterOverview(0)'>Filter</button></th>"
            "<th>Data type <button class='mini-btn' onclick='sortOverview(1)'>Sort</button> <button class='mini-btn' onclick='filterOverview(1)'>Filter</button></th>"
            "<th>Role <button class='mini-btn' onclick='sortOverview(2)'>Sort</button> <button class='mini-btn' onclick='filterOverview(2)'>Filter</button></th>"
            "<th>Missing % <button class='mini-btn' onclick='sortOverview(3)'>Sort</button> <button class='mini-btn' onclick='filterOverview(3)'>Filter</button></th>"
            "<th>Unique values <button class='mini-btn' onclick='sortOverview(4)'>Sort</button> <button class='mini-btn' onclick='filterOverview(4)'>Filter</button></th>"
            "<th>Example values <button class='mini-btn' onclick='sortOverview(5)'>Sort</button> <button class='mini-btn' onclick='filterOverview(5)'>Filter</button></th>"
            "<th>Comment</th>"
            "</tr>"
        )
        body = []
        for _, r in column_profile.iterrows():
            warn = ""
            if bool(r.get("high_cardinality")) and str(r.get("role")) in {"identifier", "text", "categorical"}:
                warn = (
                    "<div class='muted' style='margin-top:6px'>"
                    "Many distinct values."
                    "</div>"
                )
            body.append(
                "<tr>"
                f"<td>{html.escape(str(r['column']))}{warn}</td>"
                f"<td>{html.escape(str(r['dtype']))}</td>"
                f"<td>{html.escape(str(r['role']))}</td>"
                f"<td data-sort='{float(r['missing_pct']):.6f}'>{float(r['missing_pct']):.2f}</td>"
                f"<td data-sort='{int(r['n_unique'])}'>{int(r['n_unique'])}</td>"
                f"<td>{html.escape(str(r['example_values']))}</td>"
                f"<td><textarea class='inline-comment' id='col-comment-{html.escape(str(r['column']))}' "
                "placeholder=''></textarea></td>"
                "</tr>"
            )
        return "<div class='table-scroll overview'><table id='column-overview-table' class='corr compact-table'>" + header + "".join(body) + "</table></div>"

    def _corr_heatmap_and_extremes(
        self, corr: pd.DataFrame, top_n: int = 8
    ) -> tuple[str | None, str, str]:
        if corr is None or corr.empty or corr.shape[0] < 2:
            return None, "<div class='muted'>Need at least 2 numeric columns.</div>", ""

        mat = corr.to_numpy(dtype=float)
        heat_uri = self._heatmap_data_uri(
            mat,
            title="Pearson correlation",
            xlabel="columns",
            ylabel="columns",
            xticklabels=[str(c) for c in corr.columns],
            yticklabels=[str(c) for c in corr.index],
            cmap="coolwarm",
            figsize=(10, 8),
        )

        pairs = []
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                v = corr.iloc[i, j]
                if pd.notna(v):
                    pairs.append((cols[i], cols[j], float(v)))

        if not pairs:
            return heat_uri, "<div class='muted'>No correlations computed.</div>", ""

        pairs_sorted = sorted(pairs, key=lambda x: x[2])
        top_neg = pairs_sorted[:top_n]
        top_pos = list(reversed(pairs_sorted[-top_n:]))

        def _pairs_html(title: str, ps: list[tuple[str, str, float]]) -> str:
            header = "<tr><th>col_a</th><th>col_b</th><th>corr</th></tr>"
            body = []
            for a, b, v in ps:
                body.append(
                    "<tr>"
                    f"<td>{html.escape(a)}</td><td>{html.escape(b)}</td><td>{v:.3f}</td>"
                    "</tr>"
                )
            return (
                f"<div class='muted'>{html.escape(title)}</div>"
                f"<div class='table-scroll'><table class='corr'>{header}{''.join(body)}</table></div>"
            )

        return heat_uri, _pairs_html("Highest r", top_pos), _pairs_html("Lowest r", top_neg)

    def _col_stats_table_html(self, summary: pd.DataFrame, column: str) -> str:
        if column not in summary.index:
            return "<div class='muted'>No stats available.</div>"

        s = summary.loc[column]
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

    def _categorical_stats_table_html(self, df: pd.DataFrame, column: str) -> str:
        s = df[column]
        n_rows = len(s)
        missing = int(s.isna().sum())
        missing_pct = (missing / n_rows * 100.0) if n_rows else 0.0
        unique = int(s.nunique(dropna=True))
        unique_pct = (unique / n_rows * 100.0) if n_rows else 0.0
        mode_series = s.mode(dropna=True)
        mode_value = mode_series.iloc[0] if not mode_series.empty else ""
        top1_pct = 0.0
        vc = s.value_counts(dropna=True)
        if n_rows and not vc.empty:
            top1_pct = float(vc.iloc[0] / n_rows * 100.0)

        rows = [
            ("count", int(s.notna().sum())),
            ("missing_values", missing),
            ("missing_pct", f"{missing_pct:.6g}"),
            ("unique", unique),
            ("unique_pct", f"{unique_pct:.6g}"),
            ("mode", mode_value),
            ("top1_pct", f"{top1_pct:.2f}"),
        ]

        top_counts = vc.head(10)
        for idx, (val, cnt) in enumerate(top_counts.items(), start=1):
            rows.append((f"top_{idx}_value", val))
            rows.append((f"top_{idx}_count", int(cnt)))

        html_rows = []
        for k, v in rows:
            html_rows.append(
                f"<tr><td class='k'>{html.escape(str(k))}</td><td class='v'>{html.escape(str(v))}</td></tr>"
            )
        return "<table class='stats'><tbody>" + "".join(html_rows) + "</tbody></table>"

    def _categorical_bar_uri(self, df: pd.DataFrame, column: str, top_n: int = 10) -> str | None:
        s = df[column]
        vc = s.value_counts(dropna=True).head(top_n)
        if vc.empty:
            return None
        plt.figure(figsize=(10, 4))
        plt.bar([str(x) for x in vc.index], vc.values)
        plt.title(column)
        plt.xticks(rotation=60, ha="right", fontsize=8)
        plt.ylabel("count")
        plt.tight_layout()
        buff = BytesIO()
        plt.savefig(buff, bbox_inches="tight", format="png")
        plt.close()
        return self._img_to_data_uri(buff.getvalue())

    def write_html_report(
        self,
        *,
        df: pd.DataFrame,
        numerical_columns: list[str],
        categorical_columns: list[str],
        datetime_columns: list[str],
        column_profile: pd.DataFrame,
        summary: pd.DataFrame,
        corr_pearson: pd.DataFrame | None = None,
        report_name: str = "report.html",
    ) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        title = "Report"
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
      max-width: 1700px;
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
    .mini-btn {{
      border: 1px solid var(--border);
      background: rgba(31,41,55,0.85);
      color: var(--text);
      border-radius: 6px;
      padding: 2px 6px;
      font-size: 11px;
      cursor: pointer;
      margin-left: 4px;
    }}
    textarea.inline-comment {{
      width: 260px;
      min-height: 60px;
      padding: 8px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: rgba(0,0,0,0.2);
      color: var(--text);
      font-size: 13px;
      font-family: inherit;
      resize: vertical;
      box-sizing: border-box;
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
    .table-scroll.overview {{
      overflow-x: hidden;
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
    table.compact-table th, table.compact-table td {{
      padding: 4px 6px;
      font-size: 11px;
      line-height: 1.2;
      white-space: normal;
      word-break: break-word;
      vertical-align: top;
    }}
    table.compact-table {{
      width: 100%;
      table-layout: fixed;
    }}
    table.compact-table th:nth-child(1), table.compact-table td:nth-child(1) {{
      width: 18%;
    }}
    table.compact-table th:nth-child(2), table.compact-table td:nth-child(2) {{
      width: 10%;
    }}
    table.compact-table th:nth-child(3), table.compact-table td:nth-child(3) {{
      width: 10%;
    }}
    table.compact-table th:nth-child(4), table.compact-table td:nth-child(4) {{
      width: 8%;
    }}
    table.compact-table th:nth-child(5), table.compact-table td:nth-child(5) {{
      width: 8%;
    }}
    table.compact-table th:nth-child(6), table.compact-table td:nth-child(6) {{
      width: 24%;
    }}
    table.compact-table th:nth-child(7), table.compact-table td:nth-child(7) {{
      width: 22%;
    }}
    table.compact-table textarea.inline-comment {{
      width: 100%;
      min-height: 40px;
      font-size: 11px;
      padding: 5px;
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
        <button class="save-btn" onclick="saveReport()">Download HTML</button>
        <div class="meta">{len(df)} rows · {len(numerical_columns)} numeric · {len(categorical_columns)} categorical</div>
      </div>
    </div>
"""
        )

        parts.append("<div class='section'><h2>Overview</h2>")
        parts.append(self._dataset_overview_html(df))
        parts.append("</div>")

        parts.append("<div class='section'><h2>Columns</h2>")
        parts.append(self._column_overview_html(column_profile))
        parts.append("</div>")

        miss_uri = self._missingness_heatmap_uri(df)
        parts.append("<div class='section'><h2>Missing</h2>")
        if miss_uri:
            parts.append(f"<div class='plot'><img alt='Missing values heatmap' src='{miss_uri}' /></div>")
        parts.append("<div style='height:12px'></div>")
        parts.append("<div class='muted'>Most empty columns</div>")
        parts.append(self._top_missing_columns_html(column_profile))
        parts.append("</div>")

        if datetime_columns:
            parts.append("<div class='section'><h2>Dates</h2><div class='cols'>")
            for col in datetime_columns:
                dt = pd.to_datetime(df[col], errors="coerce")
                if dt.dropna().empty:
                    continue
                dt_min = dt.min()
                dt_max = dt.max()
                monthly = dt.dt.to_period("M").value_counts().sort_index()

                ser = monthly.copy()
                if len(ser) > 60:
                    ser = ser.tail(60)
                plt.figure(figsize=(10, 3.5))
                plt.plot([str(p) for p in ser.index], ser.values)
                plt.title(col)
                plt.xticks(rotation=60, ha="right", fontsize=8)
                plt.ylabel("count")
                plt.tight_layout()
                buff = BytesIO()
                plt.savefig(buff, bbox_inches="tight", format="png")
                plt.close()
                trend_uri = self._img_to_data_uri(buff.getvalue())

                parts.append(
                    f"""
      <div class="col-card" id="{html.escape(col)}">
        <div class="col-title">{html.escape(col)}</div>
        <div class="muted">Range: {html.escape(str(dt_min))} – {html.escape(str(dt_max))}</div>
        <div class="plot"><img alt="Datetime trend {html.escape(col)}" src="{trend_uri}" /></div>
      </div>
"""
                )
            parts.append("</div></div>")

        parts.append("<div class='section'><h2>Numeric</h2><div class='cols'>")

        for col in numerical_columns:
            hist_uri = self._plot_data_uri(df[col], kind="hist")
            box_uri = self._plot_data_uri(df[col], kind="box")

            hist_html = (
                f"<img alt='Histogram {html.escape(col)}' src='{hist_uri}' />"
                if hist_uri
                else "<div class='muted'>No values.</div>"
            )
            box_html = (
                f"<img alt='Boxplot {html.escape(col)}' src='{box_uri}' />"
                if box_uri
                else "<div class='muted'>No values.</div>"
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
          <label for="comment-{html.escape(col)}">Comment</label>
          <textarea id="comment-{html.escape(col)}" placeholder=""></textarea>
        </div>
      </div>
"""
            )

        parts.append("</div></div>")

        parts.append("<div class='section'><h2>Categorical</h2><div class='cols'>")
        for col in categorical_columns:
            bar_uri = self._categorical_bar_uri(df, col, top_n=10)
            bar_html = (
                f"<div class='plot'><img alt='Value counts {html.escape(col)}' src='{bar_uri}' /></div>"
                if bar_uri
                else ""
            )
            parts.append(
                f"""
      <div class="col-card" id="{html.escape(col)}">
        <div class="col-title">{html.escape(col)}</div>
        {bar_html}
        {self._categorical_stats_table_html(df, col)}
        <div class="comment-box">
          <label for="comment-{html.escape(col)}">Comment</label>
          <textarea id="comment-{html.escape(col)}" placeholder=""></textarea>
        </div>
      </div>
"""
            )
        parts.append("</div></div>")

        if corr_pearson is not None:
            heat_uri, top_pos_html, top_neg_html = self._corr_heatmap_and_extremes(corr_pearson)
            parts.append("<div class='section'><h2>Correlations</h2>")
            if heat_uri:
                parts.append(f"<div class='plot'><img alt='Correlation heatmap' src='{heat_uri}' /></div>")
                parts.append("<div style='height:12px'></div>")
            parts.append(top_pos_html)
            parts.append("<div style='height:12px'></div>")
            parts.append(top_neg_html)
            parts.append("</div>")

        parts.append(
            """
<div class="section">
  <h2>Notes</h2>
  <textarea id="general-comment" style="width:100%; min-height:120px; padding:10px; border-radius:10px; border:1px solid #243044; background:rgba(0,0,0,0.2); color:#e5e7eb; font-size:13px; font-family:inherit; resize:vertical; box-sizing:border-box" placeholder=""></textarea>
</div>
"""
        )

        parts.append(
            """
<script>
function restoreSavedComments() {
  const textareas = document.querySelectorAll("textarea[data-saved-value]");
  for (const ta of textareas) {
    const v = ta.getAttribute("data-saved-value");
    if (v !== null) ta.value = v;
  }
}

function saveReport() {
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

function sortOverview(colIdx) {
  const table = document.getElementById("column-overview-table");
  if (!table) return;
  const rows = Array.from(table.querySelectorAll("tr")).slice(1);
  const currentCol = table.dataset.sortCol ? parseInt(table.dataset.sortCol, 10) : -1;
  const currentDir = table.dataset.sortDir || "asc";
  const nextDir = currentCol === colIdx && currentDir === "asc" ? "desc" : "asc";
  table.dataset.sortCol = String(colIdx);
  table.dataset.sortDir = nextDir;

  rows.sort((ra, rb) => {
    const aCell = ra.children[colIdx];
    const bCell = rb.children[colIdx];
    const aRaw = (aCell?.getAttribute("data-sort") ?? aCell?.innerText ?? "").trim();
    const bRaw = (bCell?.getAttribute("data-sort") ?? bCell?.innerText ?? "").trim();
    const aNum = Number(aRaw);
    const bNum = Number(bRaw);
    let cmp = 0;
    if (!Number.isNaN(aNum) && !Number.isNaN(bNum)) {
      cmp = aNum - bNum;
    } else {
      cmp = aRaw.localeCompare(bRaw, undefined, { sensitivity: "base" });
    }
    return nextDir === "asc" ? cmp : -cmp;
  });

  for (const r of rows) table.appendChild(r);
}

function filterOverview(colIdx) {
  const table = document.getElementById("column-overview-table");
  if (!table) return;
  const query = prompt("Filter (empty = show all):", "");
  const rows = Array.from(table.querySelectorAll("tr")).slice(1);
  if (query === null || query.trim() === "") {
    for (const r of rows) r.style.display = "";
    return;
  }
  const q = query.trim().toLowerCase();
  for (const r of rows) {
    const text = (r.children[colIdx]?.innerText ?? "").toLowerCase();
    r.style.display = text.includes(q) ? "" : "none";
  }
}

restoreSavedComments();
</script>
"""
        )
        parts.append("</div></body></html>")

        report_path = self.output_dir / report_name
        report_path.write_text("".join(parts), encoding="utf-8")
        return report_path

