from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask, make_response

from column_profiler import profile_columns
from data_loader import load_csv
from numerical_analyzer import NumericalAnalyzer
from report_generator import ReportGenerator


def _build_report_html(df, *, base_dir: Path) -> str:
    col_profile = profile_columns(df)
    numerical_cols = col_profile.loc[col_profile["role"] == "numerical", "column"].tolist()
    categorical_cols = col_profile.loc[col_profile["role"] == "categorical", "column"].tolist()
    categorical_cols += col_profile.loc[col_profile["role"] == "boolean", "column"].tolist()
    datetime_cols = col_profile.loc[col_profile["role"] == "datetime", "column"].tolist()

    analyzer = NumericalAnalyzer(df, numerical_cols)
    summary = analyzer.summarize()
    corr_pearson = analyzer.correlations(method="pearson")

    report = ReportGenerator(output_dir=base_dir)
    path = report.write_html_report(
        df=df,
        numerical_columns=numerical_cols,
        categorical_columns=categorical_cols,
        datetime_columns=datetime_cols,
        column_profile=col_profile,
        summary=summary,
        corr_pearson=corr_pearson,
        report_name="EDA_report.html",
    )
    return path.read_text(encoding="utf-8")


def create_app(*, base_dir: Path):
    app = Flask(__name__)

    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    app.logger.disabled = True

    data_path = base_dir / "data" / "DrugSalesData.csv"
    df_original = load_csv(data_path)

    @app.get("/")
    def index():
        html = _build_report_html(df_original, base_dir=base_dir)
        resp = make_response(html)
        resp.mimetype = "text/html"
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        return resp

    return app
