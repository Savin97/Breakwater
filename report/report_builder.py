# report/report_builder.py
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from weasyprint import HTML
from pathlib import Path

def generate_report(stock, data):
    project_root = Path(__file__).resolve().parents[1]
    env = Environment(
        loader=FileSystemLoader("report/templates"),
        undefined=StrictUndefined,
        autoescape=True,)
    template = env.get_template("earnings_report.html")

    # Cover Page
    html_out = template.render(
        stock = stock,
        earnings_date = data["earnings_date"],
        risk_level = data["risk_level"], # (Low / Moderate / Elevated / Extreme)
        risk_score = data["risk_score"], #(0-100)
        hist_extreme_prob = data["hist_extreme_prob"],
        base_extreme_prob = data["base_extreme_prob"],
        current_lift_vs_baseline = data["current_lift_vs_baseline"],
        current_lift_vs_same_bucket_global = data["current_lift_vs_same_bucket_global"],
        bucket_table = data["bucket_table"],
        sector = data["sector"],
        sub_sector = data["sub_sector"]
    )
    # Executive Summary (1 page)
    # Plain English:
    # "This stock has structurally elevated earnings jump risk."
    # "Top 5% of earnings tail risk across coverage universe."
    # "Historically, extreme 3-day reactions occurred in 38% of similar regimes."

    REPORT_OUTPUT_PATH = f"output/{stock}_report.pdf"
    HTML(string=html_out, base_url=project_root).write_pdf(REPORT_OUTPUT_PATH)
    print(f"Report created in {REPORT_OUTPUT_PATH}")
