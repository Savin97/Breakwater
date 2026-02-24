# report/report_builder.py
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from pathlib import Path

def generate_report(stock, data):
    project_root = Path(__file__).resolve().parents[1]
    env = Environment(loader=FileSystemLoader("report/templates"))
    template = env.get_template("earnings_report.html")

    # Cover Page
    html_out = template.render(
        stock = stock,
        earnings_date = earnings_date,
        risk_level = risk_level, # (Low / Moderate / Elevated / Extreme)
        tail_risk_score = risk_score, #(0-100)
        hist_xtreme_prob = hist_xtreme_prob,
        base_xtreme_prob = base_xtreme_prob,
        risk_lift = risk_lift
    )
    # Executive Summary (1 page)
    # Plain English:
    # “This stock has structurally elevated earnings jump risk.”
    # “Top 5% of earnings tail risk across coverage universe.”
    # “Historically, extreme 3-day reactions occurred in 38% of similar regimes.”

    REPORT_OUTPUT_PATH = "output/report.pdf"
    HTML(string=html_out, base_url=project_root).write_pdf(REPORT_OUTPUT_PATH)
    print(f"Report created in {REPORT_OUTPUT_PATH}")
