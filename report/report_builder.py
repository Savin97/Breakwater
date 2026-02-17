# report/report_builder.py
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from pathlib import Path

def generate_report():
    project_root = Path(__file__).resolve().parents[1]
    env = Environment(loader=FileSystemLoader("report/templates"))
    template = env.get_template("earnings_report.html")

    html_out = template.render(
        stock = "AAPL",
        risk_level = "Mid",
        p_extreme = 0.1
    )

    HTML(string=html_out, base_url=project_root).write_pdf("output/report.pdf")
