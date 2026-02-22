# main.py
from pipeline.pipeline import run_pipeline
from db.db_main import db_main
from report.report_builder import generate_report
from data_ingestion.api_functions import get_earnings_data_from_api

def main():
    print("--------------------\nRunning pipeline...\n--------------------\n")
    #db_main()
    #generate_report()
    run_pipeline()
    print("Pipeline execution completed.\n--------------------")

if __name__ == "__main__":
    main()
