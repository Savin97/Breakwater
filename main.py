# main.py
from pipeline.pipeline import run_pipeline
from db.db_testing_for_stage_1_overhaul import db_main
#from report.report_builder import generate_report

def main():
    print("--------------------\nRunning pipeline...\n--------------------\n")
    #db_main()
    #generate_report()
    #run_pipeline()    
    print("Pipeline execution completed.\n--------------------")

if __name__ == "__main__":
    main()
