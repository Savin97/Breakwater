# main.py
from pipeline.pipeline import run_pipeline
from report.report_builder import generate_report

def main():
    print("--------------------\nRunning pipeline...\n--------------------\n")
    generate_report()
    #run_pipeline()    
    print("Pipeline execution completed.\n--------------------")

if __name__ == "__main__":
    main()
