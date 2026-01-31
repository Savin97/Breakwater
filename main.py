# main.py
from pipeline.pipeline import run_pipeline

def main():
    print("--------------------\nRunning pipeline...\n--------------------\n")
    run_pipeline()    
    #tweak
    print("Pipeline execution completed.\n--------------------")
    
if __name__ == "__main__":
    main()
