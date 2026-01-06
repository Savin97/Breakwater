import pipeline.pipeline_stage1

def run_pipeline():
    print("Running the pipeline...")
    stock_prices = pipeline.pipeline_stage1.stage1()
    return stock_prices