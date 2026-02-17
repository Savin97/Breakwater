# pipeline/output.py

from config import cols_to_drop_for_output

def output_to_csv(*args, **kwargs):
    inputs_df = args[0]
    feature_engineering = args[1]
    scoring_df = args[2]
    backtesting_df = args[3]

    inputs_df.to_csv("output/stage_1_df.csv", index=False)
    # print("----------------\nStage 1 DF created in: output/stage_1_df.csv")

    feature_engineering.to_csv("output/stage_2_df.csv", index=False)
    # print("----------------\nStage 2 DF created in: output/stage_2_df.csv")

    scoring_df.to_csv("output/stage_3_df.csv", index=False)
    print("----------------\nStage 3 DF created in: output/stage_3_df.csv")

    # TODO: only keeping earnings days so far
    #backtesting_df = backtesting_df[backtesting_df["is_earnings_day"] == 1]

    #backtesting_df = backtesting_df.drop(columns=cols_to_drop_for_output)
    backtesting_df.to_csv("output/backtesting_df.csv", index=False)
    print("----------------\nBack testing DF created in: output/backtesting_df.csv")

    # trimmed_df = scoring_df.drop(columns=cols_to_drop_for_output)
    # trimmed_df.to_csv("output/trimmed_df.csv", index=False)
    # print("----------------\ntrimmed_df created in: output/trimmed_df.csv", "\n----------------")
