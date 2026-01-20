# stage2.py
""" Pipeline Stage 2
    Input is a df with columns
    stock  date  price_adj_close  earnings_date  fiscal_date_ending reported_eps  estimated_eps  surprise_percentage

    Feature Engineering - daily, ret_1d,ret_3d,ret_5d
"""