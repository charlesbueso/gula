"""
A L T M A N    F A C T O R
    Uses LLM to: 
        1. Parse last 5 quarterly data
        2. Create metric summary
        3. Fetch latest news (30 days) 
        4. Prompt Engineering / Train of thought
            - Takes metric summary and latest news as input
        5. Output: 
            -1 <= X <= 1
             
            When -1 <= X <= -.65 sell
            When -.65 < X <= .65 hold
            When .65 < X <= 1 buy

            Strategy: Pick X (5, 10, 30) buy signals that are in S&P500 
                Buy equally weighted, rebalance every month

"""

