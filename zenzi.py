from strategybase import StrategyBase

class Zenzi(StrategyBase):
    def __init__(self):
        super().__init__()

    # Add your Zenzi-specific methods and attributes here



if __name__ == "__main__":
    zenzi = Zenzi(name='Zenzi', 
                  description='Zenzi combines altman (50%), johnfkennedy (50%) '
                  'and uses farooqi to decide how many stocks to buy. '
                  'Stocks are chosen by matching buy signals to the S&P500.', 
                  factors=['farooqi', 'altman', 'johnfkennedy'], 
                  factors_weights={'factor1': 0.5, 'factor2': 0.5}
                  )
    zenzi.run()
    print(zenzi.get_results())
    print(zenzi.get_metadata())