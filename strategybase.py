class StrategyBase:
    """
        StrategyBase class will the following properties:
        - name: name of the strategy
        - description: description of the strategy
        - factors (list): list of factors that the strategy uses
        - factors_weights (dict): dictionary of factors and their weights
        - results (dict): dictionary of results

        It will contain the following methods:
        - __init__: initializes the StrategyBase class
        - run: runs the strategy
        - get_results: returns the results of the strategy
        - get_metadata (description, factors, weights): returns the metadata of the strategy
    """
    def __init__(self, name, description, factors, factors_weights):
        self.name = name
        self.description = description
        self.factors = factors
        self.factors_weights = factors_weights
        self.results = {}

    def run(self):
        raise NotImplementedError

    def get_results(self):
        return self.results

    def get_metadata(self):
        return {
            "description": self.description,
            "factors": self.factors,
            "factors_weights": self.factors_weights
        }