class CapitalInflowProcess:
    """
    This class models capital inflow into the Filecoin network.

    From @tmellan:
    Growth rate of FIL economy (market cap) between
    Gr = logistic(beta*X) - 0.01
    X = {QAP, QAP_grad, ROI%, locked/supply, token price, }
    """

    def __init__(self, filecoin_model):
        self.model = filecoin_model

    