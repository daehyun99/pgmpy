from pgmpy.base import DAG
from collections import defaultdict

class BayesianNetwork(DAG):
    def __init__(self, ebunch=None):
        super().__init__()
        if ebunch:
            self.add_edges_from(ebunch)
        self.cpds = []
        self.cardinalities = defaultdict(int)

    def add_cpds(self, *cpds):


    def get_cpds(self, ):

    def check_model(self) -> bool:

    