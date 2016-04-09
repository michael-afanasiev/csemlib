import os


class CSEM(object):

    def __init__(self, root_dir):

        self.root_dir = root_dir
        self.solvers = os.listdir(self.root_dir)
        self.models = {solver: os.listdir(os.path.join(self.root_dir, solver)) for solver in self.solvers}
