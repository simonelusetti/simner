from itertools import product
from dora import Explorer, Launcher
from dora.conf import SlurmConfig

@Explorer
def explorer(launcher):
    for model_type in ["base","attention","transformer"]:
        launcher({"config.split":"train[:100]"},{"config.model_type":model_type})
