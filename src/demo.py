"""
Demo file for showing process pipeline end-to-end
"""
from optimizer_parallel import Optimizer
if __name__ == "__main__":
    optimizer = Optimizer({},"/")
    optimizer.coarse_optimize(n_steps = 10_000, device="cuda", lambda_consensus = 12.0)
    pass
