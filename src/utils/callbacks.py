class EarlyStopping:
    """
    Early stopping with a single learning rate reduction.
    After LR reduction, allows a final set of epochs before stopping.
    """

    def __init__(
        self, initial_patience=10, final_patience=5, verbose=False, delta=1e-7
    ):
        self.initial_patience = initial_patience
        self.final_patience = final_patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float("inf")
        self.delta = delta
        self.lr_reduced = False

    def __call__(self, val_loss):
        # print(f"\nCurrent val_loss: {val_loss:.6f}")
        # print(f"Best loss: {self.best_loss:.6f}")
        # print(f"Delta check: {val_loss} < {self.best_loss - self.delta}")

        # Check if we've improved
        if val_loss < self.best_loss - self.delta:
            # print("Loss improved!")
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"EarlyStopping counter: {self.counter} out of "
                    f'{"initial " + str(self.initial_patience) if not self.lr_reduced else str(self.final_patience)}'
                )

            # Check if we should reduce LR or stop
            if not self.lr_reduced and self.counter >= self.initial_patience:
                self.lr_reduced = True
                self.counter = 0
                print("Triggering learning rate reduction")
                return "reduce_lr"
            elif self.lr_reduced and self.counter >= self.final_patience:
                print("Early stopping triggered")
                return "stop"

        return "continue"

    def reset(self):
        # print("Resetting EarlyStopping")
        self.counter = 0
        self.best_loss = float("inf")
        self.lr_reduced = False


class LRScheduler:
    """Simple learning rate reducer with a single reduction."""

    def __init__(self, optimizer, factor=0.1):
        self.optimizer = optimizer
        self.factor = factor

    def step(self):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= self.factor
