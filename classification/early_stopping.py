class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.005):
        """
        :param patience: Number of epochs with no improvement after which training will stop.
        :param min_delta: Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0  # Number of epochs with no improvement
        self.best_loss = float("inf")  # Best validation loss so far
        self.best_epoch = 0  # Epoch when the best validation loss occurred

    def should_stop(self, val_loss, current_epoch):
        """
        :param val_loss: Current validation loss.
        :param current_epoch: The current epoch number.
        :return: True if training should stop, False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = current_epoch
            self.counter = 0  # Reset counter as we have improved
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(f"Stopping early at epoch {current_epoch}, no improvement after {self.patience} epochs.")
            return True
        return False
