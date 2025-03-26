class ModelTrainingStats():
    def __init__(self, training_loss, val_loss, best_loss, epoch, best_epoch):
        self.training_loss = training_loss
        self.validation_loss = val_loss
        self.best_loss = best_loss
        self.epoch = epoch
        self.best_epoch = best_epoch
        