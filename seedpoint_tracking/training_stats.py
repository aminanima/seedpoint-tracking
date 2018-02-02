class TrainingStats(object):
  def __init__(self):
    self.train_batches = 0
    self.val_batches = 0
    self.epoch = 0

  def increment_train_batches(self):
    self.train_batches += 1

  def get_train_batches(self):
    return self.train_batches

  def reset_train_batches(self):
    self.train_batches = 0

  def increment_val_batch(self):
    self.val_batches += 1

  def get_val_batches(self):
    return self.val_batches

  def reset_val_batches(self):
    self.val_batches = 0

  def increment_epochs(self):
    self.epoch += 1

  def get_epoch(self):
    return self.epoch
