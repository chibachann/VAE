class Config:
    def __init__(self):
        self.batch_size = 128
        self.epochs = 10
        self.learning_rate = 1e-3
        self.input_dim = 784
        self.hidden_dim = 400
        self.latent_dim = 20