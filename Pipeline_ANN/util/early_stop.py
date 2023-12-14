import torch

class EarlyStopper:
    '''
    function for Early Stopping of learning epochs 
    
    if loss value is not step until patience value 
    your machine will stop after patience
    
    class args:
            patience: patience value of step -> int
            min_delta: min change amount for early stop -> int
    '''
    def __init__(self, patience:int=3, min_delta:int=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, model, validation_loss:float, name='test.pth', mode=True):
        if validation_loss < self.min_validation_loss:
            if mode:
                torch.save(model.state_dict(), name)
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False