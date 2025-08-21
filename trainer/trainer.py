# create classification trainer
# train the model
import torch, math, os
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics import F1Score

import logging
from tqdm import tqdm
import time
import dataclasses

torch.autograd.set_detect_anomaly(True)

# --- 1. Configure logging to work well with tqdm ---
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

@dataclasses.dataclass
class TrainConfig:
    report_path: str = "model/reports"
    epochs: int = 100
    num_warmup_steps: int = 5
    batch_size: int = 32
    num_workers: int = 0
    learning_rate: float = 3e-3
    weight_decay: float = 0
    checkpoints: int = 10 # the model every <checkpoints> epochs

# Create Cosine LR Scheduler
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """
    Create a schedule with a learning rate that increases linearly during warmup,
    then decreases following a cosine function.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # progress after warmup [0, 1]
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class ClassificationTrainer():
    def __init__(self, model, training_config, train_dataset, val_dataset, task = "binary", device = "cuda"):
        """This is the training function. Here are some of the parameters:
    
        1. model: your Pytorch model.
        2. train_dataset: Your dataset for training (pytorch.utils.data.Dataset).
        3. val_dataset: Your dataset for validation (pytorch.utils.data.Dataset).
        4. training_config: Fill the training config with the TrainConfig class.
        5. Task is either 'binary' or 'multiclass'
        """
        assert task in {"binary", "multiclass"}, "task only accept binary or multiclass"
        # define the training config
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.epochs = training_config.epochs
        self.batch_size = training_config.batch_size
        self.num_workers = training_config.num_workers
        self.checkpoints = training_config.checkpoints
        self.report_path = training_config.report_path
        self.task = task
        self.device = device

        # define the criterion
        if task == "binary":
            self.criterion = nn.BCEWithLogitsLoss().to(device)
        else:
            self.criterion = nn.CrossEntropyLoss().to(device)
        # define the optimizers
        self.optimizer = AdamW(model.parameters(), 
                               lr = training_config.learning_rate,
                               weight_decay = training_config.weight_decay)
        #define scheduler
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, training_config.num_warmup_steps, self.epochs)
        # evaluation metrics
        self.f1_score = F1Score(task = "multiclass", num_classes = len(val_dataset.classes)).to(device)

        # initiate logger
        self.log = self.logger()

    # compute f1_score
    def compute_f1_score(self, predicted, label):
        if self.task == "binary":
            predicted = F.sigmoid(predicted, dim = -1)
        else:
            predicted = torch.argmax(predicted, dim = -1)
        f1_score = self.f1_score(predicted, label)
        return f1_score

    # Create report plot
    def create_report_plot(self, loss_train, loss_val, f1_train, f1_val):
        x = [i for i in range(1, self.epochs + 1)]

        metrics = {"Loss": {"Train": loss_train,
                            "Val": loss_val},
               "F1_Score": {"Train": f1_train,
               "Val": f1_val}}

        # Create plot for each metric
        for key in metrics.keys():
            fig, ax = plt.subplots(figsize = (10, 10))

            for data in metrics[key].keys():
                sns.lineplot(x = x, y = metrics[key][data], label = data)
            plt.title(f"{key}")
            plt.xlabel("epochs")
            plt.ylabel(f"key")

            fig.savefig(f"{self.report_path}/plots/{key}.png", dpi = 600)

    #create logging
    def logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Add custom tqdm logging handler
        handler = TqdmLoggingHandler()
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # file handler
        os.makedirs(os.path.dirname(f"{self.report_path}/log.log"), exist_ok=True)
        file_handler = logging.FileHandler(f"{self.report_path}/log.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def train(self):
        # create dataloader
        train_loader = DataLoader(self.train_dataset, 
                                  batch_size = self.batch_size, 
                                  shuffle = True, 
                                  num_workers = self.num_workers,
                                  pin_memory = True if self.device == "cuda" else False)
        
        val_loader = DataLoader(self.val_dataset, 
                                batch_size = self.batch_size, 
                                num_workers = self.num_workers,
                                pin_memory = True if self.device == "cuda" else False)        
        
        loss_train = []
        loss_val = []

        f1_train = []
        f1_val = []
        self.log.info(f"""
 ||  ||  
 \\()// 
//(__)\\
||    ||
device = {self.device}
Trainable Parameters = {sum(p.numel() for p in self.model.parameters() if p.requires_grad):d} / {sum(p.numel() for p in self.model.parameters())}
Start Training 🧠...""")
        for epoch in range(self.epochs):
            epoch_train_loss, epoch_val_loss = 0, 0
            epoch_train_f1, epoch_val_f1 = 0, 0
            start_time = time.time()
            # Add tqdm wrapper
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            # train the model
            self.model.train()
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                # create prediction
                _y = self.model(x)

                # compute loss
                loss = self.criterion(_y, y)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # add the training loss
                epoch_train_loss += loss.item()
                # add the f1 training
                f1 = self.compute_f1_score(_y, y.long())
                epoch_train_f1 += f1.item()
                pbar.set_postfix(train_loss=loss.item(), train_f1_score = f1.item())

            # compute the validation set
            self.model.eval()
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                _y = self.model(x) 
                loss = self.criterion(_y, y)

                # add the validation loss
                epoch_val_loss += loss.item()
                # add the f1 validation
                f1 = self.compute_f1_score(_y, y)
                epoch_val_f1 += f1.item()

            # compute the average of loss and metrics in one epoch
            avg_epoch_train_loss = epoch_train_loss / len(train_loader)
            avg_epoch_val_loss = epoch_val_loss / len(val_loader)
            avg_epoch_train_f1 = epoch_train_f1 / len(train_loader)
            avg_epoch_val_f1 = epoch_val_f1 / len(val_loader)

            # append it to the log
            loss_train.append(avg_epoch_train_loss)
            loss_val.append(avg_epoch_val_loss)
            f1_train.append(avg_epoch_train_f1)
            f1_val.append(avg_epoch_val_f1)

            # Display validation loss after tqdm loop ends
            pbar.set_postfix(train_loss = avg_epoch_train_loss, val_loss = avg_epoch_val_loss, 
                             train_f1_score = avg_epoch_train_f1, val_f1_score = avg_epoch_val_f1,
                             duration = start_time - time.time())
            self.log.info(str(pbar))

            # checkpoints
            if epoch + 1 % self.checkpoints == 0:
                torch.save(self.model.state_dict(), 
                           f"{self.report_path}/checkpoints/_{epoch//self.checkpoints}.pth") 
        
        # Create plot after training
        self.create_report_plot(loss_train, loss_val, f1_train, f1_val)
