from argparse import ArgumentParser
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from ignite.engine import Engine, Events

class MyDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(500,10)
        self.output = torch.randn(500,10)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx], self.output[idx]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        return self.fc3(self.fc2(self.fc1(x)))

def create_summary_writer(model, dataloader, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    x, y =  next(iter(dataloader))
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer

def run(args):

    my_dataset = MyDataset()
    dataloader = DataLoader(my_dataset, batch_size=4,
                        shuffle=True, num_workers=1)
    model = Net()
    optimizer = SGD(model.parameters(), lr=args.lr)
    writer = create_summary_writer(model, dataloader, args.log_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = args.epochs

    def train_step(engine, batch):
        optimizer.zero_grad()
        data, output = batch 
        criterion = nn.MSELoss()
        loss = criterion(output, model(data))
        O = loss.item()
        loss.backward()
        optimizer.step()

        return O 

    trainer = Engine(train_step)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_loss(engine):
        if engine.state.iteration%args.log_interval != 0:
            return
        print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
              "".format(engine.state.epoch, engine.state.iteration, len(dataloader), engine.state.output))
        writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def model_save(engine):
        os.makedirs('run/', exist_ok=True)
        print('EPOCH_COMPLETED, DO SOMETHING LIKE SAVE MODEL')
        torch.save({'model':model.state_dict()}, 'run/checkpoint.pth')

    @trainer.on(Events.EPOCH_COMPLETED)
    def act(engine):
        print('EPOCH_COMPLETED, DO SOMETHING ELSE')

    trainer.run(dataloader, args.epochs)

    writer.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--log_dir", type=str, default='mylogs',
                        help="log directory for Tensorboard log output")

    args = parser.parse_args()

    run(args)

