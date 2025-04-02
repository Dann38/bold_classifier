import torch
import torch.nn as nn

class BoldTask(nn.Module):
    def __init__(self):
        super(BoldTask, self).__init__()
        self.fc1 = nn.Linear(8, 8)  
        self.fc2 = nn.Linear(8, 1)  
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))  # (batch_size, 8)
        x = self.fc2(x)  # (batch_size, 1)
        return x


class SubCharCNNClassifier(nn.Module):
    def __init__(self):
        super(SubCharCNNClassifier, self).__init__()
        
        # сверточные слои
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # максимальный пулинг
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # полносвязный слой
        self.fc1 = nn.Linear(32 * 10 * 10, 64)
        self.fc2 = nn.Linear(64, 8)

        # функция активации
        self.relu = nn.ReLU()

    def forward(self, x):
        # применяем свертки и пулинг
        x = self.pool(self.relu(self.conv1(x)))  # (batch_size, 16, 30, 120)
        x = self.pool(self.relu(self.conv2(x)))  # (batch_size, 32, 15, 60)

        # делаем вектор одномерным для fc1
        x = x.view(x.size(0), -1)  # (batch_size, 32 * 15 * 60)

        # применяем полносвязный слой и relu
        x = self.relu(self.fc1(x))  # (batch_size, 128)
        x = self.relu(self.fc2(x))  # (batch_size, 64)

        return x

def get_font_emb_model(path: str):
    model_font_emb = SubCharCNNClassifier()
    model_font_emb.load_state_dict(torch.load(path, weights_only=True))
    return model_font_emb

def get_font_bold_model(path: str):
    model_bold = BoldTask()
    model_bold.load_state_dict(torch.load(path, weights_only=True))
    return model_bold