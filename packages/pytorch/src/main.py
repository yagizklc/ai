import torch
import torch.nn as nn

from src.config import config
from src.models.cnn import BasicCNN
from src.utils.loader import get_data
from src.utils.test_train import test, train

device = config.device


def main():
    print(f'PyTorch version: {torch.__version__}')
    print(f'MPS (Metal) device available: {torch.backends.mps.is_available()}')
    print(f'Using device: {device}')

    # data
    train_dataloader, test_dataloader = get_data()

    # model
    model = BasicCNN().to(device=device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 5
    for t in range(epochs):
        print(f'Epoch {t + 1}\n-------------------------------')
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print('Done!')

    torch.save(model.state_dict(), 'model.pth')
    print('Saved PyTorch Model State to model.pth')


def predict():
    classes = [
        'T-shirt/top',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle boot',
    ]

    model = BasicCNN().to(device)
    model.load_state_dict(torch.load('model.pth', weights_only=True))

    test_data = get_data()[1].dataset

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.to(config.device)
        pred: torch.Tensor = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')


if __name__ == '__main__':
    main()
    predict()
