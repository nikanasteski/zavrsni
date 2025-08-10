import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import FCNNClassifier
import torch

def test_fcnn_forward():
    # Testira forward pass i oblik izlaza
    model = FCNNClassifier(num_classes=10)
    X = torch.randn(32, 384)
    out = model(X)
    assert out.shape == (32, 10), f"Output shape is {out.shape}, expected (32, 10)"

def test_fcnn_loss():
    # Testira je li loss funkcija kompatibilna s izlazom modela
    model = FCNNClassifier(num_classes=10)
    dummy_features = torch.randn(32, 384)
    dummy_labels = torch.randint(0, 10, (32,))
    criterion = torch.nn.CrossEntropyLoss()
    outputs = model(dummy_features)
    loss = criterion(outputs, dummy_labels)
    print("Testni loss:", loss.item())
    assert isinstance(loss.item(), float), "Loss nije float!"

if __name__ == "__main__":
    test_fcnn_forward()
    print("FCNN model forward testiran - sve OK!")
    test_fcnn_loss()
    print("FCNN model radi i s loss funkcijom!")
