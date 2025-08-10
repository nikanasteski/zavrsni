import torch
import torch.nn as nn

class FCNNClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(384, 768),         # Prvi sloj: ulaz 384 -> 768
            nn.BatchNorm1d(768),         # BatchNorm1d za stabilnost
            nn.ReLU(inplace=True),       # ReLU aktivacija
            nn.Dropout(0.25),            # Dropout za regularizaciju
            nn.Linear(768, num_classes)  # Izlazni sloj
        )

    def forward(self, x):
        return self.layers(x)


# Direktan test modela kada se ova datoteka pokreće samostalno
if __name__ == "__main__":
    model = FCNNClassifier(num_classes=10)
    dummy_input = torch.randn(8, 384)         # batch_size=8, feature_dim=384
    output = model(dummy_input)
    print("Output shape:", output.shape)      # Očekuje torch.Size([8, 10])
