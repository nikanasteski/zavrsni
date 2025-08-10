import torch
import torch.nn as nn
import time

def data_parallel_inference(model, dataloader, device='cpu'):
    # Pravi DataParallel samo ako imaš GPU, inače simuliraj na CPU
    if torch.cuda.device_count() > 1 and device != 'cpu':
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()
    all_outputs, latencies = [], []
    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            start = time.time()
            out = model(batch_x)
            end = time.time()
            all_outputs.append(out.cpu())
            latencies.append(end - start)
    outputs = torch.cat(all_outputs)
    return outputs, latencies

class ModelPart1(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(384, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
    def forward(self, x):
        return self.seq(x)

class ModelPart2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.l2 = nn.Linear(768, num_classes)
    def forward(self, x):
        return self.l2(x)

def model_parallel_inference(model_part1, model_part2, dataloader, device1='cpu', device2='cpu'):
    model_part1 = model_part1.to(device1)
    model_part2 = model_part2.to(device2)
    model_part1.eval()
    model_part2.eval()
    all_outputs, latencies = [], []
    with torch.no_grad():
        for batch_x, _ in dataloader:
            x = batch_x.to(device1)
            start = time.time()
            x = model_part1(x)
            x = x.to(device2)
            out = model_part2(x)
            end = time.time()
            all_outputs.append(out.cpu())
            latencies.append(end - start)
    outputs = torch.cat(all_outputs)
    return outputs, latencies

# METRICS
def calculate_accuracy(outputs, labels):
    preds = outputs.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)

def save_metrics_csv(filename, latencies, throughput, accuracy):
    import csv
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['batch', 'latency'])  # batch index, batch latency
        for i, l in enumerate(latencies):
            writer.writerow([i, l])
        writer.writerow([])
        writer.writerow(['throughput', throughput])
        writer.writerow(['accuracy', accuracy])

