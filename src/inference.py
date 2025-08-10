import torch
from model import FCNNClassifier
from dataset import get_dataloader
from parallel import (
    data_parallel_inference,
    ModelPart1,
    ModelPart2,
    model_parallel_inference,
    calculate_accuracy,
    save_metrics_csv
)
import time

def main(parallel_mode='data', batch_size=32, device='cpu', num_classes=10):
    dataloader = get_dataloader(batch_size)
    total_samples = len(dataloader.dataset)

    if parallel_mode == 'data':
        model = FCNNClassifier(num_classes=num_classes)
        t0 = time.time()
        outputs, latencies = data_parallel_inference(model, dataloader, device)
        t1 = time.time()
    elif parallel_mode == 'model':
        m1 = ModelPart1()
        m2 = ModelPart2(num_classes)
        t0 = time.time()
        outputs, latencies = model_parallel_inference(m1, m2, dataloader, device, device)
        t1 = time.time()
    else:
        raise ValueError("Nije podržan režim!")

    # Za točnost na dummy podacima
    labels = torch.cat([y for _, y in dataloader])
    accuracy = calculate_accuracy(outputs, labels)
    total_time = t1 - t0
    throughput = total_samples / total_time

    print(f"Ukupno vrijeme: {total_time:.2f}s")
    print(f"Prosječna latencija: {sum(latencies)/len(latencies):.4f}s")
    print(f"Propusnost: {throughput:.2f} primjera/s")
    print(f"Točnost (na dummy podacima): {accuracy:.2%}")

    save_metrics_csv(
        f"results_{parallel_mode}_parallel.csv",
        latencies,
        throughput,
        accuracy
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--parallel', type=str, default='data', choices=['data', 'model'])
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    main(parallel_mode=args.parallel, batch_size=args.batch_size)

