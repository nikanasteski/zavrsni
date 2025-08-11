import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dataset import get_dataloader
from model import FCNNClassifier
from parallel import (
    data_parallel_inference,
    ModelPart1,
    ModelPart2,
    model_parallel_inference,
    calculate_accuracy,
    save_metrics_csv
)
import time

def distributed_main(rank, world_size, args):
    # Inicijalizacija distribuiranog procesa
    dist.init_process_group(
        backend='gloo',
        init_method=f'tcp://{args.master_addr}:{args.master_port}',
        rank=rank,
        world_size=world_size
    )

    # Priprema dataloadera
    # Možemo podijeliti podatke po procesima ako trebaš (ovisno o implementaciji u get_dataloader)
    dataloader = get_dataloader(args.batch_size)

    total_samples = len(dataloader.dataset)

    if args.parallel == 'data':
        model = FCNNClassifier(num_classes=10)
        t0 = time.time()
        # Pokreni data parallel inference na ovom procesu
        outputs, latencies = data_parallel_inference(model, dataloader, device='cpu')
        t1 = time.time()

    elif args.parallel == 'model':
        m1 = ModelPart1()
        m2 = ModelPart2(num_classes=10)
        t0 = time.time()
        outputs, latencies = model_parallel_inference(m1, m2, dataloader, device='cpu', device2='cpu')
        t1 = time.time()

    else:
        raise ValueError("Nepoznat paralelni način")

    # Prikupljanje labela i točnosti lokalno
    labels = torch.cat([y for _, y in dataloader])
    accuracy = calculate_accuracy(outputs, labels)
    total_time = t1 - t0
    throughput = total_samples / total_time

    print(f"[Proces {rank}] Ukupno vrijeme: {total_time:.2f}s")
    print(f"[Proces {rank}] Prosječna latencija: {sum(latencies)/len(latencies):.4f}s")
    print(f"[Proces {rank}] Propusnost: {throughput:.2f} primjera/s")
    print(f"[Proces {rank}] Točnost (dummy podaci): {accuracy:.2%}")

    # Spremi rezultate u datoteke (ako ti odgovara da svaki proces zapisuje svoj file)
    save_metrics_csv(
        f"results_{args.parallel}_parallel_rank{rank}.csv",
        latencies,
        throughput,
        accuracy
    )

    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parallel', type=str, default='data', choices=['data', 'model'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--master_addr', type=str, default='127.0.0.1')
    parser.add_argument('--master_port', type=str, default='12355')
    args = parser.parse_args()
