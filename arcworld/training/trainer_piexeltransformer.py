import torch
import torchmetrics
from torch import nn
from torch.utils.data import DataLoader

from arcworld.training.pixeltransformer import PixelTransformer
from arcworld.training.transformer_total_loader import TransformerOriginalDataset

if __name__ == "__main__":
    batch_size = 4
    device = torch.device("cuda")
    dataset = TransformerOriginalDataset(
        "/home/kevin/arcworld/examples/expansion/tasks"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = PixelTransformer().to(device)
    model.train()

    image_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=11).to(
        device
    )
    img_loss_metric = torchmetrics.MeanMetric().to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.0001)
    ce_loss = nn.CrossEntropyLoss(
        weight=torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.1], device=device).float()
    )
    binary_ce_loss = nn.BCEWithLogitsLoss()
    sigmoid = nn.Sigmoid()
    batch_nr = 1
    name = 0
    print("ready")
    loss_names = ["encoder", "mask"]
    for epoch in range(1):
        for (
            seq,
            original,
            target,
        ) in dataloader:
            seq_len = seq.shape[0]
            seq = seq.to(device)
            original = original.to(device)
            target = target.to(device)
            optimizer.zero_grad()

            output = model(seq, original)

            image_loss = ce_loss(output, target)

            image_accuracy_metric.update(output, target)

            img_loss_metric.update(image_loss)

            loss = image_loss
            loss.backward()
            optimizer.step()

            if batch_nr % 120 == 0:
                print(epoch, batch_nr * 16)

            batch_nr = batch_nr + 1
