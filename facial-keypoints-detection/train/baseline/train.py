import torch


def masked_mse_loss(outputs, targets, mask):
    valid = mask.bool()
    if valid.sum() == 0:
        return outputs.new_tensor(0.0)
    return torch.nn.functional.mse_loss(outputs[valid], targets[valid])


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    lr_scheduler=None,
    logger=None,
):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_count = 0.0
        train_losses = []
        for images, keypoints, mask in train_loader:
            images, keypoints = images.to(device), keypoints.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = masked_mse_loss(outputs, keypoints, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_count += images.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        val_count = 0.0
        val_losses = []
        val_accuracies = []
        with torch.no_grad():
            for images, keypoints, mask in val_loader:
                images, keypoints = images.to(device), keypoints.to(device)
                mask = mask.to(device)
                outputs = model(images)
                loss = masked_mse_loss(outputs, keypoints, mask)
                val_loss += loss.item() * images.size(0)
                val_count += images.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f},Val Loss: {val_loss:.4f}"
        )
        if logger is not None:
            logger.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )
        if lr_scheduler:
            lr_scheduler.step(train_loss)
            lr_scheduler.optimizer.zero_grad()
    return train_losses, val_losses
