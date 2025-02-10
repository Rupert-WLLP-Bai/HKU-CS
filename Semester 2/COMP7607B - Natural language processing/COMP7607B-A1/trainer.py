import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os


def train_model(
    model: torch.nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    model_save_path: str = "checkpoints",
):
    """
    Trains the Word2Vec model.

    Args:
        model: The Word2Vec model (SkipGram).
        dataloader: DataLoader for the training data.
        optimizer: Optimizer (e.g., Adam).
        num_epochs: Number of training epochs.
        device: Device to train on (CPU or GPU).
        model_save_path: Path to save the trained model.
    """
    # Move the model to the specified device (CPU or GPU)
    model.to(device)

    # Train the model for the specified number of epochs
    for epoch in range(num_epochs):
        # Initialize the total loss for this epoch
        total_loss = 0

        # Start the timer for this epoch
        start_time = time.time()

        # Iterate over the training data
        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Zero out the gradients
            optimizer.zero_grad()

            # Move the inputs and targets to the device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            output = model(inputs)

            # Calculate the loss
            loss = F.cross_entropy(output, targets)

            # Backward pass
            loss.backward()

            # Update the model parameters
            optimizer.step()

            # Add the loss to the total loss for this epoch
            total_loss += loss.item()

        # Calculate the time taken for this epoch
        end_time = time.time()
        epoch_time = end_time - start_time

        # Print the loss and time taken for this epoch
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}, Time: {epoch_time:.2f}s"
        )

    # Save the model after last epoch
    # Create the directory if it does not exist
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(model_save_path, f"model_epoch_{num_epochs}.pth"),
    )
