import numpy as np
import matplotlib.pyplot as plt
import torch


# Assuming positional_embedding is your tensor of shape (num_tokens, 128)
# And num_time_windows and num_channels are the desired dimensions of the grid

def viz_positiional_embedding(positional_embedding, num_time_windows=9, num_channels=16):
    # Calculate the number of tokens per grid
    tokens_per_grid = 16 * 8

    # Reshape the positional_embedding into a grid format
    pos_embedding_grid = positional_embedding.reshape(num_time_windows, num_channels, tokens_per_grid, -1)

    # Create the grid of subplots
    fig, axs = plt.subplots(num_time_windows, num_channels, figsize=(num_channels, num_time_windows))

    # Iterate through each time window and channel
    for i in range(num_time_windows):
        for j in range(num_channels):
            # Plot the positional embedding as an image
            axs[i, j].imshow(pos_embedding_grid[i, j], cmap='viridis')
            axs[i, j].axis('off')

    plt.tight_layout()
    plt.show()

def viz_time_positional_embedding(positional_embedding, reshape_to=(16, 8)):
    if isinstance(positional_embedding, torch.Tensor):
        positional_embedding = positional_embedding.detach().cpu().numpy()
    num_batches, pf_dim, embed_size = positional_embedding.shape
    assert np.product(reshape_to) == embed_size, f"reshape_to {reshape_to} does not match embed_size {embed_size}"
    fig, axs = plt.subplots(num_batches, pf_dim, figsize=(pf_dim, num_batches))

    for i in range(num_batches):
        for j in range(pf_dim):
            # Plot the positional embedding as an image
            axs[i*num_batches + j].imshow(positional_embedding[i, j].reshape(reshape_to), cmap='viridis')
            axs[i*num_batches + j].axis('off')
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, num_batches, squeeze=False)
    for i in range(num_batches):
        for j in range(pf_dim):
            # Plot the positional embedding as an image
            axs[i][0].plot(positional_embedding[i, j, :], label=f"PF {j}")
            axs[i][0].set_xlabel(f"Embedding index")
    plt.tight_layout()
    plt.show()
