
# This code defines the AGCRNLayer class, which is a single layer used in the AGCRN model.
# The layer calculates the hidden states for each node in the graph and updates the states
# using Chebyshev polynomials and gated recurrent unit (GRU)-like mechanisms.
# The forward pass of the AGCRNLayer calculates the normalized adjacency matrix, reset and update gates, and the candidate hidden state.
# Finally, it updates the hidden state using the update gate.

# The init_hidden_state function initializes the hidden state with zeros.


# Import necessary libraries
import torch
import torch.nn as nn

# Define AGCRNLayer class, which inherits from PyTorch's nn.Module


class AGCRNLayer(nn.Module):
    # Initialize AGCRNLayer with necessary parameters
    def __init__(self, n, dim_in, dim_out, embed_dim, k):
        # Call the parent class constructor
        super().__init__()

        # Set up AGCRNLayer class attributes
        self.node_num = n
        self.hidden_dim = dim_out
        self.dim_out = dim_out

        # Initialize learnable parameters for the reset and update gates
        self.W_reset = nn.Parameter(torch.empty(
            embed_dim, k, dim_in + dim_out, 2 * dim_out))
        self.W_update = nn.Parameter(torch.empty(
            embed_dim, k, dim_in + dim_out, dim_out))
        self.b_reset = nn.Parameter(torch.empty(embed_dim, 2 * dim_out))
        self.b_update = nn.Parameter(torch.empty(embed_dim, dim_out))

        # Activation functions
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    # Define the forward pass of the AGCRN layer
    def forward(self, x, state, node_embeddings):
        # Ensure the input state is on the same device as the input
        device = x.device
        state = state.to(device)

        # Compute the normalized adjacency matrix A_tilde
        A_tilde = self.softmax(
            self.relu(torch.matmul(node_embeddings, node_embeddings.t())))

        # Calculate the reset and update gates using Chebyshev polynomials
        S_reset = [torch.eye(node_embeddings.shape[0], device=device), A_tilde]
        for _ in range(self.W_reset.shape[1] - 2):
            S_reset.append(
                2 * torch.matmul(A_tilde, S_reset[-1]) - S_reset[-2])
        S_reset = torch.stack(S_reset)

        xg_reset = torch.matmul(S_reset.unsqueeze(1), torch.cat(
            (x, state), dim=2)).permute(1, 2, 0, 3)
        W_reset_new = torch.matmul(
            node_embeddings, self.W_reset.transpose(0, 2)).transpose(0, 2)
        b_reset_new = torch.matmul(node_embeddings, self.b_reset)


         # Compute the reset and update gates using the element-wise product of the
        # node_embeddings and the corresponding weights (W_reset_new) and biases (b_reset_new)
        # The einsum function is used to calculate this product efficiently
        # 'bnki,nkio->bno' is the Einstein summation notation, which defines how the tensors
        # are multiplied and summed. In this case, it means:
        #   b: batch size
        #   n: number of nodes
        #   k: polynomial order (Chebyshev)
        #   i: input channels
        #   o: output channels
        # The result is a tensor of shape (batch_size, num_nodes, 2 * dim_out)
        z_r = torch.sigmoid(torch.einsum(
            'bnki,nkio->bno', xg_reset, W_reset_new) + b_reset_new)
        z, r = torch.split(z_r, self.dim_out, dim=-1)

        # Compute the candidate hidden state
        candidate = torch.cat((x, r * state), dim=-1)

        S_update = [torch.eye(node_embeddings.shape[0],
                              device=device), A_tilde]
        for _ in range(self.W_update.shape[1] - 2):
            S_update.append(
                2 * torch.matmul(A_tilde, S_update[-1]) - S_update[-2])
        S_update = torch.stack(S_update)

        xg_update = torch.matmul(S_update.unsqueeze(
            1), candidate).permute(1, 2, 0, 3)
        W_update_new = torch.matmul(
            node_embeddings, self.W_update.transpose(0, 2)).transpose(0, 2)
        b_update_new = torch.matmul(node_embeddings, self.b_update)


        # Compute the candidate hidden state using the element-wise product of the
        # node_embeddings and the corresponding weights (W_update_new) and biases (b_update_new)
        # The einsum function is used to calculate this product efficiently
        # 'bnki,nkio->bno' is the Einstein summation notation, which defines how the tensors
        # are multiplied and summed. In this case, it means:
        #   b: batch size
        #   n: number of nodes
        #   k: polynomial order (Chebyshev)
        #   i: input channels
        #   o: output channels
        # The result is a tensor of shape (batch_size, num_nodes, dim_out)
        hc = torch.tanh(torch.einsum('bnki,nkio->bno',
                        xg_update, W_update_new) + b_update_new)

        # Return the updated hidden state using the update gate
        return z * state + (1 - z) * hc

    # Initialize the hidden state with zeros
    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


