# Import necessary libraries
import torch
import torch.nn as nn
# Import AGCRNLayer from the local module
from AGCRNLayer import AGCRNLayer

# Define AGCRN class, which inherits from PyTorch's nn.Module


class AGCRN(nn.Module):
    # Initialize AGCRN with necessary parameters
    def __init__(self, num_node, input_dim, hidden_dim, output_dim, horizon, num_layers, embed_dim, k):
        # Call the parent class constructor
        super().__init__()

        # Set up AGCRN class attributes
        self.num_node = num_node
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers

        # Initialize node embeddings as a learnable parameter
        self.node_embeddings = nn.Parameter(torch.randn(num_node, embed_dim))

        # Create a list of AGCRN layers using the provided parameters
        self.dcrnns = nn.ModuleList(
            [AGCRNLayer(num_node, input_dim, hidden_dim, embed_dim, k)])
        for _ in range(num_layers - 1):
            self.dcrnns.append(AGCRNLayer(
                num_node, hidden_dim, hidden_dim, embed_dim, k))

        # Define the final convolutional layer
        self.conv = nn.Conv2d(1, horizon * output_dim,
                              kernel_size=(1, hidden_dim))

    # Define the forward pass of the AGCRN model
    def forward(self, x):
        out = x
        # Pass the input through each AGCRN layer
        for i in range(self.num_layers):
            # Initialize the current state with zeros
            current_state = torch.zeros(
                x.shape[0], self.num_node, self.hidden_dim)

            # Initialize a list to store inner states
            inner_states = []

            # Loop through the time dimension of the input
            for t in range(x.shape[1]):
                # Update the current state using the AGCRN layer
                current_state = self.dcrnns[i](
                    out[:, t, :, :], current_state, self.node_embeddings)

                # Append the current state to the inner states list
                inner_states.append(current_state)

            # Stack the inner states along the time dimension
            out = torch.stack(inner_states, dim=1)

        # Apply the final convolutional layer
        out = self.conv(out[:, -1].unsqueeze(1))

        # Reshape the output and transpose dimensions for the final output
        out = out.view(-1, self.horizon, self.output_dim,
                       self.num_node).transpose(2, 3)

        return out

    # Initialize the model parameters using Xavier uniform initialization
    def init_parameters(self):
        for _, param in self.named_parameters():
            if len(param.shape) != 1:
                nn.init.xavier_uniform_(param)
