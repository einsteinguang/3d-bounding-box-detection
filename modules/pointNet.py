import torch
import torch.nn as nn
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        # Convolutions analogous to Convolution1D(64,1)->(128,1)->(1024,1)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        # FC layers: 1024 -> 512 -> 256 -> 9
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        self.fc3.weight.data.zero_()
        self.fc3.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float))

    def forward(self, x):
        """
        x: (batch_size, 3, num_points)
        """
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))

        # Now pool over num_points (max pool):
        # out shape => (batch_size, 1024, 1)
        kernel_size = out.shape[-1].item()
        out = torch.nn.functional.max_pool1d(out, kernel_size)
        # Flatten => (batch_size, 1024)
        out = out.view(-1, 1024)

        # FC layers
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))

        # Output => shape (batch_size, 9)
        out = self.fc3(out)

        # Add the identity to bias the transform towards identity
        # Reshape to (batch_size, 3, 3)
        out = out.view(-1, 3, 3)
        return out


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        # Initialize fc3 so it starts as the identity transform
        self.fc3.weight.data.zero_()
        # flatten eye(k)
        eye = torch.eye(k, dtype=torch.float).view(-1)
        self.fc3.bias.data.copy_(eye)

    def forward(self, x):
        """
        x: (batch_size, k, num_points)
        """
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        kernel_size = out.shape[-1].item()
        out = torch.nn.functional.max_pool1d(out, kernel_size)
        out = out.view(-1, 1024)

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        out = out.view(-1, self.k, self.k)
        return out


class PointNet(nn.Module):
    """
    The overall PointNet that uses:
      1) STN3d (input transform) => 3x3
      2) MLP on points => produce 64-dim features
      3) STNkd (feature transform) => 64x64
      4) More MLP => global feature
    """
    def __init__(self, hidden_dim=512):
        super(PointNet, self).__init__()
        self.stn3d = STN3d()      # input transform
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)

        self.stnkd = STNkd(k=64) # feature transform
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, hidden_dim, 1)

    def forward(self, x):
        """
        x shape: (batch_size, num_points, 3) if coming
                 from e.g. DataLoader with that format.
        We'll transpose to (batch_size, 3, num_points) for Conv1d.
        """
        # (B, num_points, 3) -> (B, 3, num_points)
        x = x.transpose(2, 1)

        # Input transform net => 3x3
        trans3x3 = self.stn3d(x)              # (B, 3, 3)
        x = torch.bmm(trans3x3, x)            # (B, 3, num_points)  multiply per-batch

        # First stage convs
        x = F.relu(self.conv1(x))            # -> (B,64,num_points)
        x = F.relu(self.conv2(x))            # -> (B,64,num_points)

        # Feature transform net => 64x64
        trans64 = self.stnkd(x)              # (B, 64, 64)
        x = x.transpose(2, 1)                # => (B, num_points, 64) so we can do bmm
        x = torch.bmm(x, trans64)            # => (B, num_points, 64)
        x = x.transpose(2, 1)                # => (B, 64, num_points)

        # Second stage convs
        x = F.relu(self.conv3(x))            # (B, 64, num_points)
        x = F.relu(self.conv4(x))            # (B,128,num_points)
        x = F.relu(self.conv5(x))            # (B,hidden_dim,num_points)

        # Global feature
        kernel_size = x.shape[-1].item()
        x = torch.nn.functional.max_pool1d(x, kernel_size) # -> (B, hidden_dim, 1)
        x = x.squeeze(-1)                # -> (B, hidden_dim)

        return x  # This is the 'global_feature'