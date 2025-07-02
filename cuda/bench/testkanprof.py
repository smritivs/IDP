# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from torch import nn
import time
from torch.profiler import profile, record_function, ProfilerActivity



class KolmogorovArnoldNetwork(nn.Module):
    """
    Kolmogorov–Arnold Network (KAN) layer using Fourier-based function approximation.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    output_dim : int
        Dimensionality of the output features.
    num_harmonics : int, optional
        Number of Fourier harmonics to use (default: 5).
    add_bias : bool, optional
        Whether to include an additive bias term (default: True).
    """

    def __init__(self, input_dim, output_dim, fourier_coeffs,num_harmonics=5, add_bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_harmonics = num_harmonics
        self.add_bias = add_bias

        # Initialize Fourier coefficients (cosine and sine) with scaling for stability.
        # Shape: [2, output_dim, input_dim, num_harmonics]
        # self.fourier_coeffs = nn.Parameter(
        #     torch.randn(2, output_dim, input_dim, num_harmonics)
        #     / (np.sqrt(input_dim) * np.sqrt(num_harmonics))
        # )

        self.fourier_coeffs = fourier_coeffs

        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(1, output_dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the KAN layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """
        batch_size = x.size(0)
        # Reshape input to (batch_size, input_dim, 1) for harmonic multiplication.
        x = x.view(batch_size, self.input_dim, 1)
        # x = x.to('cuda')
        # self.bias = self.bias.to('cuda')
        # Create harmonic multipliers (from 1 to num_harmonics).
        k = torch.arange(1, self.num_harmonics + 1, device=x.device).view(
            1, 1, self.num_harmonics
        )
        # Compute cosine and sine components.
        cos_terms = torch.cos(k * x)
        sin_terms = torch.sin(k * x)
        # Perform Fourier expansion using Einstein summation for efficiency.
        y_cos = torch.einsum("bij,oij->bo", cos_terms, self.fourier_coeffs[0])
        y_sin = torch.einsum("bij,oij->bo", sin_terms, self.fourier_coeffs[1])
        y = y_cos + y_sin
        if self.add_bias:
            y = y + self.bias
        return y

if __name__ == "__main__":
    device = 'cuda'
    batch_size = 16
    num_harmonics = 8
    input_dim = 8
    output_dim = 12
    bias = 0
    add_bias = False
    x = torch.ones((batch_size,input_dim)).to(device)
    y = torch.zeros((batch_size,output_dim)).to(device)
    fourier_coeffs = torch.ones((2,output_dim,input_dim,num_harmonics))

    with open("input_x.txt","r") as file:
        contents = file.read()
        contents = contents.split()
        index = 0
        for i in range(batch_size):
            for j in range(input_dim):
                x[i][j] = float(contents[index])
                index+=1

    with open("input_fourier_coeffs_cos.txt","r") as file:
        contents = file.read()
        contents = contents.split()
        index = 0
        for i in range(output_dim):
            for j in range(input_dim):
                for k in range(num_harmonics):
                    fourier_coeffs[0][i][j][k] = float(contents[index])
                    index+=1

    with open("input_fourier_coeffs_sin.txt","r") as file:
        contents = file.read()
        contents = contents.split()
        index = 0
        for i in range(output_dim):
            for j in range(input_dim):
                for k in range(num_harmonics):
                    fourier_coeffs[1][i][j][k] = float(contents[index])
                    index+=1


    fourier_coeffs = torch.FloatTensor(fourier_coeffs).to(device)
    kan = KolmogorovArnoldNetwork(input_dim=input_dim,output_dim=output_dim,num_harmonics=num_harmonics,fourier_coeffs=fourier_coeffs)
    kan = kan.to(device)
    activities = [ProfilerActivity.CUDA]
    # activities = [ProfilerActivity.CUDA, ProfilerActivity.CPU]
    sort_by_keyword = device + "_time_total"

    with profile(activities=activities, record_shapes=True) as prof:
   	    with record_function("forward"):
   	        y = kan.forward(x)

    print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=100))
    y = y.cpu().detach().numpy().tolist()
    # print(y)

    with open("pydump.txt","w") as file:
        for i in y:
            for j in i:
                file.write(str(j) + "\n")
