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

    def __init__(self, input_dim, output_dim, fourier_coeffs, num_harmonics=5, add_bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_harmonics = num_harmonics
        self.add_bias = add_bias

        self.fourier_coeffs = fourier_coeffs

        # Initialize Fourier coefficients (cosine and sine) with scaling for stability.
        # Shape: [2, output_dim, input_dim, num_harmonics]
        # self.fourier_coeffs = nn.Parameter(
            # torch.randn(2, output_dim, input_dim, num_harmonics)
            # / (np.sqrt(input_dim) * np.sqrt(num_harmonics))
        # )

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
        # Create harmonic multipliers (from 1 to num_harmonics).
        k = torch.arange(1, self.num_harmonics + 1, device=x.device).view(
            1, 1, self.num_harmonics
        )
        # Compute cosine and sine components.
        a = k * x
        print(f"k shape: {k.shape} | x_shape: {x.shape} | prod shape: {a.shape}")
        # print(k)
        # print(a)
        cos_terms = torch.cos(k * x)
        sin_terms = torch.sin(k * x)
        print(cos_terms.shape)
        # Perform Fourier expansion using Einstein summation for efficiency.
        y_cos = torch.einsum("bij,oij->bo", cos_terms, self.fourier_coeffs[0])
        y_sin = torch.einsum("bij,oij->bo", sin_terms, self.fourier_coeffs[1])
        print(f"cos shape {cos_terms.shape} | coeff shape {self.fourier_coeffs[0].shape}")
        print(y_cos.shape)
        print("-"*10)
        print(self.fourier_coeffs[0])
        print(self.fourier_coeffs[1])
        print("-"*10)

        y = y_cos + y_sin
        if self.add_bias:
            y = y + self.bias

        with open("./test/dump.txt","w") as file:
            file.write("Fourier Coeffs Cos\n")
            f0 = str(self.fourier_coeffs[0].detach().numpy().tolist())
            f1 = str(self.fourier_coeffs[1].detach().numpy().tolist())
            file.write(f0)
            file.write("\n")
            file.write("Fourier Coeffs Sin\n")
            file.write(f1)
            file.write("\n");
            c0 = str(cos_terms.detach().numpy().tolist())
            s0 = str(sin_terms.detach().numpy().tolist())
            file.write(f"cos_terms\n{c0}\n")
            file.write(f"sin_terms\n{s0}\n")
            file.write(f"In\n{x}\n")
            file.write(f"Res\n{y}\n")

        return y

if __name__ == "__main__":
    batch_size = 1
    num_harmonics = 2
    input_dim = 4
    output_dim = 3
    x = torch.ones((batch_size,input_dim))
    y = torch.zeros((batch_size,output_dim))
    fourier_coeffs = [[[[-0.11764081567525864, 0.26520204544067383],
                     [-0.2647557258605957, -0.6589452028274536],
                     [-0.26754602789878845, -0.1423449069261551],
                     [0.5330049395561218, -0.27065348625183105]],
                     [[-0.5419355034828186, -0.5137631297111511], 
                     [-0.056270383298397064, 0.05014132708311081],
                     [0.23562470078468323, 1.270749807357788], 
                     [0.0936775878071785, -0.22838330268859863]], 
                     [[-0.2776329219341278, -0.1495910882949829], 
                     [0.3131037950515747, 0.1637701541185379], 
                     [0.31852877140045166, -0.38924485445022583], 
                     [-0.4049484431743622, -0.09340766817331314]]],
                     [[[-0.7642549872398376, -0.3950747549533844], 
                     [0.5043845772743225, -0.0421271026134491], 
                     [0.07050096988677979, -0.34597474336624146], 
                     [-0.3534367084503174, 0.14451274275779724]], 
                     [[0.15364940464496613, 0.22368177771568298], 
                     [-0.40279123187065125, 0.4280928373336792], 
                     [0.29595276713371277, -0.8042312860488892], 
                     [0.5119664669036865, -0.18502190709114075]], 
                     [[-0.31381455063819885, 0.635210394859314], 
                     [0.049303196370601654, -0.07333413511514664], 
                     [-0.1196710392832756, -0.3811095058917999], 
                     [0.15586987137794495, 0.5923671126365662]]]]

    fourier_coeffs = torch.FloatTensor(fourier_coeffs)
    kan = KolmogorovArnoldNetwork(input_dim=input_dim,output_dim=output_dim,num_harmonics=num_harmonics,fourier_coeffs=fourier_coeffs)

    y = kan.forward(x)

    print(y)

    a = torch.FloatTensor([[-0.5419355034828186, -0.5137631297111511],[-0.056270383298397064, 0.05014132708311081],[0.23562470078468323, 1.270749807357788], [0.0936775878071785, -0.22838330268859863]])
    b = torch.FloatTensor([[0.5403023362159729, -0.416146844625473], [0.5403023362159729, -0.416146844625473], [0.5403023362159729, -0.416146844625473], [0.5403023362159729, -0.416146844625473]])

    d = torch.FloatTensor([[0.15364940464496613, 0.22368177771568298], 
                     [-0.40279123187065125, 0.4280928373336792], 
                     [0.29595276713371277, -0.8042312860488892], 
                     [0.5119664669036865, -0.18502190709114075]])
    e = torch.FloatTensor([[0.8414709568023682, 0.9092974066734314], [0.8414709568023682, 0.9092974066734314], [0.8414709568023682, 0.9092974066734314], [0.8414709568023682, 0.9092974066734314]])


    f = d*e

    c = a*b;
    print(c)
    print(f)
    print(c.sum())
    print(f.sum())