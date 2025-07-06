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

import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.profiler import profile, record_function, ProfilerActivity



class FourierLayer(nn.Module):
    """Fourier layer used in the Fourier feature network"""

    def __init__(
        self,
        in_features: int,
        frequencies,
    ) -> None:
        super().__init__()

        # To do: Need more robust way for these params
        # if isinstance(frequencies[0], str):
        #     if "gaussian" in frequencies[0]:
        #         nr_freq = frequencies[2]
        #         np_f = (
        #             np.random.normal(0, 1, size=(nr_freq, in_features)) * frequencies[1]
        #         )
        #     else:
        #         nr_freq = len(frequencies[1])
        #         np_f = []
        #         if "full" in frequencies[0]:
        #             np_f_i = np.meshgrid(
        #                 *[np.array(frequencies[1]) for _ in range(in_features)],
        #                 indexing="ij",
        #             )
        #             np_f.append(
        #                 np.reshape(
        #                     np.stack(np_f_i, axis=-1),
        #                     (nr_freq**in_features, in_features),
        #                 )
        #             )
        #         if "axis" in frequencies[0]:
        #             np_f_i = np.zeros((nr_freq, in_features, in_features))
        #             for i in range(in_features):
        #                 np_f_i[:, i, i] = np.reshape(
        #                     np.array(frequencies[1]), (nr_freq)
        #                 )
        #             np_f.append(
        #                 np.reshape(np_f_i, (nr_freq * in_features, in_features))
        #             )
        #         if "diagonal" in frequencies[0]:
        #             np_f_i = np.reshape(np.array(frequencies[1]), (nr_freq, 1, 1))
        #             np_f_i = np.tile(np_f_i, (1, in_features, in_features))
        #             np_f_i = np.reshape(np_f_i, (nr_freq * in_features, in_features))
        #             np_f.append(np_f_i)
        #         np_f = np.concatenate(np_f, axis=-2)

        # else:
        #     np_f = frequencies  # [nr_freq, in_features]

        # frequencies = torch.tensor(np_f, dtype=torch.get_default_dtype())
        # frequencies = frequencies.t().contiguous()
        # self.register_buffer("frequencies", frequencies)

        self.frequencies = frequencies

    def out_features(self) -> int:
        return int(self.frequencies.size(1) * 2)

    def forward(self, x, freqs):
        x_hat = torch.matmul(x, freqs)
        x_sin = torch.sin(2.0 * math.pi * x_hat)
        x_cos = torch.cos(2.0 * math.pi * x_hat)
        x_i = torch.cat([x_sin, x_cos], dim=-1)
        return x_i


class FourierFilter(nn.Module):
    """Fourier filter used in the multiplicative filter network"""

    def __init__(
        self,
        in_features: int,
        layer_size: int,
        nr_layers: int,
        input_scale: float,
    ) -> None:
        super().__init__()

        self.weight_scale = input_scale / math.sqrt(nr_layers + 1)

        self.frequency = nn.Parameter(torch.empty(in_features, layer_size))
        # The shape of phase tensor was supposed to be [1, layer_size], but it has issue
        # with batched tensor in FuncArch.
        # We could just rely on broadcast here.
        self.phase = nn.Parameter(torch.empty(layer_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets parameters"""
        nn.init.xavier_uniform_(self.frequency)
        nn.init.uniform_(self.phase, -math.pi, math.pi)

    def forward(self, x: Tensor) -> Tensor:
        frequency = self.weight_scale * self.frequency

        x_i = torch.sin(torch.matmul(x, 2.0 * math.pi * frequency) + self.phase)
        return x_i


class GaborFilter(nn.Module):
    """Gabor filter used in the multiplicative filter network"""

    def __init__(
        self,
        in_features: int,
        layer_size: int,
        nr_layers: int,
        input_scale: float,
        alpha: float,
        beta: float,
    ) -> None:
        super().__init__()

        self.layer_size = layer_size
        self.alpha = alpha
        self.beta = beta

        self.weight_scale = input_scale / math.sqrt(nr_layers + 1)

        self.frequency = nn.Parameter(torch.empty(in_features, layer_size))
        self.phase = nn.Parameter(torch.empty(layer_size))
        self.mu = nn.Parameter(torch.empty(in_features, layer_size))
        self.gamma = nn.Parameter(torch.empty(layer_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets parameters"""
        nn.init.xavier_uniform_(self.frequency)
        nn.init.uniform_(self.phase, -math.pi, math.pi)
        nn.init.uniform_(self.mu, -1.0, 1.0)
        with torch.no_grad():
            self.gamma.copy_(
                torch.from_numpy(
                    np.random.gamma(self.alpha, 1.0 / self.beta, (self.layer_size)),
                )
            )

    def forward(self, x: Tensor, freqs: Tensor) -> Tensor:
        frequency = self.weight_scale * (freqs * self.gamma.sqrt())

        x_c = x.unsqueeze(-1)
        x_c = x_c - self.mu
        # The norm dim changed from 1 to -2 to be compatible with BatchedTensor
        x_c = torch.square(x_c.norm(p=2, dim=-2))
        x_c = torch.exp(-0.5 * x_c * self.gamma)
        x_i = x_c * torch.sin(torch.matmul(x, 2.0 * math.pi * frequency) + self.phase)
        return x_i


if __name__ == "__main__":
	device = 'cuda'
	batch_size = 4
	num_freqs = 3
	input_dim = 2
	output_dim = input_dim
	x_in = torch.ones((batch_size,input_dim)).to(device)
	y = torch.zeros((batch_size,output_dim)).to(device)
	freqs = torch.ones((input_dim,num_freqs)).to(device)

	with open("input_x.txt","r") as file:
		contents = file.read()
		contents = contents.split()
		index = 0
		for i in range(batch_size):
			for j in range(input_dim):
				x_in[i][j] = float(contents[index])
				index+=1

	with open("input_freqs.txt","r") as file:
		contents = file.read()
		contents = contents.split()
		index = 0
		for i in range(input_dim):
			for j in range(num_freqs):
				freqs[i][j] = float(contents[index])
				index+=1



	fl = FourierLayer(in_features=input_dim,frequencies=freqs)
	fl = fl.to(device)

	activities = [ProfilerActivity.CUDA]
	# activities = [ProfilerActivity.CUDA, ProfilerActivity.CPU]
	sort_by_keyword = device + "_time_total"

	with profile(activities=activities, record_shapes=True) as prof:
		with record_function("forward"):
			y = fl.forward(x=x_in, freqs=freqs)

	print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=100))
	print(y.cpu().detach().numpy().tolist())
