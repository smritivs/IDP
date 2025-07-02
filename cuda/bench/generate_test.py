import numpy as np
import torch

batch_size = 16
num_harmonics = 8
input_dim = 8
output_dim = 12

if __name__ == "__main__":
	input_x = np.random.rand(batch_size,input_dim)

	with open("./input_x.txt","w") as file:
		for i in input_x:
			for j in i:
				file.write(str(j) + "\n")
		file.write("\n")

	input_fourier_coeffs = np.random.rand(2, output_dim, input_dim, num_harmonics) / (np.sqrt(input_dim) * np.sqrt(num_harmonics))

	input_fourier_coeffs_cos = input_fourier_coeffs[0]

	with open("./input_fourier_coeffs_cos.txt","w") as file:
		for i in input_fourier_coeffs_cos:
			for j in i:
				for k in j:
					file.write(str(k) + "\n")
		file.write("\n")

	input_fourier_coeffs_sin = input_fourier_coeffs[1]

	with open("./input_fourier_coeffs_sin.txt","w") as file:
		for i in input_fourier_coeffs_sin:
			for j in i:
				for k in j:
					file.write(str(k) + "\n")
		file.write("\n")
