import torch
from weighted_sum_triton import weighted_sum_fwd
import triton
import triton.language as tl
from einops import rearrange

class WeightedSumFunc(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, weight):
		# Cache x and weight to be used in the backward pass, when we
		# only receive the gradient wrt. the output tensor, and
		# need to compute the gradients wrt. x and weight.
		D, output_dims = x.shape[-1], x.shape[:-1]

		# Reshape input tensor to 2D
		input_shape = x.shape
		x = rearrange(x, "... d -> (...) d")

		ctx.save_for_backward(x, weight)

		assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
		assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
		assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

		ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16  # Roughly 16 loops through the embedding dimension
		ctx.ROWS_TILE_SIZE = 16  # Each thread processes 16 batch elements at a time
		ctx.input_shape = input_shape

		# Need to initialize empty result tensor. Note that these elements are not necessarily 0!
		y = torch.empty(output_dims, device=x.device)

		# Launch our kernel with n instances in our 1D grid.
		n_rows = y.numel()
		weighted_sum_fwd[(tl.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
			x,
			weight,
			y,
			x.stride(0),
			x.stride(1),
			weight.stride(0),
			y.stride(0),
			ROWS=n_rows,
			D=D,
			ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE,
			D_TILE_SIZE=ctx.D_TILE_SIZE,
		)

		return y.view(input_shape[:-1])