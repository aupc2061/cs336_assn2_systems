import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    x_block = tl.make_block_ptr(
        x_ptr,
        shape=(n_elements,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    y_block = tl.make_block_ptr(
        y_ptr,
        shape=(n_elements,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    out_block = tl.make_block_ptr(
        x_ptr,
        shape=(n_elements,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    x = tl.load(x_block, boundary_check=(0,))
    y = tl.load(y_block, boundary_check=(0,))
    out = x + y
    tl.store(out_block, output_ptr, boundary_check=(0,))