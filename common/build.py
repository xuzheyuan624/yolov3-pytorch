from torch.utils.ffi import create_extension
ffi = create_extension(
    name='_ext.bbox',
    headers='src/bbox.h',
    sources=['src/bbox.c'],
    with_cuda=False
)
ffi.build()