import onnx2tf

kat = [
    'conv_buf',
    'deconv_buf',
    'istft_buf',
]

for i in range(6):
    kat.append(f'gridnet_bufs____buf{i}____h0')
    kat.append(f'gridnet_bufs____buf{i}____c0')

# Output is very inefficient
onnx2tf.convert('pretrained/ONNX/model.onnx',
                output_folder_path='pretrained/tf',
                keep_shape_absolutely_input_names=kat)