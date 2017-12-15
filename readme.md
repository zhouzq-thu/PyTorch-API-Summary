- # torch.nn

  ## Parameters

  - *class* `torch.nn.` `Parameter`[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/parameter.html#Parameter)

  ## Containers

  - *class* `torch.nn.` `Module`[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/module.html#Module)

  - *class* `torch.nn.` `Sequential`(**args*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/container.html#Sequential)

  - *class* `torch.nn.` `ModuleList`(*modules=None*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/container.html#ModuleList)

  - *class* `torch.nn.` `ParameterList`(*parameters=None*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/container.html#ParameterList)

  ## Convolution Layers

  - *class* `torch.nn.` `Conv1d`(*in_channels*, *out_channels*, *kernel_size*, *stride=1*, *padding=0*, *dilation=1*, *groups=1*, *bias=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/conv.html#Conv1d)

  - *class* `torch.nn.` `Conv2d`(*in_channels*, *out_channels*, *kernel_size*, *stride=1*, *padding=0*, *dilation=1*, *groups=1*, *bias=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/conv.html#Conv2d)

  - *class* `torch.nn.` `Conv3d`(*in_channels*, *out_channels*, *kernel_size*, *stride=1*, *padding=0*, *dilation=1*, *groups=1*, *bias=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/conv.html#Conv3d)

  - *class* `torch.nn.` `ConvTranspose1d`(*in_channels*, *out_channels*, *kernel_size*, *stride=1*, *padding=0*, *output_padding=0*, *groups=1*, *bias=True*, *dilation=1*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/conv.html#ConvTranspose1d)

  - *class* `torch.nn.` `ConvTranspose2d`(*in_channels*, *out_channels*, *kernel_size*, *stride=1*, *padding=0*, *output_padding=0*, *groups=1*, *bias=True*, *dilation=1*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/conv.html#ConvTranspose2d)

  - *class* `torch.nn.` `ConvTranspose3d`(*in_channels*, *out_channels*, *kernel_size*, *stride=1*, *padding=0*, *output_padding=0*, *groups=1*, *bias=True*, *dilation=1*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/conv.html#ConvTranspose3d)

  ## Pooling Layers

  - *class* `torch.nn.` `MaxPool1d`(*kernel_size*, *stride=None*, *padding=0*, *dilation=1*, *return_indices=False*, *ceil_mode=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/pooling.html#MaxPool1d)

  - *class* `torch.nn.` `MaxPool2d`(*kernel_size*, *stride=None*, *padding=0*, *dilation=1*, *return_indices=False*, *ceil_mode=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/pooling.html#MaxPool2d)

  - *class* `torch.nn.` `MaxPool3d`(*kernel_size*, *stride=None*, *padding=0*, *dilation=1*, *return_indices=False*, *ceil_mode=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/pooling.html#MaxPool3d)

  - *class* `torch.nn.` `MaxUnpool1d`(*kernel_size*, *stride=None*, *padding=0*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/pooling.html#MaxUnpool1d)

  - *class* `torch.nn.` `MaxUnpool2d`(*kernel_size*, *stride=None*, *padding=0*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/pooling.html#MaxUnpool2d)

  - *class* `torch.nn.` `MaxUnpool3d`(*kernel_size*, *stride=None*, *padding=0*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/pooling.html#MaxUnpool3d)

  - *class* `torch.nn.` `AvgPool1d`(*kernel_size*, *stride=None*, *padding=0*, *ceil_mode=False*, *count_include_pad=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/pooling.html#AvgPool1d)

  - *class* `torch.nn.` `AvgPool2d`(*kernel_size*, *stride=None*, *padding=0*, *ceil_mode=False*, *count_include_pad=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/pooling.html#AvgPool2d)

  - *class* `torch.nn.` `AvgPool3d`(*kernel_size*, *stride=None*, *padding=0*, *ceil_mode=False*, *count_include_pad=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/pooling.html#AvgPool3d)

  - *class* `torch.nn.` `FractionalMaxPool2d`(*kernel_size*, *output_size=None*, *output_ratio=None*, *return_indices=False*, *_random_samples=None*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/pooling.html#FractionalMaxPool2d)

  - *class* `torch.nn.` `LPPool2d`(*norm_type*, *kernel_size*, *stride=None*, *ceil_mode=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/pooling.html#LPPool2d)

  - *class* `torch.nn.` `AdaptiveMaxPool1d`(*output_size*, *return_indices=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/pooling.html#AdaptiveMaxPool1d)

  - *class* `torch.nn.` `AdaptiveMaxPool2d`(*output_size*, *return_indices=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/pooling.html#AdaptiveMaxPool2d)

  - *class* `torch.nn.` `AdaptiveMaxPool3d`(*output_size*, *return_indices=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/pooling.html#AdaptiveMaxPool3d)

  - *class* `torch.nn.` `AdaptiveAvgPool1d`(*output_size*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/pooling.html#AdaptiveAvgPool1d)

  - *class* `torch.nn.` `AdaptiveAvgPool2d`(*output_size*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/pooling.html#AdaptiveAvgPool2d)

  - *class* `torch.nn.` `AdaptiveAvgPool3d`(*output_size*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/pooling.html#AdaptiveAvgPool3d)

  ## Padding Layers

  - *class* `torch.nn.` `ReflectionPad2d`(*padding*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/padding.html#ReflectionPad2d)

  - *class* `torch.nn.` `ReplicationPad2d`(*padding*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/padding.html#ReplicationPad2d)

  - *class* `torch.nn.` `ReplicationPad3d`(*padding*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/padding.html#ReplicationPad3d)

  - *class* `torch.nn.` `ZeroPad2d`(*padding*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/padding.html#ZeroPad2d)

  - *class* `torch.nn.` `ConstantPad2d`(*padding*, *value*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/padding.html#ConstantPad2d)

  ## Non-linear Activations

  - *class* `torch.nn.` `ReLU`(*inplace=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/activation.html#ReLU)

  - *class* `torch.nn.` `ReLU6`(*inplace=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/activation.html#ReLU6)

  - *class* `torch.nn.` `ELU`(*alpha=1.0*, *inplace=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/activation.html#ELU)

  - *class* `torch.nn.` `SELU`(*inplace=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/activation.html#SELU)

  - *class* `torch.nn.` `PReLU`(*num_parameters=1*, *init=0.25*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/activation.html#PReLU)

  - *class* `torch.nn.` `LeakyReLU`(*negative_slope=0.01*, *inplace=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/activation.html#LeakyReLU)

  - *class* `torch.nn.` `Threshold`(*threshold*, *value*, *inplace=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/activation.html#Threshold)

  - *class* `torch.nn.` `Hardtanh`(*min_val=-1*, *max_val=1*, *inplace=False*, *min_value=None*, *max_value=None*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/activation.html#Hardtanh)

  - *class* `torch.nn.` `Sigmoid`[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/activation.html#Sigmoid)

  - *class* `torch.nn.` `Tanh`[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/activation.html#Tanh)

  - *class* `torch.nn.` `LogSigmoid`[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/activation.html#LogSigmoid)

  - *class* `torch.nn.` `Softplus`(*beta=1*, *threshold=20*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/activation.html#Softplus)

  - *class* `torch.nn.` `Softshrink`(*lambd=0.5*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/activation.html#Softshrink)

  - *class* `torch.nn.` `Softsign`[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/activation.html#Softsign)

  - *class* `torch.nn.` `Tanhshrink`[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/activation.html#Tanhshrink)

  - *class* `torch.nn.` `Softmin`(*dim=None*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/activation.html#Softmin)

  - *class* `torch.nn.` `Softmax`(*dim=None*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/activation.html#Softmax)

  - *class* `torch.nn.` `Softmax2d`[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/activation.html#Softmax2d)

  - *class* `torch.nn.` `LogSoftmax`(*dim=None*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/activation.html#LogSoftmax)

  ## Normalization layers

  - *class* `torch.nn.` `BatchNorm1d`(*num_features*, *eps=1e-05*, *momentum=0.1*, *affine=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/batchnorm.html#BatchNorm1d)

  - *class* `torch.nn.` `BatchNorm2d`(*num_features*, *eps=1e-05*, *momentum=0.1*, *affine=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/batchnorm.html#BatchNorm2d)

  - *class* `torch.nn.` `BatchNorm3d`(*num_features*, *eps=1e-05*, *momentum=0.1*, *affine=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/batchnorm.html#BatchNorm3d)

  - *class* `torch.nn.` `InstanceNorm1d`(*num_features*, *eps=1e-05*, *momentum=0.1*, *affine=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/instancenorm.html#InstanceNorm1d)

  - *class* `torch.nn.` `InstanceNorm2d`(*num_features*, *eps=1e-05*, *momentum=0.1*, *affine=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/instancenorm.html#InstanceNorm2d)

  - *class* `torch.nn.` `InstanceNorm3d`(*num_features*, *eps=1e-05*, *momentum=0.1*, *affine=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/instancenorm.html#InstanceNorm3d)

  ## Recurrent layers

  - *class* `torch.nn.` `RNN`(**args*, **\*kwargs*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#RNN)

  - *class* `torch.nn.` `LSTM`(**args*, **\*kwargs*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM)

  - *class* `torch.nn.` `GRU`(**args*, **\*kwargs*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#GRU)

  - *class* `torch.nn.` `RNNCell`(*input_size*, *hidden_size*, *bias=True*, *nonlinearity='tanh'*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#RNNCell)

  - *class* `torch.nn.` `LSTMCell`(*input_size*, *hidden_size*, *bias=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTMCell)

  - *class* `torch.nn.` `GRUCell`(*input_size*, *hidden_size*, *bias=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#GRUCell)

  ## Linear layers

  - *class* `torch.nn.` `Linear`(*in_features*, *out_features*, *bias=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/linear.html#Linear)

  - *class* `torch.nn.` `Bilinear`(*in1_features*, *in2_features*, *out_features*, *bias=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/linear.html#Bilinear)

  ## Dropout layers

  - *class* `torch.nn.` `Dropout`(*p=0.5*, *inplace=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/dropout.html#Dropout)

  - *class* `torch.nn.` `Dropout2d`(*p=0.5*, *inplace=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/dropout.html#Dropout2d)

  - *class* `torch.nn.` `Dropout3d`(*p=0.5*, *inplace=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/dropout.html#Dropout3d)

  - *class* `torch.nn.` `AlphaDropout`(*p=0.5*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/dropout.html#AlphaDropout)

  ## Sparse layers

  - *class* `torch.nn.` `Embedding`(*num_embeddings*, *embedding_dim*, *padding_idx=None*, *max_norm=None*, *norm_type=2*, *scale_grad_by_freq=False*, *sparse=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/sparse.html#Embedding)

  - *class* `torch.nn.` `EmbeddingBag`(*num_embeddings*, *embedding_dim*, *max_norm=None*, *norm_type=2*, *scale_grad_by_freq=False*, *mode='mean'*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/sparse.html#EmbeddingBag)

  ## Distance functions

  - *class* `torch.nn.` `CosineSimilarity`(*dim=1*, *eps=1e-08*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/distance.html#CosineSimilarity)

  - *class* `torch.nn.` `PairwiseDistance`(*p=2*, *eps=1e-06*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/distance.html#PairwiseDistance)

  ## Loss functions

  - *class* `torch.nn.` `L1Loss`(*size_average=True*, *reduce=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/loss.html#L1Loss)

  - *class* `torch.nn.` `MSELoss`(*size_average=True*, *reduce=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/loss.html#MSELoss)

  - *class* `torch.nn.` `CrossEntropyLoss`(*weight=None*, *size_average=True*, *ignore_index=-100*, *reduce=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/loss.html#CrossEntropyLoss)

  - *class* `torch.nn.` `NLLLoss`(*weight=None*, *size_average=True*, *ignore_index=-100*, *reduce=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/loss.html#NLLLoss)

  - *class* `torch.nn.` `PoissonNLLLoss`(*log_input=True*, *full=False*, *size_average=True*, *eps=1e-08*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/loss.html#PoissonNLLLoss)

  - *class* `torch.nn.` `NLLLoss2d`(*weight=None*, *size_average=True*, *ignore_index=-100*, *reduce=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/loss.html#NLLLoss2d)

  - *class* `torch.nn.` `KLDivLoss`(*size_average=True*, *reduce=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/loss.html#KLDivLoss)

  - *class* `torch.nn.` `BCELoss`(*weight=None*, *size_average=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/loss.html#BCELoss)

  - *class* `torch.nn.` `BCEWithLogitsLoss`(*weight=None*, *size_average=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/loss.html#BCEWithLogitsLoss)

  - *class* `torch.nn.` `MarginRankingLoss`(*margin=0*, *size_average=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/loss.html#MarginRankingLoss)

  - *class* `torch.nn.` `HingeEmbeddingLoss`(*margin=1.0*, *size_average=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/loss.html#HingeEmbeddingLoss)

  - *class* `torch.nn.` `MultiLabelMarginLoss`(*size_average=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/loss.html#MultiLabelMarginLoss)

  - *class* `torch.nn.` `SmoothL1Loss`(*size_average=True*, *reduce=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/loss.html#SmoothL1Loss)

  - *class* `torch.nn.` `SoftMarginLoss`(*size_average=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/loss.html#SoftMarginLoss)

  - *class* `torch.nn.` `MultiLabelSoftMarginLoss`(*weight=None*, *size_average=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/loss.html#MultiLabelSoftMarginLoss)

  - *class* `torch.nn.` `CosineEmbeddingLoss`(*margin=0*, *size_average=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/loss.html#CosineEmbeddingLoss)

  - *class* `torch.nn.` `MultiMarginLoss`(*p=1*, *margin=1*, *weight=None*, *size_average=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/loss.html#MultiMarginLoss)

  - *class* `torch.nn.` `TripletMarginLoss`(*margin=1.0*, *p=2*, *eps=1e-06*, *swap=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/loss.html#TripletMarginLoss)

  ## Vision layers

  - *class* `torch.nn.` `PixelShuffle`(*upscale_factor*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/pixelshuffle.html#PixelShuffle)

  - *class* `torch.nn.` `Upsample`(*size=None*, *scale_factor=None*, *mode='nearest'*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/upsampling.html#Upsample)

  - *class* `torch.nn.` `UpsamplingNearest2d`(*size=None*, *scale_factor=None*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/upsampling.html#UpsamplingNearest2d)

  - *class* `torch.nn.` `UpsamplingBilinear2d`(*size=None*, *scale_factor=None*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/modules/upsampling.html#UpsamplingBilinear2d)

  ## DataParallel layers (multi-GPU, distributed)

  - *class* `torch.nn.` `DataParallel`(*module*, *device_ids=None*, *output_device=None*, *dim=0*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/parallel/data_parallel.html#DataParallel)

  - *class* `torch.nn.parallel.` `DistributedDataParallel`(*module*, *device_ids=None*, *output_device=None*, *dim=0*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/parallel/distributed.html#DistributedDataParallel)

  ## Utilities

  - `torch.nn.utils.` `clip_grad_norm`(*parameters*, *max_norm*, *norm_type=2*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm)

  - `torch.nn.utils.` `weight_norm`(*module*, *name='weight'*, *dim=0*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/utils/weight_norm.html#weight_norm)

  - `torch.nn.utils.` `remove_weight_norm`(*module*, *name='weight'*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/utils/weight_norm.html#remove_weight_norm)

  - `torch.nn.utils.rnn.` `PackedSequence`(*data*, *batch_sizes*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/utils/rnn.html#PackedSequence)

  - `torch.nn.utils.rnn.` `pack_padded_sequence`(*input*, *lengths*, *batch_first=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/utils/rnn.html#pack_padded_sequence)

  - `torch.nn.utils.rnn.` `pad_packed_sequence`(*sequence*, *batch_first=False*, *padding_value=0.0*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/utils/rnn.html#pad_packed_sequence)

  # torch.nn.functional

  ## Convolution functions

  - `torch.nn.functional.` `conv1d`(*input*, *weight*, *bias=None*, *stride=1*, *padding=0*, *dilation=1*, *groups=1*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#conv1d)

  - `torch.nn.functional.` `conv2d`(*input*, *weight*, *bias=None*, *stride=1*, *padding=0*, *dilation=1*, *groups=1*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#conv2d)

  - `torch.nn.functional.` `conv3d`(*input*, *weight*, *bias=None*, *stride=1*, *padding=0*, *dilation=1*, *groups=1*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#conv3d)

  - `torch.nn.functional.` `conv_transpose1d`(*input*, *weight*, *bias=None*, *stride=1*, *padding=0*, *output_padding=0*, *groups=1*, *dilation=1*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#conv_transpose1d)

  - `torch.nn.functional.` `conv_transpose2d`(*input*, *weight*, *bias=None*, *stride=1*, *padding=0*, *output_padding=0*, *groups=1*, *dilation=1*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#conv_transpose2d)

  - `torch.nn.functional.` `conv_transpose3d`(*input*, *weight*, *bias=None*, *stride=1*, *padding=0*, *output_padding=0*, *groups=1*, *dilation=1*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#conv_transpose3d)

  ## Pooling functions

  - `torch.nn.functional.` `avg_pool1d`(*input*, *kernel_size*, *stride=None*, *padding=0*, *ceil_mode=False*, *count_include_pad=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#avg_pool1d)

  - `torch.nn.functional.` `avg_pool2d`(*input*, *kernel_size*, *stride=None*, *padding=0*, *ceil_mode=False*, *count_include_pad=True*) → Variable

  - `torch.nn.functional.` `avg_pool3d`(*input*, *kernel_size*, *stride=None*, *padding=0*, *ceil_mode=False*, *count_include_pad=True*) → Variable

  - `torch.nn.functional.` `max_pool1d`(*input*, *kernel_size*, *stride=None*, *padding=0*, *dilation=1*, *ceil_mode=False*, *return_indices=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#max_pool1d)

  - `torch.nn.functional.` `max_pool2d`(*input*, *kernel_size*, *stride=None*, *padding=0*, *dilation=1*, *ceil_mode=False*, *return_indices=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#max_pool2d)

  - `torch.nn.functional.` `max_pool3d`(*input*, *kernel_size*, *stride=None*, *padding=0*, *dilation=1*, *ceil_mode=False*, *return_indices=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#max_pool3d)

  - `torch.nn.functional.` `max_unpool1d`(*input*, *indices*, *kernel_size*, *stride=None*, *padding=0*, *output_size=None*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#max_unpool1d)

  - `torch.nn.functional.` `max_unpool2d`(*input*, *indices*, *kernel_size*, *stride=None*, *padding=0*, *output_size=None*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#max_unpool2d)

  - `torch.nn.functional.` `max_unpool3d`(*input*, *indices*, *kernel_size*, *stride=None*, *padding=0*, *output_size=None*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#max_unpool3d)

  - `torch.nn.functional.` `lp_pool2d`(*input*, *norm_type*, *kernel_size*, *stride=None*, *ceil_mode=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#lp_pool2d)

  - `torch.nn.functional.` `adaptive_max_pool1d`(*input*, *output_size*, *return_indices=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#adaptive_max_pool1d)

  - `torch.nn.functional.` `adaptive_max_pool2d`(*input*, *output_size*, *return_indices=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#adaptive_max_pool2d)

  - `torch.nn.functional.` `adaptive_max_pool3d`(*input*, *output_size*, *return_indices=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#adaptive_max_pool3d)

  - `torch.nn.functional.` `adaptive_avg_pool1d`(*input*, *output_size*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#adaptive_avg_pool1d)

  - `torch.nn.functional.` `adaptive_avg_pool2d`(*input*, *output_size*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#adaptive_avg_pool2d)

  - `torch.nn.functional.` `adaptive_avg_pool3d`(*input*, *output_size*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#adaptive_avg_pool3d)

  ## Non-linear activation functions

  - `torch.nn.functional.` `threshold`(*input*, *threshold*, *value*, *inplace=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#threshold)

  - `torch.nn.functional.` `threshold_`(*input*, *threshold*, *value*) → Variable

  - `torch.nn.functional.` `relu`(*input*, *threshold*, *value*, *inplace=False*) → Variable[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#relu)

  - `torch.nn.functional.` `relu_`(*input*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#relu_)

  - `torch.nn.functional.` `hardtanh`(*input*, *min_val=-1.*, *max_val=1.*, *inplace=False*) → Variable[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#hardtanh)

  - `torch.nn.functional.` `hardtanh_`(*input*, *min_val=-1.*, *max_val=1.*) → Variable

  - `torch.nn.functional.` `relu6`(*input*, *inplace=False*) → Variable[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#relu6)

  - `torch.nn.functional.` `elu`(*input*, *alpha=1.0*, *inplace=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#elu)

  - `torch.nn.functional.` `elu_`(*input*, *alpha=1.*) → Variable

  - `torch.nn.functional.` `selu`(*input*, *inplace=False*) → Variable[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#selu)

  - `torch.nn.functional.` `leaky_relu`(*input*, *negative_slope=0.01*, *inplace=False*) → Variable[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#leaky_relu)

  - `torch.nn.functional.` `leaky_relu_`(*input*, *negative_slope=0.01*) → Variable

  - `torch.nn.functional.` `prelu`(*input*, *weight*) → Variable

  - `torch.nn.functional.` `rrelu`(*input*, *lower=1./8*, *upper=1./3*, *training=False*, *inplace=False*) → Variable[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#rrelu)

  - `torch.nn.functional.` `rrelu_`(*input*, *lower=1./8*, *upper=1./3*, *training=False*) → Variable

  - `torch.nn.functional.` `glu`(*input*, *dim=-1*) → Variable

  - `torch.nn.functional.` `logsigmoid`(*input*) → Variable

  - `torch.nn.functional.` `hardshrink`(*input*, *lambd=0.5*) → Variable

  - `torch.nn.functional.` `tanhshrink`(*input*) → Variable[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#tanhshrink)

  - `torch.nn.functional.` `softsign`(*input*) → Variable[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#softsign)

  - `torch.nn.functional.` `softplus`(*input*, *beta=1*, *threshold=20*) → Variable

  - `torch.nn.functional.` `softmin`(*input*, *dim=None*, *_stacklevel=3*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#softmin)

  - `torch.nn.functional.` `softmax`(*input*, *dim=None*, *_stacklevel=3*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#softmax)

  - `torch.nn.functional.` `softshrink`(*input*, *lambd=0.5*) → Variable[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#softshrink)

  - `torch.nn.functional.` `log_softmax`(*input*, *dim=None*, *_stacklevel=3*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#log_softmax)

  - `torch.nn.functional.` `tanh`(*input*) → Variable[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#tanh)

  - `torch.nn.functional.` `sigmoid`(*input*) → Variable[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#sigmoid)

  ## Normalization functions

  - `torch.nn.functional.` `batch_norm`(*input*, *running_mean*, *running_var*, *weight=None*, *bias=None*, *training=False*, *momentum=0.1*, *eps=1e-05*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#batch_norm)

  - `torch.nn.functional.` `normalize`(*input*, *p=2*, *dim=1*, *eps=1e-12*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#normalize)

  ## Linear functions

  - `torch.nn.functional.` `linear`(*input*, *weight*, *bias=None*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#linear)

  ## Dropout functions

  - `torch.nn.functional.` `dropout`(*input*, *p=0.5*, *training=False*, *inplace=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#dropout)

  - `torch.nn.functional.` `alpha_dropout`(*input*, *p=0.5*, *training=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#alpha_dropout)

  - `torch.nn.functional.` `dropout2d`(*input*, *p=0.5*, *training=False*, *inplace=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#dropout2d)

  - `torch.nn.functional.` `dropout3d`(*input*, *p=0.5*, *training=False*, *inplace=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#dropout3d)


  ## Distance functions

  - `torch.nn.functional.` `pairwise_distance`(*x1*, *x2*, *p=2*, *eps=1e-06*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#pairwise_distance)

  - `torch.nn.functional.` `cosine_similarity`(*x1*, *x2*, *dim=1*, *eps=1e-08*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#cosine_similarity)

  ## Loss functions

  - `torch.nn.functional.` `binary_cross_entropy`(*input*, *target*, *weight=None*, *size_average=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#binary_cross_entropy)

  - `torch.nn.functional.` `poisson_nll_loss`(*input*, *target*, *log_input=True*, *full=False*, *size_average=True*, *eps=1e-08*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#poisson_nll_loss)

  - `torch.nn.functional.` `cosine_embedding_loss`(*input1*, *input2*, *target*, *margin=0*, *size_average=True*) → Variable[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#cosine_embedding_loss)

  - `torch.nn.functional.` `cross_entropy`(*input*, *target*, *weight=None*, *size_average=True*, *ignore_index=-100*, *reduce=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#cross_entropy)

  - `torch.nn.functional.` `hinge_embedding_loss`(*input*, *target*, *margin=1.0*, *size_average=True*) → Variable[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#hinge_embedding_loss)

  - `torch.nn.functional.` `kl_div`(*input*, *target*, *size_average=True*) → Variable

  - `torch.nn.functional.` `l1_loss`(*input*, *target*, *size_average=True*, *reduce=True*) → Variable[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#l1_loss)

  - `torch.nn.functional.` `mse_loss`(*input*, *target*, *size_average=True*, *reduce=True*) → Variable[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#mse_loss)

  - `torch.nn.functional.` `margin_ranking_loss`(*input1*, *input2*, *target*, *margin=0*, *size_average=True*) → Variable[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#margin_ranking_loss)

  - `torch.nn.functional.` `multilabel_margin_loss`(*input*, *target*, *size_average=True*) → Variable

  - `torch.nn.functional.` `multilabel_soft_margin_loss`(*input*, *target*, *weight=None*, *size_average=True*) → Variable[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#multilabel_soft_margin_loss)

  - `torch.nn.functional.` `multi_margin_loss`(*input*, *target*, *p=1*, *margin=1*, *weight=None*, *size_average=True*) → Variable[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#multi_margin_loss)

  - `torch.nn.functional.` `nll_loss`(*input*, *target*, *weight=None*, *size_average=True*, *ignore_index=-100*, *reduce=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#nll_loss)

  - `torch.nn.functional.` `binary_cross_entropy_with_logits`(*input*, *target*, *weight=None*, *size_average=True*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#binary_cross_entropy_with_logits)

  - `torch.nn.functional.` `smooth_l1_loss`(*input*, *target*, *size_average=True*) → Variable

  - `torch.nn.functional.` `soft_margin_loss`(*input*, *target*, *size_average=True*) → Variable

  - `torch.nn.functional.` `triplet_margin_loss`(*anchor*, *positive*, *negative*, *margin=1.0*, *p=2*, *eps=1e-06*, *swap=False*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#triplet_margin_loss)

  ## Vision functions

  - `torch.nn.functional.` `pixel_shuffle`(*input*, *upscale_factor*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#pixel_shuffle)

  - `torch.nn.functional.` `pad`(*input*, *pad*, *mode='constant'*, *value=0*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#pad)

  - `torch.nn.functional.` `upsample`(*input*, *size=None*, *scale_factor=None*, *mode='nearest'*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#upsample)

  - `torch.nn.functional.` `upsample_nearest`(*input*, *size=None*, *scale_factor=None*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#upsample_nearest)

  - `torch.nn.functional.` `upsample_bilinear`(*input*, *size=None*, *scale_factor=None*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#upsample_bilinear)

  - `torch.nn.functional.` `grid_sample`(*input*, *grid*, *mode='bilinear'*, *padding_mode='zeros'*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#grid_sample)

  - `torch.nn.functional.` `affine_grid`(*theta*, *size*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/functional.html#affine_grid)

  # torch.nn.init

  - `torch.nn.init.` `calculate_gain`(*nonlinearity*, *param=None*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/init.html#calculate_gain)

  - `torch.nn.init.` `uniform`(*tensor*, *a=0*, *b=1*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/init.html#uniform)

  - `torch.nn.init.` `normal`(*tensor*, *mean=0*, *std=1*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/init.html#normal)

  - `torch.nn.init.` `constant`(*tensor*, *val*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/init.html#constant)

  - `torch.nn.init.` `eye`(*tensor*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/init.html#eye)

  - `torch.nn.init.` `dirac`(*tensor*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/init.html#dirac)

  - `torch.nn.init.` `xavier_uniform`(*tensor*, *gain=1*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/init.html#xavier_uniform)

  - `torch.nn.init.` `xavier_normal`(*tensor*, *gain=1*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/init.html#xavier_normal)

  - `torch.nn.init.` `kaiming_uniform`(*tensor*, *a=0*, *mode='fan_in'*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/init.html#kaiming_uniform)

  - `torch.nn.init.` `kaiming_normal`(*tensor*, *a=0*, *mode='fan_in'*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/init.html#kaiming_normal)

  - `torch.nn.init.` `orthogonal`(*tensor*, *gain=1*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/init.html#orthogonal)

  - `torch.nn.init.` `sparse`(*tensor*, *sparsity*, *std=0.01*)[[source\]](http://pytorch.org/docs/master/_modules/torch/nn/init.html#sparse)