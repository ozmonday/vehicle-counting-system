import tensorflow as tf
from tensorflow.keras import layers, backend, initializers, models


def conv(x, filters, kernel_size, downsampling=False, activation='leaky', batch_norm=True):
    def mish(x):
        return x * tf.math.tanh(tf.math.softplus(x))

    if downsampling:
        x = layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(x)  # top & left padding
        padding = 'valid'
        strides = 2
    else:
        padding = 'same'
        strides = 1
    x = layers.Conv2D(filters,
                      kernel_size,
                      strides=strides,
                      padding=padding,
                      use_bias=not batch_norm,
                      # kernel_regularizer=regularizers.l2(0.0005),
                      kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                      # bias_initializer=initializers.Zeros()
                      )(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    if activation == 'mish':
        x = mish(x)
    elif activation == 'leaky':
        x = layers.LeakyReLU(alpha=0.1)(x)
    return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
         new_v += divisor
    return new_v


def conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1), block_id=0):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    x = layers.Conv2D(filters, kernel, padding='same', use_bias=False, strides=strides, name='conv_block_%d' % block_id)(inputs)
    x = layers.BatchNormalization(axis=channel_axis, name='conv_block_%d_bn' % block_id)(x)
    return layers.ReLU(6., name='conv_block_%d_relu' % block_id)(x)


def inverted_res_block(inputs, expansion, strides, filters, block_id):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    in_channels = backend.int_shape(inputs)[channel_axis]
    pointwise_filters = _make_divisible(filters, 8)
    prefix = 'block_{}_'.format(block_id)
    x = inputs
    x = conv_block(x, expansion * in_channels, kernel=(1, 1), block_id=block_id) if block_id else x
    x = x if strides == (1, 1) else layers.ZeroPadding2D(((0, 1), (0, 1)), name='conv_pad_%d' % block_id)(x)
    # Depthwise
    x = layers.DepthwiseConv2D((3, 3),
                               padding='same' if strides == (1, 1) else 'valid',
                               strides=strides,
                               use_bias=False,
                               name='conv_dw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)
    x = layers.Conv2D(pointwise_filters, (1, 1), padding='same', use_bias=False, strides=(1, 1),
                      name='conv_pw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)

    if in_channels == pointwise_filters and strides == (1, 1):
        return layers.Add(name=prefix + 'add')([inputs, x])
    return x


def residual_block(x, filters1, filters2, activation='leaky'):
    """
    :param x: input tensor
    :param filters1: num of filter for 1x1 conv
    :param filters2: num of filter for 3x3 conv
    :param activation: default activation function: leaky relu
    :return:
    """
    y = conv(x, filters1, kernel_size=1, activation=activation)
    y = conv(y, filters2, kernel_size=3, activation=activation)
    return layers.Add()([x, y])


def csp_block(x, residual_out, repeat, residual_bottleneck=False):
    """
    Cross Stage Partial Network (CSPNet)
    transition_bottleneck_dims: 1x1 bottleneck
    output_dims: 3x3
    :param x:
    :param residual_out:
    :param repeat:
    :param residual_bottleneck:
    :return:
    """
    route = x
    route = conv(route, residual_out, 1, activation="mish")
    x = conv(x, residual_out, 1, activation="mish")
    for i in range(repeat):
        x = residual_block(x,
                           residual_out // 2 if residual_bottleneck else residual_out,
                           residual_out,
                           activation="mish")
    x = conv(x, residual_out, 1, activation="mish")

    x = layers.Concatenate()([x, route])
    return x


def MobileNetV2(inputs, num_classes, anchor_size):
    first_block_filters = _make_divisible(32, 8)
    x = conv_block(inputs=inputs, filters=first_block_filters,
                   kernel=(3, 3), strides=(2, 2), block_id=0)

    x = inverted_res_block(x, filters=16, strides=1, expansion=1, block_id=1)

    x = inverted_res_block(x, filters=24, strides=2, expansion=6, block_id=2)
    x = inverted_res_block(x, filters=24, strides=1, expansion=6, block_id=3)

    x = inverted_res_block(x, filters=32, strides=2, expansion=6, block_id=4)
    x = inverted_res_block(x, filters=32, strides=1, expansion=6, block_id=5)
    x = inverted_res_block(x, filters=32, strides=1, expansion=6, block_id=6)

    x = inverted_res_block(x, filters=64, strides=2, expansion=6, block_id=7)
    x = inverted_res_block(x, filters=64, strides=1, expansion=6, block_id=8)
    x = inverted_res_block(x, filters=64, strides=1, expansion=6, block_id=9)
    x = inverted_res_block(x, filters=64, strides=1, expansion=6, block_id=10)

    x = inverted_res_block(x, filters=96, strides=1, expansion=6, block_id=11)
    x = inverted_res_block(x, filters=96, strides=1, expansion=6, block_id=12)
    x = inverted_res_block(x, filters=96, strides=1, expansion=6, block_id=13)
    
    x = inverted_res_block(x, filters=160, strides=2, expansion=6, block_id=14)
    x = inverted_res_block(x, filters=160, strides=1, expansion=6, block_id=15)
    x = inverted_res_block(x, filters=160, strides=1, expansion=6, block_id=16)
    
    x = inverted_res_block(x, filters=320, strides=1, expansion=6, block_id=17)
    x = conv_block(x, 1280, kernel=(1, 1), block_id=18)
    outputs = conv(x, anchor_size * (num_classes + 5), 1, activation=None, batch_norm=False)

    return models.Model(inputs, outputs, name='mobilenetv2')
    


def DarkNet53(x):
    x = conv(x, 32, 3)
    x = conv(x, 64, 3, downsampling=True)

    for i in range(1):
        x = residual_block(x, 32, 64)
    x = conv(x, 128, 3, downsampling=True)

    for i in range(2):
        x = residual_block(x, 64, 128)
    x = conv(x, 256, 3, downsampling=True)

    for i in range(8):
        x = residual_block(x, 128, 256)
    route_1 = x
    x = conv(x, 512, 3, downsampling=True)

    for i in range(8):
        x = residual_block(x, 256, 512)
    route_2 = x
    x = conv(x, 1024, 3, downsampling=True)

    for i in range(4):
        x = residual_block(x, 512, 1024)

    return route_1, route_2, x


def CSPDarkNet53(input):
    x = conv(input, 32, 3)
    x = conv(x, 64, 3, downsampling=True)

    x = csp_block(x, residual_out=64, repeat=1, residual_bottleneck=True)
    x = conv(x, 64, 1, activation='mish')
    x = conv(x, 128, 3, activation='mish', downsampling=True)

    x = csp_block(x, residual_out=64, repeat=2)
    x = conv(x, 128, 1, activation='mish')
    x = conv(x, 256, 3, activation='mish', downsampling=True)

    x = csp_block(x, residual_out=128, repeat=8)
    x = conv(x, 256, 1, activation='mish')
    route0 = x
    x = conv(x, 512, 3, activation='mish', downsampling=True)

    x = csp_block(x, residual_out=256, repeat=8)
    x = conv(x, 512, 1, activation='mish')
    route1 = x
    x = conv(x, 1024, 3, activation='mish', downsampling=True)

    x = csp_block(x, residual_out=512, repeat=4)

    x = conv(x, 1024, 1, activation="mish")

    x = conv(x, 512, 1)
    x = conv(x, 1024, 3)
    x = conv(x, 512, 1)

    x = layers.Concatenate()([layers.MaxPooling2D(pool_size=13, strides=1, padding='same')(x),
                              layers.MaxPooling2D(pool_size=9, strides=1, padding='same')(x),
                              layers.MaxPooling2D(pool_size=5, strides=1, padding='same')(x),
                              x
                              ])
    x = conv(x, 512, 1)
    x = conv(x, 1024, 3)
    route2 = conv(x, 512, 1)
    return models.Model(input, [route0, route1, route2])


def PANet(backbone_model, num_classes = 1, anchor_size = 3):
    route0, route1, route2 = backbone_model.output

    route_input = route2
    x = conv(route2, 256, 1)
    x = layers.UpSampling2D()(x)
    route1 = conv(route1, 256, 1)
    x = layers.Concatenate()([route1, x])

    x = conv(x, 256, 1)
    x = conv(x, 512, 3)
    x = conv(x, 256, 1)
    x = conv(x, 512, 3)
    x = conv(x, 256, 1)

    route1 = x
    x = conv(x, 128, 1)
    x = layers.UpSampling2D()(x)
    route0 = conv(route0, 128, 1)
    x = layers.Concatenate()([route0, x])

    x = conv(x, 128, 1)
    x = conv(x, 256, 3)
    x = conv(x, 128, 1)
    x = conv(x, 256, 3)
    x = conv(x, 128, 1)

    route0 = x
    x = conv(x, 256, 3)
    conv_sbbox = conv(x, anchor_size * (num_classes + 5), 1, activation=None, batch_norm=False)

    x = conv(route0, 256, 3, downsampling=True)
    x = layers.Concatenate()([x, route1])

    x = conv(x, 256, 1)
    x = conv(x, 512, 3)
    x = conv(x, 256, 1)
    x = conv(x, 512, 3)
    x = conv(x, 256, 1)

    route1 = x
    x = conv(x, 512, 3)
    conv_mbbox = conv(x, anchor_size * (num_classes + 5), 1, activation=None, batch_norm=False)

    x = conv(route1, 512, 3, downsampling=True)
    x = layers.Concatenate()([x, route_input])

    x = conv(x, 512, 1)
    x = conv(x, 1024, 3)
    x = conv(x, 512, 1)
    x = conv(x, 1024, 3)
    x = conv(x, 512, 1)

    x = conv(x, 1024, 3)
    conv_lbbox = conv(x, anchor_size * (num_classes + 5), 1, activation=None, batch_norm=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]
    

def get_boxes(pred, anchors, classes, strides, xyscale):
    # (batch_size, grid_size, grid_size, 3, 5+classes)
    pred = layers.Reshape((pred.shape[1], pred.shape[1], anchors.shape[0], 5 + classes))(pred)
    # (?, 52, 52, 3, 2) (?, 52, 52, 3, 2) (?, 52, 52, 3, 1) (?, 52, 52, 3, 80)
    box_xy, box_wh, obj_prob, class_prob = tf.split(pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)  # (?, 52, 52, 3, 2)
    obj_prob = tf.sigmoid(obj_prob)  # (?, 52, 52, 3, 1)
    class_prob = tf.sigmoid(class_prob)  # (?, 52, 52, 3, 80)
    pred_box_xywh = tf.concat((box_xy, box_wh), axis=-1)  # (?, 52, 52, 3, 4)

    grid = tf.meshgrid(tf.range(pred.shape[1]), tf.range(pred.shape[1]))  # (52, 52) (52, 52)
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # (52, 52, 1, 2)
    grid = tf.cast(grid, dtype=tf.float32)

    box_xy = ((box_xy * xyscale) - 0.5 * (xyscale - 1) + grid) * strides  # (?, 52, 52, 1, 4)

    box_wh = tf.exp(box_wh) * anchors  # (?, 52, 52, 3, 2)
    box_x1y1 = box_xy - box_wh / 2  # (?, 52, 52, 3, 2)
    box_x2y2 = box_xy + box_wh / 2  # (?, 52, 52, 3, 2)
    pred_box_x1y1x2y2 = tf.concat([box_x1y1, box_x2y2], axis=-1)  # (?, 52, 52, 3, 4)

    return [pred_box_x1y1x2y2, obj_prob, class_prob, pred_box_xywh]


def yolo_detector(prediction, anchors, classes, strides, xyscale):
    small = get_boxes(prediction[0], anchors[0, :, :], classes, strides[0], xyscale[0])
    medium = get_boxes(prediction[1], anchors[1, :, :], classes, strides[1], xyscale[1])
    large = get_boxes(prediction[2], anchors[2, :, :], classes, strides[2], xyscale[2])

    return [*small, *medium, *large]