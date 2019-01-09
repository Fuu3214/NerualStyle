import tensorflow as tf
import vgg
import reader

def gram(layer):
    shape = tf.shape(layer)
    B = shape[0]
    H = shape[1]
    W = shape[2]
    C = shape[3]
    filters = tf.reshape(layer, tf.stack([B, -1, C]))
    grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(C * H * W)
    return grams

def gram_style(path, size, style_layers, style_weights):
    with tf.Graph().as_default() as g:
        image = tf.expand_dims(reader.get_style_img(path, size), 0)
        net, _ = vgg.net(image)
        layers = []
        for layer, weight in zip(style_layers, style_weights):
            layers.append(weight * gram(net[layer]))

        with tf.Session() as sess:
            return sess.run(layers)

def phi_content(path, size, content_layers):
    with tf.Graph().as_default() as g:
        image = tf.expand_dims(reader.get_content_img(path, size), 0)
        net, _ = vgg.net(image)
        layers = []
        for layer in content_layers:
            layers.append(net[layer])

        with tf.Session() as sess:
            return sess.run(layers)

def style(net, gram_style_t, style_layers):
    loss = 0
    for gram_style, layer in zip(gram_style_t, style_layers):
        layer_size = tf.size(gram_style)
        loss += tf.nn.l2_loss(gram(net[layer]) - gram_style) / tf.to_float(layer_size)
    return loss

def content(net, phi_content_t, content_layers):
    loss = 0
    for phi_content, layer in zip(phi_content_t, content_layers):
        layer_size = tf.size(phi_content)
        loss += tf.nn.l2_loss(net[layer] - phi_content) / tf.to_float(layer_size)
    return loss

def total_variation(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, height - 1, -1, -1])) - tf.slice(layer, [0, 1, 0, 0], [-1, -1, -1, -1])
    x = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, -1, width - 1, -1])) - tf.slice(layer, [0, 0, 1, 0], [-1, -1, -1, -1])
    loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))
    return loss