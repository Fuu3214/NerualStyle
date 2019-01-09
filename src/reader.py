import tensorflow as tf

VGG_MEAN = [123.68, 116.78, 103.94] 

def read_img(path):
    img_bytes = tf.read_file(path);
    img = tf.image.decode_jpeg(img_bytes, channels=3);
    return img;

def resize(img, size):
    size_t = tf.constant(size, tf.float64)
    shape = tf.shape(img);
    height = tf.cast(shape[0], tf.float64)
    width = tf.cast(shape[1], tf.float64)

    cond_op = tf.less(height, width);
    
    new_height, new_width = tf.cond(
        cond_op, 
        lambda: (size_t, (width * size_t) / height),
        lambda: ((height * size_t) / width, size_t)
    );
    
    resized = tf.image.resize_images(
                img,
                [tf.to_int32(new_height), tf.to_int32(new_width)],
                method=tf.image.ResizeMethod.BICUBIC)    
    return resized

def get_style_img(path, size):
    img = read_img(path);
    img = resize(img, size)
    return (img - VGG_MEAN)

def get_content_img(path, size):
    img = read_img(path);
    img = resize(img, size)
    return (img - VGG_MEAN)
