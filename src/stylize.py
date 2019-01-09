import time
import tensorflow as tf
import vgg
import reader
import loss

STYLE_WEIGHTS = [0.5, 1, 1.5, 1.25, 0.75]
STYLE_LAYERS = ["relu1_1", "relu2_1", "relu3_1",  "relu4_1", "relu5_1"]
CONTENT_LAYERS = ["relu3_2", "relu4_2"]
LAMBDA = [152, 4, 23]
ITERATIONS = 50

def gaussian_noise(img, mean, stddev, lambda_noise):
    noise = tf.random_normal(
                tf.shape(img), 
                mean = mean , stddev = stddev, 
                dtype = tf.float32
            )
    noisy_img = (1 - lambda_noise) * img + lambda_noise * noise
    with tf.Session() as sess:
        return sess.run(noisy_img)
    
def print_loss(loss_evaled):
    print(loss_evaled);
#     if(COUNT_ITER % 100 == 0):
#         image_t = sess.run(output)
#         with open('out_step_' + str(step) + ".png", 'wb') as f:
#             f.write(image_t)
    
def stylize(style_path, content_path, size):
    init = reader.get_content_img(content_path, size)
    random = tf.random_normal(tf.Session().run(init).shape)
    random = init_noisy = tf.expand_dims(random, 0)
#     init_noisy = gaussian_noise(init, 0 , 46, 0.9)
#     init_noisy = tf.expand_dims(init_noisy, 0)
    
    generated = tf.Variable(random)

    gram_style_t = loss.gram_style(style_path, size, STYLE_LAYERS, STYLE_WEIGHTS)
    phi_content_t = loss.phi_content(content_path, size, CONTENT_LAYERS)
    
    net,_ = vgg.net(generated)

    style_loss = loss.style(net, gram_style_t, STYLE_LAYERS)
    content_loss = loss.content(net, phi_content_t, CONTENT_LAYERS)
    tv_loss = loss.total_variation(generated)
    
    total_loss = LAMBDA[0] * style_loss + LAMBDA[1] * content_loss + LAMBDA[2] * tv_loss
    
#     train_op = tf.train.AdamOptimizer(leanring_rate).minimize(total_loss)
    
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                    total_loss, method='L-BFGS-B',
                    options={'maxiter': ITERATIONS,
                             'disp': 0})

    output = tf.image.encode_png(tf.saturate_cast(tf.squeeze(generated) + reader.VGG_MEAN, tf.uint8))
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        image_t = sess.run(output)
        with open('init.png', 'wb') as f:
            f.write(image_t)
        for i in range(10):
            optimizer.minimize(sess,
                         loss_callback=print_loss,
                         fetches=[total_loss])
            image_t = sess.run(output)
            with open('out_step_' + str((i+1) * ITERATIONS) + '.png', 'wb') as f:
                f.write(image_t)