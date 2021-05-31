from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import glob

import config as cfg
from networks.facade_network import inference_deeplabv3_plus_16
from utils.utils import pred_vision, data_crop_eval_output

os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('mode', "multi_eval", "single/multi-scale to evaluate.")


def main(argv=None):
    image = tf.placeholder(tf.float32, shape=[1, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3], name="input_image")

    # Define model
    _, logits = inference_deeplabv3_plus_16(image)
    logits = tf.nn.softmax(logits)

    sess = tf.Session()
    print("Setting up Saver...")
    saver = tf.train.Saver()

    # Load parameters
    files = os.path.join(cfg.SAVE_DIR + 'model.ckpt-*.index')
    sfile = glob.glob(files)
    if len(sfile) >= 0:
        sess.run(tf.global_variables_initializer())

        sfile = glob.glob(files)
        steps = []
        for s in sfile:
            part = s.split('.')
            step = int(part[1].split('-')[1])
            steps.append(step)
        epo = max(steps)

        model = cfg.SAVE_DIR + 'model.ckpt-' + str(epo)

        print('\nRestoring weights from: ' + model)
        saver.restore(sess, model)
        print('End Restore')
    else:
        # restore from pre-train on imagenet or pre-trained
        variables = tf.global_variables()
        sess.run(tf.variables_initializer(variables, name='init'))
        print('Model initialized random.')

    eval_save_dir = cfg.SAVE_DIR + 'output/'
    if not os.path.exists(eval_save_dir):
        os.mkdir(eval_save_dir)

    # Image Lists
    img_list = glob.glob(cfg.EVAL_DIR + '*.jpg') + glob.glob(cfg.EVAL_DIR + '*.png')

    if FLAGS.mode == "single_eval":
        print('----- Start testing single scale images -----')
        import cv2
        print('Input size: ' + str([cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH]))

        for item in range(len(img_list)):
            ori_img = cv2.imread(img_list[item])
            ori_img_h, ori_img_w = ori_img.shape[0], ori_img.shape[1]

            im_name = img_list[item].split('/')[-1]

            valid_images = cv2.resize(ori_img, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
            valid_images = valid_images - cfg.IMG_MEAN

            # Run
            score_map = sess.run([logits], feed_dict={image: [valid_images]})

            score_map = cv2.resize(score_map[0][0], (ori_img_w, ori_img_h))

            pred_label = np.argmax(score_map, 2)

            # Change balcony to window
            pred_label = np.asarray(pred_label, dtype='uint8')
            pred_label = pred_label[:, :, np.newaxis]
            pred_label_copy = pred_label.copy()
            pred_label[pred_label_copy == 2] = 1  # window
            pred_label[pred_label_copy == 1] = 2  # wall
            pred_label[pred_label_copy == 4] = 1  # balcony
            pred_label[pred_label_copy == 3] = 4  # door

            # Save to path
            save_name = eval_save_dir + im_name
            pred_vision(pred_label, save_name)

            print('image ' + str(item))

    elif FLAGS.mode == "multi_eval":
        print('---------Start multi-scale test img-------------')
        import cv2
        crop_size_h = cfg.IMAGE_HEIGHT
        crop_size_w = cfg.IMAGE_WIDTH
        print('crop size: ' + str(crop_size_h))
        stride = int(crop_size_w / 3)

        # Start evaluating
        for item in range(len(img_list)):
            ori_img = cv2.imread(img_list[item])
            im_name = img_list[item].split('/')[-1]
            ori_img_h, ori_img_w, _ = ori_img.shape

            # Scales to eval
            scs = [0.32, 0.5, 0.75]

            maps = []
            for sc in scs:
                img = cv2.resize(ori_img, (int(float(ori_img_w) * sc), int(float(ori_img_h) * sc)),
                                 interpolation=cv2.INTER_LINEAR)
                score_map = data_crop_eval_output(sess, image, logits, img, cfg.IMG_MEAN, crop_size_h,
                                                  crop_size_w, stride)
                score_map = cv2.resize(score_map, (ori_img_w, ori_img_h), interpolation=cv2.INTER_LINEAR)
                maps.append(score_map)
            score_map = np.mean(np.stack(maps), axis=0)

            maps2 = []
            for sc in scs:
                img2 = cv2.resize(ori_img, (int(float(ori_img_w) * sc), int(float(ori_img_h) * sc)),
                                  interpolation=cv2.INTER_LINEAR)
                img2 = cv2.flip(img2, 1)
                score_map2 = data_crop_eval_output(sess, image, logits, img2, cfg.IMG_MEAN, crop_size_h,
                                                   crop_size_w, stride)
                score_map2 = cv2.resize(score_map2, (ori_img_w, ori_img_h), interpolation=cv2.INTER_LINEAR)
                maps2.append(score_map2)
            score_map2 = np.mean(np.stack(maps2), axis=0)
            score_map2 = cv2.flip(score_map2, 1)

            # Mean
            score_map = (score_map + score_map2) / 2
            pred_label = np.argmax(score_map, 2)

            # Change balcony to window
            pred_label = np.asarray(pred_label, dtype='uint8')
            pred_label = pred_label[:, :, np.newaxis]
            pred_label_copy = pred_label.copy()
            pred_label[pred_label_copy == 2] = 1    # window
            pred_label[pred_label_copy == 1] = 2    # wall
            pred_label[pred_label_copy == 4] = 1    # balcony
            pred_label[pred_label_copy == 3] = 4    # door

            # Save to path
            save_name = eval_save_dir + im_name
            pred_vision(pred_label, save_name)
            print('image ' + str(item))

if __name__ == "__main__":
    tf.app.run()