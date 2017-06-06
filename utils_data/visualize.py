from PIL import Image, ImageDraw
import os
import cPickle as pickle
import numpy as np

mean_pixel = [114.78968156, 114.81268655, 113.29880948] # [ 116.45473115, 112.27971193, 106.51900657] # VOT
orig_shape = (180, 320)
# VTB # 114.78968156  114.81268655  113.29880948]

def generate_images(sequence, predicted_bboxes, true_bboxes, save_path, basename):
    mean_pixel_array = np.ones_like(sequence[0], dtype='float64')
    mean_pixel_array[:, :, 0] *= mean_pixel[0]
    mean_pixel_array[:, :, 1] *= mean_pixel[1]
    mean_pixel_array[:, :, 2] *= mean_pixel[2]
    for i in range(len(sequence)):
        s = sequence[i]
        img = np.uint8(s + mean_pixel)
        img = Image.fromarray(img)
        predicted_bbox = predicted_bboxes[i]
        true_bbox = true_bboxes[i]
        predicted_bbox = [predicted_bbox[1] * orig_shape[1], predicted_bbox[0] * orig_shape[0], (predicted_bbox[3]+predicted_bbox[1])* orig_shape[1], (predicted_bbox[2]+predicted_bbox[0])* orig_shape[0]]
        true_bbox = [true_bbox[1] * orig_shape[1], true_bbox[0] * orig_shape[0], (true_bbox[3]+true_bbox[1])* orig_shape[1], (true_bbox[2]+true_bbox[0])* orig_shape[0]]
        
        draw = ImageDraw.Draw(img)
        draw.rectangle(predicted_bbox)
        draw.rectangle(true_bbox, outline=(255,0,0,255))
        img.save(os.path.join(save_path, basename + str(i) +".png"))

def generate_multi_object_images(sequence, predicted_bboxes, true_bboxes, save_path, basename):
    mean_pixel_array = np.ones_like(sequence[0], dtype='float64')
    mean_pixel_array[:, :, 0] *= mean_pixel[0]
    mean_pixel_array[:, :, 1] *= mean_pixel[1]
    mean_pixel_array[:, :, 2] *= mean_pixel[2]
    for i in range(len(sequence)):
        s = sequence[i]
        img = np.uint8(s + mean_pixel)
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        
        for j in range(len(predicted_bboxes)):
            predicted_bbox = predicted_bboxes[j][i]
            true_bbox = true_bboxes[j][i]
            predicted_bbox = [predicted_bbox[1] * orig_shape[1], predicted_bbox[0] * orig_shape[0], (predicted_bbox[3]+predicted_bbox[1])* orig_shape[1], (predicted_bbox[2]+predicted_bbox[0])* orig_shape[0]]
            true_bbox = [true_bbox[1] * orig_shape[1], true_bbox[0] * orig_shape[0], (true_bbox[3]+true_bbox[1])* orig_shape[1], (true_bbox[2]+true_bbox[0])* orig_shape[0]]
            
            draw.rectangle(predicted_bbox)
            draw.rectangle(true_bbox, outline=(255,0,0,255))
        img.save(os.path.join(save_path, basename + str(i) +".png"))

if __name__ == "__main__":
    filename = "/data/vtb/data/test/BlurBody-0001"
    with open(filename, 'rb') as f:
        curr_seq, curr_labels, seq_len = pickle.load(f)
        curr_seq = np.asarray(curr_seq)
        curr_labels = np.asarray(curr_labels)
        pred_labels = np.copy(curr_labels)
        pred_labels[:, 0] += 0.1
        pred_labels[:, 1] += 0.1
        generate_images(curr_seq, pred_labels, curr_labels, "/data/testvisualize", "BlurBody")
        
        curr_labels = [curr_labels, pred_labels]
        pred_labels_obj1 = np.copy(pred_labels)
        pred_labels_obj1[:, 0] += 0.1
        pred_labels_obj1[:, 1] += 0.1
        pred_labels_obj2 = np.copy(pred_labels_obj1)
        pred_labels_obj2[:, 0] += 0.1
        pred_labels_obj2[:, 1] += 0.1
        pred_labels = [pred_labels_obj1, pred_labels_obj2]
        generate_multi_object_images(curr_seq, pred_labels, curr_labels, "/data/testvisualize", "MultiBlurBody")