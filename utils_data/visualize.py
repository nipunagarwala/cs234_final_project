from PIL import Image, ImageDraw
import os
import cPickle as pickle
import numpy as np

mean_pixel = 128# VOT
orig_shape = (180, 320)
# VTB

def generate_images(sequence, predicted_bboxes, true_bboxes, save_path, basename):
    i = 0
    for s in sequence:
        img = np.uint8(s + mean_pixel)
        img = Image.fromarray(img)
        predicted_bbox = predicted_bboxes[i]
        true_bbox = true_bboxes[i]
        predicted_bbox = [predicted_bbox[1] * orig_shape[1], predicted_bbox[0] * orig_shape[0], (predicted_bbox[3]+predicted_bbox[1])* orig_shape[1], (predicted_bbox[2]+predicted_bbox[0])* orig_shape[0]]
        true_bbox = [true_bbox[1] * orig_shape[1], true_bbox[0] * orig_shape[0], (true_bbox[3]+true_bbox[1])* orig_shape[1], (true_bbox[2]+true_bbox[0])* orig_shape[0]]
        print predicted_bbox, true_bbox
        
        draw = ImageDraw.Draw(img)
        draw.rectangle(predicted_bbox)
        draw.rectangle(true_bbox, outline=(255,0,0,255))
        img.save(os.path.join(save_path, basename + str(i) +".png"))
        i += 1

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