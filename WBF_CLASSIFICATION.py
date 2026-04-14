#edit line 17-22 (for getting different file path),
# line 26 (for 1 class); line 27 for 2 classes
# line 158--2 class: {0: 0, 1: 0}; 1 class: {0:0}
# line 234 change to [0] only if 1 class; [0,1] for 2 classes
# line 193 Chang CONF_THRESH=0.25, FINAL_CONF_THRESH=0.25!!!! IoU_THRESH=0.5 keep same (for mAP50)
import os
import cv2
import numpy as np
import torch
import psutil
import time
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_1_PATH = r"C:\Users\Xin Yun\Desktop\From OLD\Universiti Malaya\FYP\YOLOv8l\runs\detect\train_CLASSIFY_1C_ONOFF\weights\best.pt"
MODEL_2_PATH = r"C:\Users\Xin Yun\Desktop\From OLD\Universiti Malaya\FYP\Training Code\runs\detect\train_CLASSIFY_1C_ONOFF\weights\best.pt"

IMAGES_DIR = r"C:\Users\Xin Yun\Desktop\From OLD\Universiti Malaya\CLASSIFY_1C_YOLOv8_ONOFF\test\images"
LABELS_DIR = r"C:\Users\Xin Yun\Desktop\From OLD\Universiti Malaya\CLASSIFY_1C_YOLOv8_ONOFF\test\labels"
OUTPUT_DIR = r"C:\Users\Xin Yun\Desktop\From OLD\Universiti Malaya\FYP\Ensemble_Results\MLassignment"

# Map your class IDs to names here for cleaner output
#CLASS_NAMES = {0: "streetlight"} #for 1 class
CLASS_NAMES = {0: "SL_OFF", 1: "SL_ON"} #for 2 classes

# --- HELPER CLASSES ---

class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.max_ram = 0
        self.max_gpu = 0
        self.start_time = 0
        self.end_time = 0

    def start_timer(self):
        self.start_time = time.time()

    def stop_timer(self):
        self.end_time = time.time()

    def update(self):
        ram = self.process.memory_info().rss / (1024 ** 2)
        if ram > self.max_ram: self.max_ram = ram
        if torch.cuda.is_available():
            gpu = torch.cuda.memory_allocated() / (1024 ** 2)
            if gpu > self.max_gpu: self.max_gpu = gpu

    def print_stats(self, total_images):
        duration = self.end_time - self.start_time
        if duration <= 0: duration = 1e-6
        print(f"\n--- Resource & Performance ---")
        print(f"Time: {duration:.2f}s | Speed: {total_images/duration:.2f} FPS | Latency: {(duration/total_images)*1000:.2f} ms")
        print(f"Peak RAM: {self.max_ram:.2f} MB | Peak VRAM: {self.max_gpu:.2f} MB")

class MetricEvaluator:
    def __init__(self):
        # Stats list: (is_tp, confidence, pred_class, target_class)
        self.stats = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def process_batch(self, pred_boxes, pred_scores, pred_classes, gt_path, img_shape):
        """ Compare predictions for one image against its Ground Truth file. """
        labels = []
        if os.path.exists(gt_path):
            with open(gt_path, 'r') as f:
                for line in f:
                    c, x, y, w, h = map(float, line.strip().split())
                    h_img, w_img = img_shape[:2]
                    x1, y1, x2, y2 = (x - w/2)*w_img, (y - h/2)*h_img, (x + w/2)*w_img, (y + h/2)*h_img
                    labels.append([int(c), x1, y1, x2, y2])
        
        gt_boxes = torch.tensor([x[1:] for x in labels], device=self.device)
        gt_classes = torch.tensor([x[0] for x in labels], device=self.device)
        
        preds = torch.tensor(pred_boxes, device=self.device)
        scores = torch.tensor(pred_scores, device=self.device)
        p_classes = torch.tensor(pred_classes, device=self.device)

        self.match_predictions(preds, scores, p_classes, gt_boxes, gt_classes)

    def match_predictions(self, preds, scores, p_classes, gt_boxes, gt_classes, iou_thres=0.5):
        if len(preds) == 0: return 
        if len(gt_boxes) == 0:
            for i in range(len(preds)):
                self.stats.append((False, float(scores[i]), int(p_classes[i]), -1)) 
            return

        iou = self.box_iou(preds, gt_boxes)
        
        if iou.shape[0] > 0:
            iou_vals, gt_idx = iou.max(1)
            detected_gt = []
            for pred_i, (val, gt_i) in enumerate(zip(iou_vals, gt_idx)):
                if val >= iou_thres and gt_i not in detected_gt:
                    if p_classes[pred_i] == gt_classes[gt_i]:
                        detected_gt.append(gt_i.item())
                        self.stats.append((True, float(scores[pred_i]), int(p_classes[pred_i]), int(gt_classes[gt_i])))
                    else:
                        self.stats.append((False, float(scores[pred_i]), int(p_classes[pred_i]), int(gt_classes[gt_i])))
                else:
                    self.stats.append((False, float(scores[pred_i]), int(p_classes[pred_i]), -1)) 

    def box_iou(self, box1, box2):
        def box_area(box): return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        area1, area2 = box_area(box1), box_area(box2)
        lt = torch.max(box1[:, None, :2], box2[:, :2])
        rb = torch.min(box1[:, None, 2:], box2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        return inter / (area1[:, None] + area2 - inter)

# --- WBF FUNCTION ---
#function takes prediction from both models plus image sizes and three thresholds (IOU_THRESH, CONF_THRESH, FINAL_CONF_THRESH) that control filtering
def perform_weighted_boxes_fusion(pred_confs, pred_boxes, pred_classes, resolution_dict, IOU_THRESH=0.5, CONF_THRESH=0.001, FINAL_CONF_THRESH=0.1):
    wbf_boxes_dict, wbf_scores_dict, wbf_classes_dict = {}, {}, {}
    for image_id, res in resolution_dict.items():
        res_array = np.array([res[1], res[0], res[1], res[0]]) 
        all_boxes, all_scores, all_classes = [], [], []

        for boxes, scores, classes in zip(pred_boxes, pred_confs, pred_classes):
            if image_id not in boxes:
                all_boxes.append(np.array([])); all_scores.append(np.array([])); all_classes.append(np.array([]))
                continue
            all_boxes.append((boxes[image_id] / res_array).clip(0,1))
            all_scores.append(scores[image_id])
            all_classes.append(classes[image_id])

        f_boxes, f_scores, f_labels = weighted_boxes_fusion(all_boxes, all_scores, all_classes, weights=None, iou_thr=IOU_THRESH, skip_box_thr=CONF_THRESH)
        
        keep = f_scores > FINAL_CONF_THRESH
        f_boxes = (f_boxes[keep] * res_array).astype(int)
        
        wbf_boxes_dict[image_id] = f_boxes
        wbf_scores_dict[image_id] = f_scores[keep]
        wbf_classes_dict[image_id] = f_labels[keep]

    return wbf_boxes_dict, wbf_scores_dict, wbf_classes_dict

# --- MAIN ---
def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    labels_out_dir = Path(OUTPUT_DIR) / "labels"
    labels_out_dir.mkdir(parents=True, exist_ok=True)

    monitor = ResourceMonitor()
    evaluator = MetricEvaluator()
    
    print("Loading models...")
    model1 = YOLO(MODEL_1_PATH)
    model2 = YOLO(MODEL_2_PATH)

    # Before running inference, we will collect predictions from both models for all images and store them in dictionaries. This way, we can perform WBF in a single pass after collecting all predictions, which is more efficient than doing it image by image.
    m1_box, m1_conf, m1_cls = {}, {}, {}
    m2_box, m2_conf, m2_cls = {}, {}, {}
    res_dict = {}
    total_gt_per_class = {0: 0, 1: 0} #2 class: {0: 0, 1: 0}; 1 class: {0:0}
    images_with_gt_per_class = {0: 0, 1: 0}
    tn_per_class = {0: 0, 1: 0}

    image_files = list(Path(IMAGES_DIR).glob("*.jpg")) + list(Path(IMAGES_DIR).glob("*.png"))
    print(f"Running inference on {len(image_files)} images...")
    
    monitor.start_timer()

    for img_path in tqdm(image_files, desc="Inference"):
        monitor.update()
        img_id = img_path.name
        
        # Run inference for both models and store predictions in dictionaries
        #Model 1 inference
        r1 = model1(str(img_path), verbose=False)[0]
        m1_box[img_id] = r1.boxes.xyxy.cpu().numpy()
        m1_conf[img_id] = r1.boxes.conf.cpu().numpy()
        m1_cls[img_id] = r1.boxes.cls.cpu().numpy()
        res_dict[img_id] = r1.orig_shape

        #Model 2 inference
        r2 = model2(str(img_path), verbose=False)[0]
        m2_box[img_id] = r2.boxes.xyxy.cpu().numpy()
        m2_conf[img_id] = r2.boxes.conf.cpu().numpy()
        m2_cls[img_id] = r2.boxes.cls.cpu().numpy()

        gt_file = Path(LABELS_DIR) / (img_path.stem + ".txt")
        gt_classes = set()
        if gt_file.exists():
            with open(gt_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    cls_id = int(line.split()[0])
                    gt_classes.add(cls_id)
                    if cls_id in total_gt_per_class:
                        total_gt_per_class[cls_id] += 1
        for cls in [0,1]:
            if cls in gt_classes:
                images_with_gt_per_class[cls] += 1

    monitor.stop_timer()

    print("Performing Weighted Box Fusion...")

    #Calls the WBF function with both models' predictions packaged as lists. 
    #Returns one clean set of fused boxes per image.
    wbf_boxes, wbf_scores, wbf_classes = perform_weighted_boxes_fusion(
        [m1_conf, m2_conf], [m1_box, m2_box], [m1_cls, m2_cls],
        res_dict, IOU_THRESH=0.5, CONF_THRESH=0.25, FINAL_CONF_THRESH=0.25
    )

    print("Saving labels and calculating metrics...")

    #Use the fused predictions to save label files and annotated images, 
    #and also feed them into the MetricEvaluator for final metric calculation.
    for img_path in tqdm(image_files, desc="Processing"):
        img_id = img_path.name
        if img_id not in wbf_boxes: continue

        boxes = wbf_boxes[img_id]
        scores = wbf_scores[img_id]
        classes = wbf_classes[img_id]
        h, w = res_dict[img_id]

        txt_name = labels_out_dir / (img_path.stem + ".txt")
        with open(txt_name, 'w') as f:
            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box
                x_c, y_c = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
                w_n, h_n = (x2 - x1) / w, (y2 - y1) / h
                f.write(f"{int(cls)} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

        img = cv2.imread(str(img_path))
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if cls == 1 else (0, 0, 255) 
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{CLASS_NAMES.get(int(cls), cls)} {score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.imwrite(str(Path(OUTPUT_DIR) / img_id), img)

        gt_path = Path(LABELS_DIR) / (img_path.stem + ".txt")
        evaluator.process_batch(boxes, scores, classes, gt_path, (h, w))

        pred_classes_set = set(classes.astype(int)) if len(classes) > 0 else set()
        gt_classes = set()
        if gt_path.exists():
            with open(gt_path, 'r') as f:
                for line in f:
                    cls_id = int(line.split()[0])
                    gt_classes.add(cls_id)
        for cls in [0,1]:
            if cls not in pred_classes_set and cls not in gt_classes:
                tn_per_class[cls] += 1

    # --- 4. Final Metric Calculation (PER CLASS + mAP) ---
    print("\n" + "="*110)
    print(f"{'CLASS':<12} {'TP':<5} {'FP':<5} {'FN':<5} {'TN':<5} {'PREC':<8} {'RECALL':<8} {'F1':<8} {'ACC':<8} {'AP-50':<8}")
    print("="*110)

    all_stats = np.array(evaluator.stats)
    aps = []
    
    if len(all_stats) > 0:
        for cls_id in [0,1]: #1 class: [0]; 2 classes: [0,1]
            cls_mask = (all_stats[:, 2] == cls_id)
            cls_stats = all_stats[cls_mask]
            total_gt = total_gt_per_class.get(cls_id, 0)
            
            if total_gt == 0 and len(cls_stats) == 0:
                tn = len(image_files)
                acc = 1.0
                print(f"{CLASS_NAMES[cls_id]:<12} 0     0     0     {tn:<5} 0.0000   0.0000   0.0000   {acc:<8.4f} 0.0000")
                aps.append(0.0)
                continue

            if len(cls_stats) > 0:
                i = np.argsort(-cls_stats[:, 1])
                tp_sorted = cls_stats[i, 0]
                tp_cumsum = np.cumsum(tp_sorted)
                fp_cumsum = np.cumsum(1 - tp_sorted)
                
                recalls = tp_cumsum / (total_gt + 1e-6)
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
                
                # Calculate AP (Area Under Precision-Recall Curve)
                ap = np.trapezoid(precisions, recalls)
                aps.append(ap)
                
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
                best_i = np.argmax(f1_scores)
                
                p, r, f1 = precisions[best_i], recalls[best_i], f1_scores[best_i]
                tp, fp = int(tp_cumsum[best_i]), int(fp_cumsum[best_i])
                fn = int(total_gt - tp)
                tn = tn_per_class[cls_id]
                acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                
                print(f"{CLASS_NAMES[cls_id]:<12} {tp:<5} {fp:<5} {fn:<5} {tn:<5} {p:<8.4f} {r:<8.4f} {f1:<8.4f} {acc:<8.4f} {ap:<8.4f}")
            else:
                tn = len(image_files) - images_with_gt_per_class.get(cls_id, 0)
                acc = (0 + tn) / (0 + tn + 0 + total_gt) if (tn + total_gt) > 0 else 0
                print(f"{CLASS_NAMES[cls_id]:<12} 0     0     {total_gt:<5} {tn:<5} 0.0000   0.0000   0.0000   {acc:<8.4f} 0.0000")
                aps.append(0.0)

    map50 = np.mean(aps) if aps else 0
    print("="*110)
    print(f"{'mAP-50 (Mean of AP column):':<85} {map50:.4f}")
    print("="*110)
    
    monitor.print_stats(len(image_files))

if __name__ == "__main__":
    main()