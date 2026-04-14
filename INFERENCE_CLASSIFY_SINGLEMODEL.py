import os
import cv2
import numpy as np
import torch
import psutil
import time
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

# ==========================================
# --- CONFIGURATION ---
# ==========================================

# 1. Path to your single trained weights
MODEL_PATH = r"C:\Users\Xin Yun\Desktop\From OLD\Universiti Malaya\FYP\Training Code\runs\detect\train_CLASSIFY_1C_ONOFF\weights\best.pt"

# 2. Dataset Paths
IMAGES_DIR = r"C:\Users\Xin Yun\Desktop\From OLD\Universiti Malaya\CLASSIFY_1C_YOLOv8_ONOFF\test\images"
LABELS_DIR = r"C:\Users\Xin Yun\Desktop\From OLD\Universiti Malaya\CLASSIFY_1C_YOLOv8_ONOFF\test\labels"
OUTPUT_DIR = r"C:\Users\Xin Yun\Desktop\From OLD\Universiti Malaya\FYP\Single_Model_Results\ML_assignment_YOLO11"

# 3. Class Settings (2 Classes)
CLASS_NAMES = {0: "SL_OFF", 1: "SL_ON"}
ACTIVE_CLASSES = [0, 1]

# 4. Thresholds
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5 

# ==========================================

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
        ram = self.process.memory_info().rss / (1024 ** 2) #divide bytes by 1024^2 to get MB
        if ram > self.max_ram: self.max_ram = ram #store peak RAM usage
        if torch.cuda.is_available():
            gpu = torch.cuda.memory_allocated() / (1024 ** 2)
            if gpu > self.max_gpu: self.max_gpu = gpu #store peak GPU VRAM usage

    def print_stats(self, total_images): #summary table of total time, FPS, latency, and resource usage
        duration = self.end_time - self.start_time
        if duration <= 0: duration = 1e-6
        print(f"\n--- Resource & Performance ---")
        print(f"Total Inference Time: {duration:.2f}s")
        print(f"Processing Speed:     {total_images/duration:.2f} FPS")
        print(f"Average Latency:      {(duration/total_images)*1000:.2f} ms/image")
        print(f"Peak CPU RAM Usage:   {self.max_ram:.2f} MB")
        if torch.cuda.is_available():
            print(f"Peak GPU VRAM Usage:  {self.max_gpu:.2f} MB")

class MetricEvaluator:
    def __init__(self):
        self.stats = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Reads the ground-truth .txt label file for one image and converts YOLO's normalised xywh format back to pixel coordinates. 
    def process_batch(self, pred_boxes, pred_scores, pred_classes, gt_path, img_shape):
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

    # Then it matches the predicted boxes to the GT boxes using IoU and class labels, and stores TP/FP/FN stats for metric calculation later.
    def match_predictions(self, preds, scores, p_classes, gt_boxes, gt_classes, iou_thres=0.5):
        if len(preds) == 0: return 
        if len(gt_boxes) == 0:
            for i in range(len(preds)):
                self.stats.append((False, float(scores[i]), int(p_classes[i]), -1)) 
            return
        iou = self.box_iou(preds, gt_boxes) #calculate IoU between each predicted box and each GT box
        if iou.shape[0] > 0:
            iou_vals, gt_idx = iou.max(1) #for each predictions, find the GT box with highest IoU
            detected_gt = [] 
            for pred_i, (val, gt_i) in enumerate(zip(iou_vals, gt_idx)):
                if val >= iou_thres and gt_i not in detected_gt:
                    if p_classes[pred_i] == gt_classes[gt_i]:
                        detected_gt.append(gt_i.item())
                        # If IoU is above threshold and class matches, it's a True Positive
                        self.stats.append((True, float(scores[pred_i]), int(p_classes[pred_i]), int(gt_classes[gt_i])))
                    else:
                        # If IoU is above threshold but class does not match, it's a False Positive
                        self.stats.append((False, float(scores[pred_i]), int(p_classes[pred_i]), int(gt_classes[gt_i])))
                else:
                    # If IoU is below threshold, it's a False Positive
                    self.stats.append((False, float(scores[pred_i]), int(p_classes[pred_i]), -1)) 

    # The box_iou function computes the Intersection over Union (IoU) between two sets of boxes, which is essential for determining how well the predicted boxes match the ground truth boxes.
    def box_iou(self, box1, box2):
        def box_area(box): return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        area1, area2 = box_area(box1), box_area(box2)
        lt = torch.max(box1[:, None, :2], box2[:, :2]) #top left corner of intersection
        rb = torch.min(box1[:, None, 2:], box2[:, 2:]) #bottom right corner of intersection
        wh = (rb - lt).clamp(min=0) #width and height of intersection, 0 if no overlap
        inter = wh[:, :, 0] * wh[:, :, 1] #intersection area
        return inter / (area1[:, None] + area2 - inter)

# --- MAIN ---
def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    labels_out_dir = Path(OUTPUT_DIR) / "labels"
    labels_out_dir.mkdir(parents=True, exist_ok=True)

    monitor = ResourceMonitor()
    evaluator = MetricEvaluator()
    
    print(f"Loading Model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # Initialize counters for 2 classes
    total_gt_per_class = {0: 0, 1: 0}
    tn_per_class = {0: 0, 1: 0}
    images_with_gt_per_class = {0: 0, 1: 0}

    #collect all image files from the TEST directory (IMAGES_DIR) 
    image_files = list(Path(IMAGES_DIR).glob("*.jpg")) + list(Path(IMAGES_DIR).glob("*.png"))
    
    print(f"Running evaluation on {len(image_files)} images...")
    monitor.start_timer()

    for img_path in tqdm(image_files, desc="Inference"):
        monitor.update()
        img_id = img_path.name
        
        # Inference
        results = model(str(img_path), conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)[0]
        
        pred_boxes = results.boxes.xyxy.cpu().numpy()
        pred_scores = results.boxes.conf.cpu().numpy()
        pred_classes = results.boxes.cls.cpu().numpy()
        h, w = results.orig_shape

        # Save Text Labels
        txt_name = labels_out_dir / (img_path.stem + ".txt")
        with open(txt_name, 'w') as f:
            for box, score, cls in zip(pred_boxes, pred_scores, pred_classes):
                x1, y1, x2, y2 = box
                x_c, y_c = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                f.write(f"{int(cls)} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")

        # Visual Output
        img = cv2.imread(str(img_path))
        for box, score, cls in zip(pred_boxes, pred_scores, pred_classes):
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if cls == 1 else (0, 0, 255) 
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{CLASS_NAMES.get(int(cls), cls)} {score:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.imwrite(str(Path(OUTPUT_DIR) / img_id), img)

        # Count Ground Truth
        gt_path = Path(LABELS_DIR) / (img_path.stem + ".txt")
        gt_classes_in_img = set()
        
        if gt_path.exists():
            with open(gt_path, 'r') as f:
                for line in f:
                    cid = int(line.split()[0])
                    gt_classes_in_img.add(cid)
                    if cid in total_gt_per_class:
                        total_gt_per_class[cid] += 1
        
        for cls in ACTIVE_CLASSES:
            if cls in gt_classes_in_img:
                images_with_gt_per_class[cls] += 1

        # Calculate Box Metrics (TP/FP/FN)
        evaluator.process_batch(pred_boxes, pred_scores, pred_classes, gt_path, (h, w))

        # --- TN Calculation Logic ---
        pred_classes_set = set(pred_classes.astype(int)) if len(pred_classes) > 0 else set()
        
        for cls in ACTIVE_CLASSES:
            # TN = Not Predicted AND Not in GT
            if (cls not in pred_classes_set) and (cls not in gt_classes_in_img):
                tn_per_class[cls] += 1

    monitor.stop_timer()

    # --- 4. Final Metric Calculation ---
    print("\n" + "="*110)
    print(f"{'CLASS':<12} {'TP':<5} {'FP':<5} {'FN':<5} {'TN':<5} {'PREC':<8} {'RECALL':<8} {'F1':<8} {'ACC':<8} {'AP-50':<8}")
    print("="*110)

    all_stats = np.array(evaluator.stats)
    aps = []
    
    for cls_id in ACTIVE_CLASSES: 
        cls_mask = (all_stats[:, 2] == cls_id) if len(all_stats) > 0 else []
        cls_stats = all_stats[cls_mask]
        total_gt = total_gt_per_class.get(cls_id, 0)
        
        if total_gt == 0 and len(cls_stats) == 0: # If there are no GT and no predictions for this class, it's a special case where we consider all images as True Negatives for this class.
            tn = len(image_files)
            acc = 1.0
            print(f"{CLASS_NAMES[cls_id]:<12} 0     0     0     {tn:<5} 0.0000   0.0000   0.0000   {acc:<8.4f} 0.0000")
            aps.append(0.0)
            continue

        if len(cls_stats) > 0: # If there are predictions for this class, calculate metrics
            i = np.argsort(-cls_stats[:, 1])
            tp_sorted = cls_stats[i, 0]
            tp_cumsum = np.cumsum(tp_sorted)
            fp_cumsum = np.cumsum(1 - tp_sorted)
            
            recalls = tp_cumsum / (total_gt + 1e-6)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
            
            ap = np.trapezoid(precisions, recalls)
            aps.append(ap)
            
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
            best_i = np.argmax(f1_scores)
            
            p, r, f1 = precisions[best_i], recalls[best_i], f1_scores[best_i]
            tp, fp = int(tp_cumsum[best_i]), int(fp_cumsum[best_i])
            fn = int(total_gt - tp)
            tn = tn_per_class[cls_id]
            
            # Accuracy Calculation
            total_instances = tp + fp + fn + tn
            acc = (tp + tn) / total_instances if total_instances > 0 else 0
            
            print(f"{CLASS_NAMES[cls_id]:<12} {tp:<5} {fp:<5} {fn:<5} {tn:<5} {p:<8.4f} {r:<8.4f} {f1:<8.4f} {acc:<8.4f} {ap:<8.4f}")
        
        else:
            tn = len(image_files) - images_with_gt_per_class.get(cls_id, 0)
            acc = tn / (tn + total_gt) if (tn + total_gt) > 0 else 0
            print(f"{CLASS_NAMES[cls_id]:<12} 0     0     {total_gt:<5} {tn:<5} 0.0000   0.0000   0.0000   {acc:<8.4f} 0.0000")
            aps.append(0.0)

    map50 = np.mean(aps) if aps else 0
    print("="*110)
    print(f"{'mAP-50:':<89} {map50:.4f}")
    print("="*110)
    
    monitor.print_stats(len(image_files))

if __name__ == "__main__":
    main()