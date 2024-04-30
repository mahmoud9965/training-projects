import cv2
from PIL import Image
import torch
from torchvision.transforms import functional as F
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

def detect_objects(img_path, model, device, conf_threshold=0.4, iou_threshold=0.5):
    img0 = cv2.imread(img_path)  
    img = Image.fromarray(img0)
    img_tensor = F.to_tensor(img).unsqueeze(0).to(device)  
    model.eval()
    with torch.no_grad():
        pred = model(img_tensor)[0] 

    
    pred = non_max_suppression(pred, conf_threshold, iou_threshold)[0]  
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img.shape[1:], pred[:, :4], img0.shape).round()

    return pred

if __name__ == "__main__":
    
    weights = 'yolov5s.pt'  
    model = attempt_load(weights, map_location=torch.device('cpu')).fuse().eval()  
    device = select_device('')  

   
    cap = cv2.VideoCapture(0)  
    while cap.isOpened():
        ret, frame = cap.read()  
        if not ret:
            break

        pred = detect_objects(frame, model, device) 

        if pred is not None and len(pred):
            for det in pred:
                xyxy = det[:4].cpu().numpy().astype(int)
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                label = f'{det[-1]} {det[-2]:.2f}'
                cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Real-time Object Detection', frame)  
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break

    cap.release()  
    cv2.destroyAllWindows()
