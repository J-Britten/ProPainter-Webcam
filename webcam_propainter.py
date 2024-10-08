import cv2
import numpy as np
import torch
import argparse
import os
import sys
from PIL import Image
import threading
import queue


from pixellib.torchbackend.instance import instanceSegmentation
from inference_propainter import (
    get_device, load_file_from_url, RAFT_bi, RecurrentFlowCompleteNet,
    InpaintGenerator, to_tensors, resize_frames
)

# Initialize PixelLib
segment_video = instanceSegmentation()

# Download PointRend model
pointrend_url = 'https://github.com/ayoolaolafenwa/PixelLib/releases/download/0.2.0/pointrend_resnet50.pkl'
pointrend_ckpt_path = load_file_from_url(url=pointrend_url, 
                                         model_dir='weights', 
                                         progress=True, 
                                         file_name='pointrend_resnet50.pkl')

segment_video.load_model(pointrend_ckpt_path, detection_speed="fast")
target_classes = segment_video.select_target_classes(car=True)

# Initialize ProPainter models
device = get_device()
pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'

# RAFT model
raft_ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'raft-things.pth'), 
                                    model_dir='weights', progress=True, file_name=None)
fix_raft = RAFT_bi(raft_ckpt_path, device)

# Flow completion model
flow_completion_ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'), 
                                               model_dir='weights', progress=True, file_name=None)
fix_flow_complete = RecurrentFlowCompleteNet(flow_completion_ckpt_path)
fix_flow_complete.to(device)
fix_flow_complete.eval()

# ProPainter model
propainter_ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'ProPainter.pth'), 
                                          model_dir='weights', progress=True, file_name=None)
model = InpaintGenerator(model_path=propainter_ckpt_path).to(device)
model.eval()

def process_frame(frame, mask):
    # Prepare inputs
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    mask_pil = Image.fromarray(mask)
    
    frame_resized, (w, h), _ = resize_frames([frame_pil], size=(640, 480))
    mask_resized, _, _ = resize_frames([mask_pil], size=(w, h))
    
    # Duplicate the current frame and mask
    frames = [frame_resized[0], frame_resized[0]]
    masks = [mask_resized[0], mask_resized[0]]
    
    frames = to_tensors()(frames).unsqueeze(0) * 2 - 1
    masks = to_tensors()(masks).unsqueeze(0)
    
    frames, masks = frames.to(device), masks.to(device)
    
    #print(f"Frames shape: {frames.shape}")
    #print(f"Masks shape: {masks.shape}")
    
    with torch.no_grad():
        flows_f, flows_b = fix_raft(frames, iters=20)
        flows_bi = (flows_f, flows_b)
        
        #print(f"Flows forward shape: {flows_f.shape}") 
        #print(f"Flows backward shape: {flows_b.shape}")
        
        pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(flows_bi, masks)
        pred_flows_bi = fix_flow_complete.combine_flow(flows_bi, pred_flows_bi, masks)
        
        #print(f"Pred flows bi shape: {pred_flows_bi[0].shape}, {pred_flows_bi[1].shape}")
        
        masked_frames = frames * (1 - masks)
        prop_imgs, updated_local_masks = model.img_propagation(masked_frames, pred_flows_bi, masks, 'nearest')
        b, t, _, _, _ = masks.size()
        updated_frames = frames * (1 - masks) + prop_imgs.view(b, t, 3, h, w) * masks
        
        #print(f"Updated frames shape: {updated_frames.shape}")
        #print(f"Updated local masks shape: {updated_local_masks.shape}")
        
        try:
            pred_img = model(updated_frames, pred_flows_bi, masks, updated_local_masks.view(b, t, 1, h, w), t)
            #print(f"Pred img shape: {pred_img.shape}")
        except Exception as e:
            #print(f"Error in model forward pass: {e}")
            #print(f"pred_flows_bi[0] shape: {pred_flows_bi[0].shape}")
            #print(f"pred_flows_bi[1] shape: {pred_flows_bi[1].shape}")
            #print(f"masks shape: {masks.shape}")
            #print(f"updated_local_masks shape: {updated_local_masks.shape}")
            import traceback
            traceback.print_exc()
            return frame_rgb
        
        pred_img = pred_img.view(-1, 3, h, w)
        pred_img = (pred_img + 1) / 2
        pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
        
    return pred_img[0].astype(np.uint8)

def propainter_thread(input_queue, output_queue):
    while True:
        item = input_queue.get()
        if item is None:
            break
        
        original_frame, combined_mask = item
        
        print("Valid mask found, processing with ProPainter")
        inpainted_frame = process_frame(original_frame, combined_mask)
        output_queue.put((combined_mask, inpainted_frame))

def main():
    capture = cv2.VideoCapture(0)
    
    # Get the frame size
    ret, frame = capture.read()
    if not ret:
        print("Failed to capture initial frame")
        return
    
    frame_height, frame_width = frame.shape[:2]
    
    # Initialize combined_mask as an empty frame
    combined_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    
    # Initialize variables for FPS calculation
    fps = 0
    frame_count = 0
    start_time = cv2.getTickCount()
    
    # Create queues for inter-thread communication
    input_queue = queue.Queue(maxsize=1)
    output_queue = queue.Queue()
    
    # Start the ProPainter thread
    propainter_thread_obj = threading.Thread(target=propainter_thread, args=(input_queue, output_queue))
    propainter_thread_obj.start()
    
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        
        # Create a copy of the original frame
        original_frame = frame.copy()
        
        # Calculate FPS
        frame_count += 1
        if frame_count >= 10:  # Update FPS every 10 frames
            end_time = cv2.getTickCount()
            elapsed_time = (end_time - start_time) / cv2.getTickFrequency()
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = cv2.getTickCount()
        
        results, output = segment_video.segmentFrame(frame, show_bboxes=True, segment_target_classes=target_classes)
        
        # Display FPS on the segmented frame
        cv2.putText(output, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Segmented Frame", output)
        cv2.imshow("Original Frame", original_frame)

        if results is not None and 'masks' in results:
            masks = results['masks']
            if masks is not None and len(masks.shape) == 3 and masks.shape[2] > 0:
                # Update combined_mask
                combined_mask = np.any(masks, axis=2)
            else:
                # Clear the combined_mask if no valid masks found
                combined_mask.fill(0)
        else:
            # Clear the combined_mask if no segmentation results
            combined_mask.fill(0)
        
        # Display the combined mask
        cv2.imshow("Combined Mask", (combined_mask * 255).astype(np.uint8))
        
        # Send to ProPainter thread if the queue is empty
        if input_queue.empty() and np.any(combined_mask):
            input_queue.put((original_frame, combined_mask))
        
        # Check for ProPainter results
        try:
            _, inpainted_frame = output_queue.get_nowait()
            cv2.imshow("Inpainted Frame", cv2.cvtColor(inpainted_frame, cv2.COLOR_RGB2BGR))
        except queue.Empty:
            pass
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Signal the ProPainter thread to exit
    input_queue.put(None)
    propainter_thread_obj.join()
    
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()