# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 將 ROOT 添加到 PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # 模型路徑或 triton URL
        source=ROOT / 'data/images',  # 文件/目錄/URL/glob/screen/0（網絡攝像頭）
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml路徑
        imgsz=(640, 640),  # 推理大小（高度, 寬度）
        conf_thres=0.25,  # confidence 閾值
        iou_thres=0.45,  # NMS IOU 閾值
        max_det=1000,  # 每張圖片的最大檢測數
        device='',  # cuda 設備，即 0 或 0,1,2,3 或 cpu
        view_img=False,  # 顯示結果
        save_txt=False,  # 將結果保存到 *.txt
        save_conf=False,  # 在 --save-txt 標籤中保存信心
        save_crop=False,  # 保存裁剪的預測框
        nosave=False,  # 不保存圖片/視頻
        classes=None,  # 按類別過濾：--class 0，或--class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # 增強推理
        visualize=False,  # 可視化特徵
        update=False,  # 更新所有模型
        project=ROOT / 'runs/detect',  # 將結果保存到項目/名稱
        name='exp',  # 將結果保存到項目/名稱
        exist_ok=False,  # 現有項目/名稱可以，不要增加
        line_thickness=3,  # 邊界框厚度（pixels像素）
        hide_labels=False,  # 隱藏標籤
        hide_conf=False,  # hide confidences
        half=False,  # 使用 FP16 半精度推理
        dnn=False,  # 使用 OpenCV DNN 進行 ONNX 推理
        vid_stride=1,  # 視頻幀率步幅
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # 保存推理圖像
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # 目錄
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # 加載模型
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # 檢查圖像大小

    # 數據加載器
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # 運行推理
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # 推理
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # 第二階段分類器（可選）
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # 過程預測
        for i, det in enumerate(pred):  # 每張圖片
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # 將框從 img_size 重新縮放為 im0 大小
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print結果
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # 每class檢測
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write結果
                for *xyxy, conf, cls in reversed(det):

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh # 歸一化的xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label 格式
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # 串流結果
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # 允許調整窗口大小 (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0) #(視窗名稱,顯示圖片)
                cv2.waitKey(1)  # 1 毫秒.

            # 保存結果（帶檢測的圖像）
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # 在結果視頻上強制 *.mp4 後綴
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # 提升在訓練時的效率，生成的"images"和"labels"可以用來加強訓練。
        # 條件是辨識新的圖片或影片，做為新的訓練樣本
        if len(det):
            ok_file="images" #【有】辨識到的圖，存放的資料夾名稱
            img_path = str(save_dir / ok_file / p.stem) +("_"+str(seen)+".jpg")  # im.txt
            if not os.path.isdir(str(save_dir / ok_file)):
                os.makedirs(str(save_dir / ok_file))
            cv2.imwrite(img_path,im0s) #(寫入路徑,圖片) #im0s為辨識前原始圖 #im0為辨識過後有眶圖
            box_file="images_box" #【有】辨識到的圖，存放的資料夾名稱
            img_path = str(save_dir / box_file / p.stem) +("_"+str(seen)+".jpg")  # im.txt
            if not os.path.isdir(str(save_dir / box_file)):
                os.makedirs(str(save_dir / box_file))
            cv2.imwrite(img_path,im0) #(寫入路徑,圖片) #im0s為辨識前原始圖 #im0為辨識過後有眶圖
        else:
            # 可以手動將沒有辨識到的圖，另外進行labelimg處理。
            no_file="no_images" #【沒有】辨識到的圖，存放的資料夾名稱
            img_path = str(save_dir / no_file / p.stem) +("_"+str(seen)+".jpg")  # im.txt
            if not os.path.isdir(str(save_dir / no_file)):
                os.makedirs(str(save_dir / no_file))
            cv2.imwrite(img_path,im0s) #(寫入路徑,圖片) #im0s為辨識前原始圖 #im0為辨識過後有眶圖

        # Print時間（僅推理）
        LOGGER.info(f"{s}{'' if len(det) else '(No detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print結果
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image # 每幅圖像的速度
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # 更新模型（修復 SourceChangeWarning）


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='（可選）模型路徑或 triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='文件/目錄/URL/glob/screen/0（網絡攝像頭）')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='（可選）dataset.yaml 路徑')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[480,640], help='推理大小 h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence 閾值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU 閾值')
    parser.add_argument('--max-det', type=int, default=1000, help='每張圖片的最大檢測數')
    parser.add_argument('--device', default='', help='cuda 設備，即 0 或 0,1,2,3 或 cpu')
    parser.add_argument('--view-img', action='store_true', help='顯示辨識結果')
    parser.add_argument('--save-txt', action='store_true', help='將結果保存到 *.txt')
    parser.add_argument('--save-conf', action='store_true', help='在 --save-txt 標籤中保存信心')
    parser.add_argument('--save-crop', action='store_true', help='保存裁剪的預測框')
    parser.add_argument('--nosave', action='store_true', help='不保存圖片/視頻')
    parser.add_argument('--classes', nargs='+', type=int, help='按類別過濾：--classes 0，或--classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='增強推理')
    parser.add_argument('--visualize', action='store_true', help='可視化特徵')
    parser.add_argument('--update', action='store_true', help='更新所有模型')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='將結果保存到項目/名稱')
    parser.add_argument('--name', default='exp', help='將結果保存到項目/名稱')
    parser.add_argument('--exist-ok', action='store_true', help='現有項目/名稱可以，不要增加')
    parser.add_argument('--line-thickness', default=1, type=int, help='邊界框厚度（pixels像素）')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='隱藏標籤')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='隱藏confidences')
    parser.add_argument('--half', action='store_true', help='使用FP16半精度推斷')
    parser.add_argument('--dnn', action='store_true', help='使用OpenCV DNN進行ONNX推斷')
    parser.add_argument('--vid-stride', type=int, default=1, help='視頻幀率步幅')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
