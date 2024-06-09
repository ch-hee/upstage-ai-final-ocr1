import json
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from collections import OrderedDict
import cv2 

Image.MAX_IMAGE_PIXELS = 108000000
EXIF_ORIENTATION = 274  # Orientation Information: 274


class OCRDataset(Dataset):
    def __init__(self, image_path, annotation_path, bbox_path, transform):
        self.image_path = Path(image_path)
        self.transform = transform
        self.anns = OrderedDict()
        self.bbox = OrderedDict()
        
        if bbox_path:
            with open(bbox_path, 'r') as f:
                bboxes = json.load(f)
                for filename in bboxes.keys():
                # Image file이 경로에 존재하는지 확인
                    if (self.image_path / filename).exists():
                        # words 정보를 가지고 있는지 확인
                        self.bbox[filename] = bboxes[filename]
        
        # annotation_path가 없다면, image_path에서 이미지만 불러오기
        if annotation_path is None:
            for ext in ['jpg', 'jpeg', 'png']:
                for file in self.image_path.glob(f'*.{ext}'):
                    if file.suffix.lower() == f'.{ext}':
                        self.anns[file.name] = None
            return

        with open(annotation_path, 'r') as f:
            annotations = json.load(f)

            for filename in annotations['images'].keys():
                # Image file이 경로에 존재하는지 확인
                if (self.image_path / filename).exists():
                    # words 정보를 가지고 있는지 확인
                    if 'words' in annotations['images'][filename]:
                        # Words의 Points 변환
                        gt_words = annotations['images'][filename]['words']
                        polygons = [np.array([np.round(word_data['points'])], dtype=np.int32)
                                    for word_data in gt_words.values()
                                    if len(word_data['points'])]
                        self.anns[filename] = polygons
                    else:
                        self.anns[filename] = None

    def __len__(self):
        return len(self.anns.keys())

    def __getitem__(self, idx):
        image_filename = list(self.anns.keys())[idx]
        image = Image.open(self.image_path / image_filename).convert('RGB')

        # EXIF정보를 확인하여 이미지 회전
        exif = image.getexif()
        if exif:
            if EXIF_ORIENTATION in exif:
                image = OCRDataset.rotate_image(image, exif[EXIF_ORIENTATION])
        org_shape = image.size

        polygons = self.anns[image_filename] or None

        # 영수증 부분만 crop 하기 
        image = np.array(image)
        ###################################################
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 히스토그램 평활화를 수행
        equalized_gray_image = cv2.equalizeHist(gray_image)

        # 그레이스케일 이미지를 다시 3차원 컬러 이미지로 변환
        image = cv2.cvtColor(equalized_gray_image, cv2.COLOR_GRAY2BGR)
        ####################################################
        h, w, c = image.shape 
        if polygons:
            new_min_x, new_min_y, new_max_x, new_max_y = self.get_crop_coord_use_gt(image, polygons)
            crop_image = image[new_min_y:new_max_y, new_min_x:new_max_x, :]
            
            croped_polygons = []
            for polygon in polygons:
                croped_polygon = polygon.copy()
                croped_polygon[:, :, 0]  -= new_min_x
                croped_polygon[:, :, 1]  -= new_min_y
                croped_polygons.append(croped_polygon)

            item = OrderedDict(image=crop_image, image_filename=image_filename, shape=org_shape, min_coordinates = (new_min_x, new_min_y))

            if self.transform is None:
                raise ValueError("Transform function is a required value.")

            # Image transform
            transformed = self.transform(image=crop_image, polygons=croped_polygons)
            item.update(image=transformed['image'],
                        polygons=transformed['polygons'],
                        inverse_matrix=transformed['inverse_matrix'],
                        )
        else:
            bbox = self.bbox[image_filename] or None

            new_min_x, new_min_y, new_max_x, new_max_y = bbox

            new_min_x = max(new_min_x, 0)
            new_min_y = max(new_min_y, 0)
            new_max_x = min(new_max_x, w)
            new_max_y = min(new_max_y, h)

            crop_image = image[new_min_y:new_max_y, new_min_x:new_max_x, :]
            item = OrderedDict(image=crop_image, image_filename=image_filename, shape=org_shape, min_coordinates = (new_min_x, new_min_y))

            if self.transform is None:
                raise ValueError("Transform function is a required value.")

            # Image transform
            transformed = self.transform(image=np.array(crop_image), polygons=polygons)
            item.update(image=transformed['image'],
                        polygons=transformed['polygons'],
                        inverse_matrix=transformed['inverse_matrix'],
                        )

        return item

    def get_crop_coord_use_gt(self, image, polygons):
        min_x, min_y = min(p[0][0][0] for p in polygons), min(p[0][0][1] for p in polygons)
        max_x, max_y = max(p[0][2][0] for p in polygons), max(p[0][2][1] for p in polygons)

        h, w, c = image.shape

        crop_w, crop_h = max_x - min_x, max_y - min_y

        new_min_x = 0 if min_x - ((w - crop_w) / 10) < 0 else min_x - ((w - crop_w) / 10) 
        new_max_x = w if max_x + ((w - crop_w) / 10) > w else max_x + ((w - crop_w) / 10)

        new_min_y = 0 if min_y - ((h - crop_h) / 4) < 0 else min_y - ((h - crop_h) / 4) 
        new_max_y = h if max_y + ((h - crop_h) / 4) > h else max_y + ((h - crop_h) / 4)

        new_min_x, new_min_y, new_max_x, new_max_y = int(new_min_x), int(new_min_y), int(new_max_x), int(new_max_y)
        
        return [new_min_x, new_min_y, new_max_x, new_max_y]
    
    """
    cv를 이용하여 영수증 부분만 crop한 부분 
    gt를 이용한 방법보다 안좋은거 같음 
    """
    # def get_crop_coord_use_cv2(image):
    #     blurred = cv2.GaussianBlur(image, (7, 7), 1.75)
    #     edges = cv2.Canny(blurred, 50, 200)

    #     contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     contours = sorted(contours, key=cv2.contourArea, reverse=True)

    #     bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    #     x, y, w, h = max(bounding_boxes, key=lambda bbox: bbox[2] * bbox[3])

    #     return [x, y, x + w, y + h]
    
    @staticmethod
    def rotate_image(image, orientation):
        if orientation == 2:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            image = image.rotate(180)
        elif orientation == 4:
            image = image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            image = image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            image = image.rotate(-90, expand=True)
        elif orientation == 7:
            image = image.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            image = image.rotate(90, expand=True)
        return image
