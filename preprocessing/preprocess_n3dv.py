import os
import argparse
import cv2
from tqdm import tqdm
import multiprocessing as mp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess N3DV dataset")
    parser.add_argument("--data_dir", type=str, default="data/n3d/flame_salmon", help="Path to N3DV dataset")
    parser.add_argument("--output_dir", type=str, default="omnimotion", help="Path to save preprocessed data")
    parser.add_argument("--resize", type=float, default=1.0, help="resize ratio for images")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for parallel processing")
    args = parser.parse_args()

    images_dir = os.path.join(args.data_dir, "images")
    cams = sorted(os.listdir(images_dir))
    for cam in tqdm(cams):
        os.makedirs(os.path.join(args.data_dir, args.output_dir, cam, "color"), exist_ok=True)
        images = sorted(os.listdir(os.path.join(images_dir, cam)))
        
        def resize_image(image):
            image_path = os.path.join(images_dir, cam, image)
            output_path = os.path.join(args.data_dir, args.output_dir, cam, "color", image)
            img = cv2.imread(image_path)
            img = cv2.resize(img, None, fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA)
            cv2.imwrite(output_path, img)

        with mp.Pool(processes=args.num_workers) as pool:
            pool.map(resize_image, images)