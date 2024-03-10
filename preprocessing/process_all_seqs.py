import argparse
import subprocess
import os


def main(args):
    for cam in args.cams:
        data_dir = f'{args.data_dir}/{cam}'
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        try:
            subprocess.run(['python3', 'main_processing.py', '--chain',
                            '--data_dir', data_dir,
                            '--frame_sample', str(args.frame_sample[0]), str(args.frame_sample[1]),
                            '--K', str(args.K),
                            ])
        except:
            print(f'Error in processing {data_dir}')
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess N3DV dataset")
    parser.add_argument("--data_dir", type=str, help="Path to the data directory")
    parser.add_argument("--cams", nargs="+", type=str, help="Sequence numbers to preprocess")
    parser.add_argument('--gpu', type=int, default=0, help='GPU number')
    parser.add_argument('--frame_sample', type=int, nargs=2, default=[0, 50], help='Frame range to sample')
    parser.add_argument('--K', type=int, default=5, help='Number of top K flows to keep')
    args = parser.parse_args()
    main(args)