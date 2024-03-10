import os
import shutil
import sys
import subprocess
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='dataset dir')
    parser.add_argument('--frame_sample', type=int, nargs=2, default=[0, 50], help="interval of images to process")
    parser.add_argument('--model', default='models/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--cycle_th', type=float, default=3., help='threshold for cycle consistency error')
    parser.add_argument('--chain', action='store_true', help='if chaining cycle consistent flows (optional)')
    parser.add_argument('--K', type=int, default=5, help="number of top K flows to keep")

    args = parser.parse_args()

    # compute raft optical flows between all pairs
    os.chdir('RAFT')
    subprocess.run(['python', 'exhaustive_raft.py', 
                    '--data_dir', args.data_dir, 
                    '--model', args.model, 
                    '--frame_sample', str(args.frame_sample[0]), str(args.frame_sample[1]),
                    '--K', str(args.K),                   
                    ])

    # compute dino feature maps
    os.chdir('../dino')
    subprocess.run(['python', 'extract_dino_features.py', 
                    '--data_dir', args.data_dir, 
                    '--frame_sample', str(args.frame_sample[0]), str(args.frame_sample[1])])

    # filtering
    os.chdir('../RAFT')
    subprocess.run(['python', 'filter_raft.py', 
                    '--data_dir', args.data_dir, 
                    '--cycle_th', str(args.cycle_th), 
                    '--frame_sample', str(args.frame_sample[0]), str(args.frame_sample[1]),
                    '--K', str(args.K),
                    ])

    # chaining (optional)
    subprocess.run(['python', 'chain_raft.py', 
                    '--data_dir', args.data_dir, 
                    '--frame_sample', str(args.frame_sample[0]), str(args.frame_sample[1]),
                    '--K', str(args.K),
                    ])


