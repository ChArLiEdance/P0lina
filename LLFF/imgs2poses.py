from llff.poses.pose_utils import gen_poses
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--match_type', type=str, 
					default='exhaustive_matcher', help='type of matcher used.  Valid options: \
					exhaustive_matcher sequential_matcher.  Other matchers not supported at this time')
parser.add_argument('scenedir', type=str,
                    default="/mnt/mydisk1/lyh/P0lina/LLFF/sachelBag/",
                    help='input scene directory')
args = parser.parse_args()

if args.match_type != 'exhaustive_matcher' and args.match_type != 'sequential_matcher':
	print('ERROR: matcher type ' + args.match_type + ' is not valid.  Aborting')
	sys.exit()

if __name__=='__main__':
    gen_poses(args.scenedir, args.match_type)