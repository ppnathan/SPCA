#!/usr/bin/env python

import cv
import extract_surf

def parse():
    import argparse
    parser = argparse.ArgumentParser(description = "SURF feature extraction batch")
    parser.add_argument('imagepathfile', type = str, help = 'input images path file')
    parser.add_argument('-o', '--outdir', dest = 'out_dir', type = str, default = '', help = 'output directory, default is the same folder of the input image')
    return parser.parse_args()


def main(args):
    imagefile = open(args.imagepathfile, 'r')
    for image_path in imagefile.readlines():
        image_path = image_path.strip('\n')
        if args.out_dir == '':
            extract_surf.extract_surf(image_path, isWriteFile = 1)
        else:
            extract_surf.extract_surf(image_path, isWriteFile = 1, out_path = args.out_dir)


if __name__ == '__main__':
    args = parse()
    main(args)