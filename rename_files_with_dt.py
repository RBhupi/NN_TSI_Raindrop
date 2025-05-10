#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 11:30:33 2022
"""
from pathlib import Path
import re
import subprocess
import logging
import argparse
from datetime import datetime


def extract_timestamp(path: Path):
    timestamp_str, filename = path.name.split("-", 1)
    timestamp = int(timestamp_str)/10**9
    dt = datetime.fromtimestamp(timestamp)
    dt_str = dt.strftime(format="%Y%m%d-%H%M%S")
    return dt_str


def rename_files(fname:Path):
    root_dir, file_name = fname.parent, fname.name
    dt_str = extract_timestamp(fname)
    file_name = re.sub("\d{19}", dt_str, file_name)
    fname_new = Path(root_dir).joinpath(file_name)
    fname.replace(fname_new)

def main(args):
    for f in args.dir.glob("*.jpg"):
        rename_files(f)

        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rename files with datetime"
    )
    parser.add_argument("--debug", action="store_true", help="enable debug logs")
    parser.add_argument(
        "-dir",
        dest="dir",
        type=Path,
        default="/Users/bhupendra/data/sage_data_camera/storage.sagecontinuum.org/api/v1/data/sage/sage-imagesampler-top-0.2.5/000048b02d15bc8c/",
        help="Directory for data.",
    )
    
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    main(args)