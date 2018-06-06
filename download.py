#!/usr/bin/env python3

from subprocess import check_output

def download_and_unzip(url):
    filename = "blob.zip"
    check_output(["wget", url, "-O", filename])
    check_output(["unzip", filename])
    check_output(["rm", filename])

url = "https://www.dropbox.com/s/15sjkv71mtr55bn/blob.zip?dl=1" # 5GB
download_and_unzip(url)
