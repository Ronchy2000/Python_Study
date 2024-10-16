##
# How to use it:
#  终端进入目录-
#     python 爬m3u8视频.py http://your_m3u8_url.m3u8  /path_to_your_local_dir

import os
import sys
import requests
from urllib.parse import urlparse

def get_cfg():
    argv = sys.argv
    if len(argv) <= 1:
        print('Usage: python3', argv[0], '[your_m3u8_url] [save_dir]')
        print('Sample: python3', argv[0], 'https://xxx.com/video.m3u8', '/Users/huzhenjie/Downloads/save_dir')
        return None
    return (argv[1], argv[2])

def get_host(url):
	urlgroup = urlparse(url)
	return urlgroup.scheme + '://' + urlgroup.hostname

def get_m3u8_body(url):
	print('read m3u8 file:', url)
	session = requests.Session()
	adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=10)
	session.mount('http://', adapter)
	session.mount('https://', adapter)
	r = session.get(url, timeout=10)
	return r.text

def get_url_list(host, body):
	lines = body.split('\n')
	ts_url_list = []
	for line in lines:
		if not line.startswith('#') and line != '':
			if line.lower().startswith('http'):
				ts_url_list.append(line)
			else:
				ts_url_list.append('%s/%s' % (host, line))
	return ts_url_list

def download_ts_file(ts_url_list, download_dir):
	ts_path_list = []
	i = 0
	for ts_url in reversed(ts_url_list):
		i += 1
		file_name = ts_url[ts_url.rfind('/'):]
		curr_path = '%s%s' % (download_dir, file_name)
		print('\n[process]: %s/%s' % (i, len(ts_url_list)))
		print('[download]:', ts_url)
		print('[target]:', curr_path)
		ts_path_list.append(curr_path)
		if os.path.isfile(curr_path):
			print('[warn]: file already exist')
			continue
		r = requests.get(ts_url)
		with open(curr_path, 'wb') as f:
			f.write(r.content)
	return ts_path_list

def check_dir(path):
	if os.path.exists(path):
		return
	os.makedirs(path)

def get_download_url_list(host, m3u8_url, url_list = []):
	body = get_m3u8_body(m3u8_url)
	ts_url_list = get_url_list(host, body)
	for url in ts_url_list:
		if url.lower().endswith('.m3u8'):
			url_list = get_download_url_list(host, url, url_list)
		else:
			url_list.append(url)
	return url_list


# def combine_ts_file(ts_path_list, target_path):
# 	with open(target_path, 'wb+') as f:
# 		for ts_path in ts_path_list:
# 			print(ts_path)
# 			f.write(open(ts_path, 'rb').read())

def download_ts(m3u8_url, save_dir):
	check_dir(save_dir)
	host = get_host(m3u8_url)
	ts_url_list = get_download_url_list(host, m3u8_url)
	print(ts_url_list)
	print('total file count:', len(ts_url_list))
	ts_path_list = download_ts_file(ts_url_list, save_dir)

def main():
	save_dir = '/Users/huzhenjie/Downloads/m3u8_sample_dir'
	m3u8_url = 'http://hls.cntv.lxdns.com/asp/hls/main/0303000a/3/default/978a64ddd3a1caa85ae70a23414e6540/main.m3u8'
	download_ts(m3u8_url, save_dir)


if __name__ == '__main__':
	# main()
	config = get_cfg()
	if config:
		download_ts(config[0], config[1])