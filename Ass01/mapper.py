#!/usr/bin/python3
import re
import sys

def mapper():
	pat = re.compile('(?P<ip>\d+.\d+.\d+.\d+).*?\d{4}:(?P<hour>\d{2}):\d{2}.*? ')
	for line in sys.stdin:
		match = pat.search(line)
		if match:
			print ('%s\t%s' % ( '[' + match.group('hour')+ ':00' + '] ' + match.group('ip'), 1))

if __name__ == "__main__":
	mapper()