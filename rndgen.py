import argparse
import csv
import random
from math import sin, pi


def main():
    parser = argparse.ArgumentParser(description='Generate random X-Y pairs for curve fitting.')
    parser.add_argument('-n', type=int, default=10, help='The number of random pairs. Default is 10.')
    parser.add_argument('-b', type=float, default=pi / 2, help='The bound of random number. Default is pi/2.')
    args = parser.parse_args()
    if args.n <= 0:
        parser.error('Expect n to be a positive integer')
    # generate random number pairs
    x = [random.uniform(-args.b, args.b) for i in range(args.n)]
    y = [sin(i) + random.gauss(0, 0.02) for i in x]
    # store the data to a file
    with open('rawdata.csv', 'w', newline='') as csvfile:
        cwrite = csv.writer(csvfile)
        for xi, yi in zip(x, y):
            cwrite.writerow([str(xi), str(yi)])


if __name__ == '__main__':
    main()
