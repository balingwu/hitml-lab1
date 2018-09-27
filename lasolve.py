import argparse
import csv
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyval

# data fields
x = []
y = []
# The X matrix
xm = []


def analsol():
    xmat = np.array(xm)
    A = np.matmul(xmat.T, xmat)
    b = np.dot(xmat.T, y)
    w = la.lstsq(A, b, rcond=None)[0]
    e = la.norm(np.dot(xmat, w) - y) ** 2 / 2
    return w, e


def analsol_re(level, fac):
    xmat = np.array(xm)
    A = np.matmul(xmat.T, xmat)
    A += fac * np.eye(level)
    b = np.dot(xmat.T, y)
    w = la.lstsq(A, b, rcond=None)[0]
    e = (la.norm(np.dot(xmat, w) - y) ** 2 + fac * la.norm(w) ** 2) / 2
    return w, e


def main():
    parser = argparse.ArgumentParser(description='Using analytical method to solve the curve fitting problem.')
    parser.add_argument('--level', '-l', type=int, help='The level of the polynominal.')
    parser.add_argument('--factor', '-f', type=float, default=0, help='The factor of the regular item. Default is 0.')
    parser.add_argument('--show-sine', action='store_true', help='Show the real sine curve on the figure.')
    args = parser.parse_args()
    if args.level is None:
        parser.error('Level not specified')
    if args.level <= 0:
        parser.error('Expect level to be a positive integer')
    if args.factor < 0:
        parser.error('Expect factor to be greater than or equal to zero')
    # Read data and then compute the polynomial
    with open('rawdata.csv', newline='') as csvfile:
        creader = csv.reader(csvfile)
        for row in creader:
            x.append(float(row[0]))
            y.append(float(row[1]))
    for xi in x:
        xm.append([xi ** l for l in range(args.level)])
    print('Without regular item:')
    w1, e1 = analsol()
    print('Fitting result: ' + str(w1))
    print('Error: ' + str(e1))
    print('With regular item:')
    w2, e2 = analsol_re(args.level, args.factor)
    print('Fitting result: ' + str(w2))
    print('Error: ' + str(e2))
    # Draw it
    px=np.linspace(min(x), max(x), num=100)
    py1=polyval(px, w1)
    py2=polyval(px, w2)
    plt.scatter(x, y, label='Original Data', color='k')
    plt.plot(px, py1, label='No reg. item', color='r')
    plt.plot(px, py2, label='With reg. item', color='b')
    if args.show_sine:
        sy=np.sin(px)
        plt.plot(px, sy, label='Real sine', color='g')
    # Set the appearance of the figure and show it
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Fitting results')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
