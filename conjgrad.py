import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyval

# data fields
x = []
y = []
# The X matrix
xm = []


def main():
    parser = argparse.ArgumentParser(description='Using gradient descending method to solve the curve fitting problem.')
    parser.add_argument('--level', '-l', type=int, help='The level of the polynomial(required). Must be a positive integer.')
    parser.add_argument('--factor', '-f', type=float, default=0, help='The factor of the regular item. Default is 0.')
    parser.add_argument('--show-sine', action='store_true', help='Show the real sine curve on the figure.')
    args = parser.parse_args()
    if not args.level or args.level < 0:
        parser.error('Expect level to be a positive integer')
    if args.factor < 0:
        parser.error('Expect factor to be greater than or equal to zero')
    with open('rawdata.csv', newline='') as csvfile:
        creader = csv.reader(csvfile)
        for row in creader:
            x.append(float(row[0]))
            y.append(float(row[1]))
    for xi in x:
        xm.append([xi ** l for l in range(args.level)])
    w, e, k = conjugate_gradient(args.level, args.factor)
    print('Optimized result: ' + str(w))
    print('Error rate: ' + str(e))
    print('Steps: ' + str(k))
    # Draw it
    px=np.linspace(min(x), max(x), num=100)
    py=polyval(px, w)
    plt.scatter(x, y, label='Original Data', color='k')
    plt.plot(px, py, label='Fitting result', color='r')
    if args.show_sine:
        sy=np.sin(px)
        plt.plot(px, sy, label='Real sine', color='g')
    # Set the appearance of the figure and show it
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Fitting results')
    plt.legend()
    plt.show()


def err_rate(w, fac):
    diff = np.dot(xm, w) - y
    return (np.dot(diff.T, diff) + fac * np.dot(w, w)) / 2


def grad_func(w, fac):
    diff = np.dot(xm, w) - y
    return np.dot(np.transpose(xm), diff) + fac * np.array(w)


def conjugate_gradient(level, fac):
    w = np.zeros((level,))
    k = 0
    a = np.dot(np.transpose(xm), xm) + fac * np.eye(level)
    ri = np.dot(np.transpose(xm), y) - np.dot(a, w)  # ri stands for (negative) gradient
    di = ri  # di stands for conjugate bases
    n = 0
    while np.linalg.norm(ri) > 1e-6:
        alpha = np.dot(ri, ri)/np.dot(np.dot(di, a), di)
        rj = ri  # rj is the previous gradient(ri)
        w += alpha * di  # calculate w(i+1)
        # calculate r(i+1)
        ri -= alpha * np.dot(a, di)
        # calculate d(i+1). Reset each l times.
        n += 1
        n %= level
        if n==1:
            di = np.dot(np.transpose(xm), y) - np.dot(a, w)
        else:
            beta = np.dot(ri, ri)/np.dot(rj, rj)
            di = beta * di + ri
        k += 1
    return w, err_rate(w,fac), k


if __name__ == '__main__':
    main()
