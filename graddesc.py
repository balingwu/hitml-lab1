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
    parser.add_argument('--alpha', '-a', type=float, default=0.1, help='Learning rate(the step length). Default is 0.1')
    parser.add_argument('--show-sine', action='store_true', help='Show the real sine curve on the figure.')
    args = parser.parse_args()
    if not args.level or args.level < 0:
        parser.error('Expect level to be a positive integer')
    if args.factor < 0:
        parser.error('Expect factor to be greater than or equal to zero')
    if args.alpha <= 0:
        parser.error('Expect alpha to be positive')
    with open('rawdata.csv', newline='') as csvfile:
        creader = csv.reader(csvfile)
        for row in creader:
            x.append(float(row[0]))
            y.append(float(row[1]))
    for xi in x:
        xm.append([xi ** l for l in range(args.level)])
    w, e, k = grad_desc(args.level, args.factor, args.alpha)
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


def grad_desc(level, fac, alpha):
    w = np.ones(level)
    grad = grad_func(w, fac)
    val = err_rate(w, fac)
    k = 0
    while np.linalg.norm(grad) > 1e-6:
        w -= alpha * grad
        # Make alpha smaller if the error rate goes up
        if err_rate(w, fac) > val:
            alpha *= 0.5
        grad = grad_func(w, fac)
        val = err_rate(w, fac)
        k += 1
    return w, val, k


if __name__ == '__main__':
    main()
