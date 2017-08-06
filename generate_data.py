import random
import numpy as np

import main, config

data_len = config.num_steps


# total_size = 3


def genernate_random_data1(total_size):
    def f():
        a, b, c = [], [], []
        for i in xrange(data_len - 1):
            ai = random.randint(0, 1)
            if ai == 0:
                bi = random.randint(0, 1)
            else:
                bi = 0
            #bi = random.randint(0, 1)
            ci = ai ^ bi
            a.append(ai)
            b.append(bi)
            c.append(ci)
        a.append(0)
        b.append(0)
        c.append(0)
        return a, b, c

    A, C = [], []
    A.append(np.column_stack(([1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1])))
    C.append([0, 0, 0, 0, 0, 0])
    for i in xrange(total_size - 1):
        a, b, c = f()
        A.append(np.column_stack((a, b)))
        C.append(c)
    input = np.array(A)
    target = np.array(C)
    return input, target


def genernate_random_data2(total_size):
    def f2():
        a, b, c = [1, ], [1, ], [0, ]
        carry = 0
        for i in xrange(data_len - 2):
            ai = random.randint(0, 1)
            if ai == 0:
                bi = random.randint(0, 1)
            else:
                bi = 0
            ci = ai + bi + carry
            carry = ci // 2
            ci = ci % 2
            a.append(ai)
            b.append(bi)
            c.append(ci)
        a.append(0)
        b.append(0)
        c.append(carry)
        return a, b, c

    A, C = [], []
    for i in xrange(total_size):
        a, b, c = f2()
        A.append(np.column_stack((a, b)))
        C.append(c)
    input = np.array(A)
    target = np.array(C)
    return input, target


def genernate_random_data3(total_size):
    def f3():
        a, b, c = [], [], []
        carry = 0
        for i in xrange(data_len - 1):
            ai = random.randint(0, 1)
            bi = random.randint(0, 1)
            ci = ai + bi + carry
            carry = ci // 2
            ci = ci % 2
            a.append(ai)
            b.append(bi)
            c.append(ci)
        a.append(0)
        b.append(0)
        c.append(carry)
        return a, b, c

    A, C = [], []
    for i in xrange(total_size):
        a, b, c = f3()
        A.append(np.column_stack((a, b)))
        C.append(c)
    input = np.array(A)
    target = np.array(C)
    return input, target


