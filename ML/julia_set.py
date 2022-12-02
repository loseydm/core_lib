import matplotlib.pyplot as plt
import itertools as it
import numpy as np
import time

from math import sqrt

def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print('{:.3f} Seconds'.format(time.time() - start))

        return result

    return wrapper

@time_it
def py_julia_set(desired_width, max_iterations, xs, ys, cs):
    """Calculates the complex coordinates/parameters of a Julia Set"""

    x_one, x_two = xs
    y_one, y_two = ys

    c_real, c_imag = cs

    xs, ys = list(), list()

    x_step = (x_two - x_one) / desired_width
    y_step = (y_one - y_two) / desired_width

    y_coord = y_two
    while y_coord > y_one:
        ys.append(y_coord)
        y_coord += y_step

    x_coord = x_one
    while x_coord < x_two:
        xs.append(x_coord)
        x_coord += x_step

    zs = list(map(complex, it.product(xs, ys)))

    return zs

@time_it
def np_julia_set(desired_width, max_iterations, xs, ys, cs):
    """Calculates the complex coordinates/parameters of a Julia Set"""

    x_one, x_two = xs
    y_one, y_two = ys

    c_real, c_imag = cs

    x_step = (x_two - x_one) / desired_width
    y_step = (y_one - y_two) / desired_width

    xs = np.full(desired_width, x_step)
    xs[0] = x_one
    xs = xs.cumsum()

    ys = np.full(desired_width, y_step)
    ys[0] = y_two
    ys = ys.cumsum()

    c = complex(c_real, c_imag)

    counter = 0
    output = np.zeros(len(xs) * len(ys))
    for x in xs:
        for y in ys:
            n = 0
            z = complex(x, y)

            while abs(z) < 2 and n < max_iterations:
                z = z * z + c
                n += 1

            output[counter] = n
            counter += 1

    return output


if __name__ == '__main__':
    half_half = True if input('Half-and-Half (y or n): ').lower() == 'y' else False
    
    xs, ys, cs = (-1.8, 1.8), (-1.8, 1.8), (-.62772, -.42193)

    output = np_julia_set(1000, 255, xs, ys, cs)
    output = output.reshape((1000, 1000))

    numbers = np.zeros((1000, 1000, 3), dtype = 'uint8')

    TOP_OFFSET = None
    for i, row in enumerate(output):
        for j, cell in enumerate(row):
            if cell == 255 and TOP_OFFSET is None:
                TOP_OFFSET = 255 / (1000 - i)

    for i, row in enumerate(output):
        for j, cell in enumerate(row):
            if cell == 255:
                BOT = i

     # Offset of values per pixel difference
    BOT_OFFSET = 255 / i
    
    OFFSET = 255 / 1000
    
    top_x, top_y = 500, 0
    bot_x, bot_y = 500, 1000
    lef_x, lef_y = 0, 500
    rig_x, rig_y = 1000, 500
    
    for i in range(1000):
        for j in range(1000):
            if output[i][j] <= 255 and output[i][j] >= 230:

                if half_half:
                    T = int(TOP_OFFSET * sqrt((top_x - j) ** 2 + (top_y - i) ** 2))
                    B = int(BOT_OFFSET * sqrt((bot_x - j) ** 2 + (bot_y - i) ** 2))

                    
                    if T > B: # Half Red and Half Blue
                        numbers[i][j] = [0, 0, T]
                    else:
                        numbers[i][j] = [B, 0, 0]

                
                else:
                    T = int(OFFSET * sqrt((top_x - j) ** 2 + (top_y - i) ** 2))
                    B = int(OFFSET * sqrt((bot_x - j) ** 2 + (bot_y - i) ** 2))
                    L = int(OFFSET * sqrt((lef_x - j) ** 2 + (lef_y - i) ** 2))
                    R = int(OFFSET * sqrt((rig_x - j) ** 2 + (rig_y - i) ** 2))

                    numbers[i][j] = [T, 0, B] # Move the B L T R letters around in this to change the appearance
                
                

    plt.axis('off')
    plt.imshow(numbers)
    plt.show()
