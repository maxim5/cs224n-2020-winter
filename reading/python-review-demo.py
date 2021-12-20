import numpy as np

def power_iteration(A, tolerance=1e-7):
    b_old = np.random.rand(A.shape[1])
    b = np.random.rand(A.shape[1])
    num_iterations = 0
    while num_iterations == 0 or np.linalg.norm(b_old - b) > tolerance:
        b_old = np.copy(b)
        b = np.dot(A, b)
        b_norm = np.linalg.norm(b)
        b /= b_norm
        num_iterations += 1
    return np.dot(A, b), b, num_iterations

def main():
    A = np.array([[.5, .4], [.2, .8]])
    ab, b, number_iterations = power_iteration(A)

    eig1 = ab[0] / b[0]
    eig2 = ab[1] / b[1]
    assert(np.abs((eig1 - eig2) / eig2) < 1e-5)

    b /= b[1]

    print(eig1, b, number_iterations)

if __name__ == '__main__':
    main()
