class ising_utilits:
    @staticmethod
    def to_ising_file(h, J, filename):
        """ Create an ising .txt file, that contains linear and quadratic coeffients as well as other (?) information
        :param h: list of linear coefficients
        :param J: matrix of quadratic coefficients
        :param filename: name of file
        """
        n = len(h)
        f = open(filename, 'w')
        f.write("%f\n" % 3.0)
        f.write("%f\n" % 0.1)
        f.write("%f\n" % -0.9)
        f.write("%f\n" % 0.07)
        f.write("%d\n" % n)
        for i in range(n):
            if i in h.keys():
                f.write("%f\n" % h[i])
            else:
                f.write("%f\n" % 0)
        for row in range(n):
            for col in range(row + 1, n):
                if (row, col) in J.keys():
                    f.write("%d %d %f\n" % (row + 1, col + 1, J[row, col]))
                else:
                    f.write("%d %d %f\n" % (row + 1, col + 1, 0))

    @staticmethod
    def ising_to_matrix(h, J):
        """ Transfrom dictionaries into vector and matrix for ising task.
        :param h: dict of linear coefficients
        :param J: dict of quadratic coefficients
        :return: tuple (hvector, Jmatrix) - vector of linear and matrix of quadratic coefficients
        """
        n = len(h)

        hvector = np.zeros((n, 1))
        for i in range(n):
            if i in h.keys(): hvector[i] = h[i]

        Jmatrix = np.zeros((n, n))
        for row in range(n):
            for col in range(row + 1, n):
                if (row, col) in J.keys(): Jmatrix[row, col] = J[(row, col)]
        return hvector, Jmatrix