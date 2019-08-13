from markowitz import Portfolio
import numpy as np
import itertools
import dimod
from binary_problem import BinaryProblem


class Partitioning(BinaryProblem):
    def __init__(self, block_dim=None):
        if block_dim is None:
            mixed_mat = []
        else:
            mixed_mat = self.mixed_matrix_generator(block_dim)
        super().__init__(bqm=None, qubo_mat=None)
        self.mixed_mat = mixed_mat
        self.ordered_mat = self.mixed_mat
        self.permutation_mat = np.eye(self.mixed_mat.shape[0])
        self.permutation_dim = self.permutation_mat.shape[0]

        self.to_partitioning_qubo()

    def to_partitioning_qubo(self, mixed_mat=None, theta=10):
        if mixed_mat is None:
            mixed_mat = self.mixed_mat

        # Dimension of primary task
        n = mixed_mat.shape[0]

        # Tensor that goes into objective function, 4 variables
        measure_tensor = np.zeros((n, n, n, n))

        # Matrix of qubo form
        qubo_mat = np.zeros((n ** 2, n ** 2))

        # Reshape tensor into qubo mat
        for it in itertools.product([i for i in range(n)], repeat=4):
            i, j, k, l = it
            measure_tensor[i, j, k, l] = abs(mixed_mat[i, j]) * ((k - l) ** 2)
            qubo_mat[i * n + k][j * n + l] = measure_tensor[i, j, k, l]

        # Add constraints
        qubo_mat = qubo_mat + theta * ((-4) * np.eye(n ** 2)
                                       + np.kron(np.eye(n), np.ones((n, n)))
                                       + np.kron(np.ones((n, n)), np.eye(n)))
        # Make it upper-diagonal
        qubo_mat = qubo_mat + np.triu(qubo_mat, 1) - np.tril(qubo_mat, -1)
        self.qubo_mat = qubo_mat
        self.from_qubo(self.qubo_mat)
        return self.qubo_mat

    def split_metric(self, mat=None, split=None):
        if mat is None:
            mat = self.ordered_mat
        if split is None:
            split = 60  # Because D-wave can solve up to ~64 node complete graph

        # By given split (number of column after which to cut mat) calculates metric -
        # average value of non diagonal block elements.

        dim = mat.shape[0]
        m = split + 1
        assert dim > m, "Split size more or equal to mat size"
        metric = 0
        for row in range(m, dim):
            for col in range(m):
                metric += abs(mat[row, col])
        return metric / ((dim - m) * m)

    def best_split(self, mat=None):
        if mat is None:
            mat = self.ordered_mat
        # Calculates best split, that minimizes split_metric

        dim = mat.shape[0]
        metric, split = 10 ** 3, 0
        for spl in range(dim - 1):
            metr = self.split_metric(mat, spl)
            if metr < metric:
                metric, split = metr, spl
        return split

    def split_until_threshold(self, mat=None, split_threshold=None):
        # Split mat iteratively by 2 until each block has dimension less then threshold.
        if mat is None:
            mat = self.ordered_mat
        if split_threshold is None:
            split_threshold = 60  # Because D-wave can solve up to ~64 node complete graph

        mat_to_split = [mat]
        good_mat = []
        while len(mat_to_split) != 0:
            temp_mat_to_split = []
            for i in range(len(mat_to_split)):
                mat = mat_to_split.pop(i)
                split = self.best_split(mat) + 1
                up_mat = mat[:split, :split]
                good_mat.append(up_mat) if split < split_threshold else temp_mat_to_split.append(up_mat)
                lo_mat = mat[split:, split:]
                good_mat.append(lo_mat) if (mat.shape[0] - split) < split_threshold else temp_mat_to_split.append(
                    lo_mat)
            mat_to_split.extend(temp_mat_to_split)
        return good_mat

    def permutation_check(self, mat=None):
        if mat is None:
            mat = self.permutation_mat
        mat = np.array(mat)
        # Check, whether given matrix is permutation one

        return np.array_equal(np.dot(mat, mat.T), np.eye(mat.shape[0]))

    def to_permutation(self, permutation_mat = None):
        if permutation_mat is None:
            permutation_mat = self.permutation_mat
        mat = np.array(permutation_mat)
        dim = mat.shape[0]

        row_holes = []
        col_holes = []

        for i in range(dim):
            if sum(mat[i]) != 1:
                row_holes.append(i)
                mat[i] = np.zeros(dim)
            if sum(mat.T[i]) != 1:
                col_holes.append(i)
                mat.T[i] = np.zeros(dim)
        if len(row_holes) != len(col_holes):
            print('\033[93m' + "Unable to fix this shit ¯\_(ツ)_/¯" + '\033[0m')

        sub_dim = len(row_holes)
        print('\033[93m' + "Number of wrong lines:" + '\033[0m')
        print('\033[93m' + str(sub_dim) + '\033[0m')
        assert sub_dim < 9, "Too many mistakes to fix it by bruteforce"
        solutions_energy = []

        permutations = [np.array(p) for p in itertools.permutations(np.eye(sub_dim))]
        holes = list(itertools.product(row_holes, col_holes))
        simple_iters = list(itertools.product(range(sub_dim), repeat=2))
        mega_iters = list(zip(holes, simple_iters))
        for permutation in permutations:
            sub_mat = np.array(permutation)
            hole_mat = np.zeros((dim, dim))
            for iter in mega_iters:
                hole_mat[iter[0]] = sub_mat[iter[1]]
            new_mat = mat + hole_mat
            solutions_energy.append(self.bqm.energy(new_mat.reshape(dim ** 2)))

        permutation_index = solutions_energy.index(min(solutions_energy))

        sub_mat = permutations[permutation_index]
        hole_mat = np.zeros((dim, dim))
        for iter in mega_iters:
            hole_mat[iter[0]] = sub_mat[iter[1]]
        new_mat = mat + hole_mat
        return new_mat

    def mixed_matrix_generator(self, block_dim: list):
        # Generate random symmetric block matrix

        block_dim = block_dim
        mat = Partitioning.rand_sym_block_gen(block_dim)
        g_dim = sum(block_dim)
        print("Given random block matrix:")
        print(mat)
        print("\n")

        # We define here (according to the letter) permutation matrix p_matrix as one
        # that have 1 on (i, j) if Permutation(i) = j.
        # So for column permutation of matrix A we should multiply  A*P. For rows (P.T)*A.
        # Hence (P.T)*A*P is matrix A with permuted columns and rows.

        # Generate random permutation matrix
        p_mat = np.random.permutation(np.eye(g_dim))
        print("Random permutation matrix:")
        print(p_mat)
        print("\n")

        # Mix generated matrix via permutation.
        mixed_mat = np.dot(p_mat.T, np.dot(mat, p_mat))
        print("Mixed matrix")
        print(mixed_mat)
        print("\n")
        return mixed_mat

    def list_to_mat(self, solution=None):
        if solution is None:
            solution = self.current_solution[1]
        solution = np.array(solution)
        return solution.reshape(self.permutation_dim, self.permutation_dim)


    @staticmethod
    def rand_sym_block_gen(block_dim: list):
        # Check for valid block_dim list
        for i in block_dim:
            assert i > 0, "Nonpositive block dimension"

        general_dim = sum(block_dim)  # Dimension of resulting mat
        mat = np.zeros((general_dim, general_dim))

        current_dim = 0
        for dim in block_dim:
            current_dim = current_dim + dim
            block = np.random.rand(dim, dim)
            mat += mat + np.pad(block,
                                ((current_dim - dim, general_dim - current_dim),
                                 (current_dim - dim, general_dim - current_dim)),
                                "constant",
                                constant_values=(0, 0))
        # Make it symmetric
        mat = (1 / 2) * (mat + mat.T)
        return mat
