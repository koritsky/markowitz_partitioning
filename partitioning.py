from markowitz import Portfolio
import numpy as np
import itertools
from binary_problem import BinaryProblem


class Partitioning(BinaryProblem):
    def __init__(self, mixed_mat=None, theta=None):
        if mixed_mat is None:
            mixed_mat = []
        if theta is None:
            theta = 10
        super().__init__(bqm=None, qubo_mat=None)
        self.mixed_mat = mixed_mat
        self.ordered_mat = self.mixed_mat
        self.permutation_mat = np.eye(self.mixed_mat.shape[0])
        self.permutation_dim = self.permutation_mat.shape[0]
        self.theta = theta

        self.to_partitioning_qubo()

    def to_partitioning_qubo(self, mixed_mat=None, theta=None):
        if mixed_mat is None:
            mixed_mat = self.mixed_mat
        if theta is None:
            theta = self.theta

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

    def permutation_check(self, mat=None):
        if mat is None:
            mat = self.permutation_mat
        mat = np.array(mat)
        # Check, whether given matrix is permutation one

        return np.array_equal(np.dot(mat, mat.T), np.eye(mat.shape[0]))

    def to_permutation(self, permutation_mat=None):
        """ We assume that after solving bqm solution might not be permutation matrix.
        So this function fixes found correct rows/columns and tries to fix wrong ones using given algorithm.
        :param permutation_mat:
        :return:
        """
        if permutation_mat is None:
            permutation_mat = self.permutation_mat
        mat = np.array(permutation_mat)
        dim = mat.shape[0]

        row_holes = []
        col_holes = []

        # Finds wrong rows and columns and puts zeroes instead.
        for i in range(dim):
            if sum(mat[i]) != 1:
                row_holes.append(i)
                mat[i] = np.zeros(dim)
            if sum(mat.T[i]) != 1:
                col_holes.append(i)
                mat.T[i] = np.zeros(dim)

        # In some cases we are unable to fix problems
        if len(row_holes) != len(col_holes):
            print('\033[93m' + "Unable to fix this shit ¯\_(ツ)_/¯" + '\033[0m')
            return permutation_mat

        sub_dim = len(row_holes)
        print('\033[93m' + "Number of wrong lines:" + '\033[0m')
        print('\033[93m' + str(sub_dim) + '\033[0m')
        if sub_dim > 8:
            print('\033[93m' + "Too many mistakes" + '\033[0m')
            return permutation_mat

        solutions_energy = []

        # List all of sub_matrices that can fix the holes
        permutations = [np.array(p) for p in itertools.permutations(np.eye(sub_dim))]

        # Make iterator from holes coordinates
        holes = list(itertools.product(row_holes, col_holes))

        # Make iterator for our sub_matrices
        simple_iters = list(itertools.product(range(sub_dim), repeat=2))

        # Map them to each other to put element of found sub_matrix into correct holes
        mega_iters = list(zip(holes, simple_iters))

        # Run over all possible sub_matrices
        for permutation in permutations:
            # Create new matrix
            sub_mat = np.array(permutation)
            hole_mat = np.zeros((dim, dim))
            for iter in mega_iters:
                hole_mat[iter[0]] = sub_mat[iter[1]]
            new_mat = mat + hole_mat
            solutions_energy.append(self.bqm.energy(new_mat.reshape(dim ** 2)))

        # Find which sub_matrix fits best
        permutation_index = solutions_energy.index(min(solutions_energy))

        # Return this sub_matrix in correct holes
        sub_mat = permutations[permutation_index]
        hole_mat = np.zeros((dim, dim))
        for iter in mega_iters:
            hole_mat[iter[0]] = sub_mat[iter[1]]
        new_mat = mat + hole_mat
        return new_mat

    @staticmethod
    def mixed_matrix_generator(block_dim=None, block_mat=None):
        # Generate random symmetric block matrix
        if block_mat is None:
            block_mat = Partitioning.rand_sym_block_gen(block_dim)
            g_dim = sum(block_dim)
        else:
            block_mat = block_mat
            g_dim = block_mat.shape[0]

        print("Given random block matrix:")
        print(block_mat)
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
        mixed_mat = np.dot(p_mat.T, np.dot(block_mat, p_mat))
        print("Mixed matrix")
        print(mixed_mat)
        print("\n")
        return block_mat, mixed_mat

    def list_to_mat(self, solution=None):
        if solution is None:
            solution = self.current_solution[1]
        solution = np.array(solution)
        self.permutation_mat = solution.reshape(self.permutation_dim, self.permutation_dim)
        return self.permutation_mat

    def permute(self, permutation_mat=None, mixed_mat=None):
        if permutation_mat is None:
            permutation_mat = self.permutation_mat
        if mixed_mat is None:
            mixed_mat = self.mixed_mat
        ordered_mat = np.dot(permutation_mat.T, np.dot(mixed_mat, permutation_mat))
        return ordered_mat

    @staticmethod
    def rand_sym_block_gen(block_dim: list, ordered=False):
        # Check for valid block_dim list
        for i in block_dim:
            assert i > 0, "Nonpositive block dimension"
        general_dim = sum(block_dim)  # Dimension of resulting mat
        mat = np.zeros((general_dim, general_dim))
        current_dim = 0
        for dim in block_dim:
            current_dim = current_dim + dim
            block = 2 * np.random.randn(dim, dim) - 1
            if ordered:
                # Make each block such that it doesn't has to be ordered
                block_problem = Partitioning(mixed_mat=block)
                block_problem.to_partitioning_qubo()
                block_problem.exact_solver()
                block_problem.list_to_mat()
                block = block_problem.permute()
            mat += np.pad(block,
                          ((current_dim - dim, general_dim - current_dim),
                           (current_dim - dim, general_dim - current_dim)),
                          "constant",
                          constant_values=(0, 0))
        # Make it symmetric
        mat = (1 / 2) * (mat + mat.T)
        return mat

if __name__ == "__main__":

    np.random.seed(6)
    # Make console print look better
    np.set_printoptions(precision=3,  # Digits after point
                        linewidth=170,  # Length of the line
                        suppress=True)  # Always fixed point notation

    block_dim = [3, 1]
    size = sum(block_dim)
    block_mat = Partitioning.rand_sym_block_gen(block_dim, ordered=True)
    print(block_mat)
    # print(block_mat)
    # noise_mat = np.zeros((size, size))
    # _, mixed_mat = Partitioning.mixed_matrix_generator(block_mat=block_mat)
    # mixed_mat = mixed_mat + noise_mat
    # part = Partitioning(mixed_mat)
    # _, solution = part.exact_solver()
    # part.permutation_mat = part.list_to_mat(solution)
    # permuted_mixed_mat = part.permute(part.permutation_mat, mixed_mat)
    # permuted_noise_mat = part.permute(part.permutation_mat, noise_mat)
    # new_block_mat = permuted_mixed_mat - permuted_noise_mat
    # print(new_block_mat)
