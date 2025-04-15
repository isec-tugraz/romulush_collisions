#cython: language_level=3, annotation_typing=True, embedsignature=True, boundscheck=False, wraparound=False, cdivision=True
#distutils: language = c++

cimport cython
from libc.stdlib cimport rand, srand
from libc.time cimport clock_t, clock, CLOCKS_PER_SEC
from libc.math cimport log2, pow
from libc.string cimport memset, memcpy
from libc.stdio cimport printf
from cython.view cimport array as cvarray
#cimport numpy as np

import numpy as np
import sys
import tqdm

from libcpp.vector cimport vector
from libcpp.algorithm cimport sort
from skinny cimport *
from util import cache

cdef extern from *:
    """
    #define ffs(x) __builtin_ffsll(x)
    #define popcount(x) __builtin_popcountll(x)
    """
    int ffs(int x) noexcept nogil
    int popcount(unsigned int x) noexcept nogil
    int likely(int) noexcept nogil
    int unlikely(int) noexcept nogil

ctypedef struct Difference:
    uint64_t mask
    float log_prob


cpdef int getbit(const uint16_t[:, ::1] mat, size_t row, size_t col) noexcept nogil:
    return (mat[row][col // 16] >> (col % 16)) & 1

cpdef int is_identity_submatrix(const uint16_t[:, ::1] mat, size_t limit) noexcept nogil:
    cdef size_t row, col
    for row in range(limit):
        for col in range(limit):
            if getbit(mat, row, col) != (row == col):
                return 0
    return 1

cpdef size_t largest_identity_submatrix(const uint16_t[:,::1] mat) noexcept nogil:
    cdef size_t res = 0

    cdef size_t rows = mat.shape[0]
    cdef size_t cols = mat.shape[1] * 16
    cdef size_t l = rows if rows < cols else cols
    cdef size_t i

    for i in range(l):
        if not is_identity_submatrix(mat, i):
            return i - 1
    return l


cdef vector[uint16_t] non_zero(const int16_t *ddt) noexcept nogil:
    cdef vector[uint16_t] result
    cdef size_t i

    result.resize(1 << 16)

    cdef int res_size = 0
    for i in range(1 << 16):
        if ddt[i] != 0:
            result[res_size] = i
            res_size += 1

    result.resize(res_size)
    return result

cdef void xor_assign(uint16_t *dst, const uint16_t *src, size_t len) noexcept nogil:
    for i in range(len):
        dst[i] ^= src[i]

cdef void mul_acc(uint16_t *dst, const uint16_t[:, ::1] mat, uint16_t val) noexcept nogil:
    cdef size_t bit_idx
    for bit_idx in range(16):
        if val & (1 << bit_idx):
            xor_assign(dst, &mat[bit_idx, 0], mat.shape[1])


cdef uint64_t _count_good_masks(const uint16_t[:, ::1] mat, const int16_t *ddt) noexcept nogil:
    cdef uint64_t result = 0
    cdef vector[uint16_t] nz_ddt = non_zero(ddt)
    cdef size_t free_sbox, bit_idx
    cdef int64_t first_set_bit

    cdef uint64_t i, gray, gray_last, prob_tmp

    cdef uint16_t sbox_val

    cdef size_t num_sboxes = mat.shape[1]
    cdef size_t free_sboxes = largest_identity_submatrix(mat) / 16
    cdef size_t remaining_bits = mat.shape[0] - 16 * free_sboxes
    printf("%zd/%zd S-boxes are free (%zd pivots), ", free_sboxes, mat.shape[1], largest_identity_submatrix(mat))
    printf("%zd remaining bits\n", remaining_bits)

    cdef vector[uint16_t] sboxes = vector[uint16_t](num_sboxes)

    cdef int64_t sbox_transition_idx, limit = 1
    for _ in range(free_sboxes):
        limit *= nz_ddt.size()

    for sbox_transition_idx in range(limit):
        for i in range(sboxes.size()):
            sboxes[i] = 0

        for free_sbox in range(free_sboxes):
            sbox_val = nz_ddt[sbox_transition_idx % nz_ddt.size()]
            sbox_transition_idx /= nz_ddt.size()
            mul_acc(sboxes.data(), mat[16 * free_sbox:16 * (free_sbox + 1)], sbox_val)

        prev_gray = 0
        for i in range((<uint64_t> 1) << remaining_bits):
            gray = i ^ (i >> 1)

            first_set_bit = ffs(gray ^ prev_gray)
            if likely(first_set_bit > 0):
                xor_assign(sboxes.data(), &mat[16 * free_sboxes + first_set_bit - 1, 0], num_sboxes)
            prev_gray = gray

            prob_tmp = 1
            for sbox in sboxes:
                prob_tmp *= ddt[sbox]
                if prob_tmp == 0:
                    break

            if unlikely(prob_tmp > 0):
                result += 1

    return result


cdef vector[Difference] _find_good_masks(const uint16_t[:, ::1] mat, const int16_t *ddt) noexcept nogil:
    cdef vector[Difference] result
    cdef vector[uint16_t] nz_ddt = non_zero(ddt)
    cdef size_t free_sbox, bit_idx
    cdef int64_t first_set_bit
    cdef Difference result_tmp

    cdef uint64_t i, gray, gray_last, prob_tmp

    cdef uint16_t sbox_val

    cdef size_t num_sboxes = mat.shape[1]
    cdef size_t free_sboxes = largest_identity_submatrix(mat) / 16
    cdef size_t remaining_bits = mat.shape[0] - 16 * free_sboxes
    printf("%zd/%zd S-boxes are free (%zd pivots), ", free_sboxes, mat.shape[1], largest_identity_submatrix(mat))
    printf("%zd remaining bits\n", remaining_bits)

    cdef vector[uint16_t] sboxes = vector[uint16_t](num_sboxes)

    cdef int64_t sbox_transition_idx, limit = 1
    for _ in range(free_sboxes):
        limit *= nz_ddt.size()

    cdef double progress = 0, last_progress = 0
    cdef double start = <double> clock() / CLOCKS_PER_SEC, now
    cdef double last_print = start

    with gil:
        pbar = tqdm.tqdm(total=limit, bar_format='{l_bar}{bar}[{elapsed}<{remaining}{postfix}]')
    for sbox_transition_idx in range(limit):
        now = <double> clock() / CLOCKS_PER_SEC
        if now - last_print > 1:
            last_print = now
            with gil:
                try:
                    pbar.set_description('{} tweakeys'.format(result.size()))
                    pbar.update(sbox_transition_idx - pbar.n)
                except KeyboardInterrupt:
                    break


        for i in range(sboxes.size()):
            sboxes[i] = 0

        for free_sbox in range(free_sboxes):
            sbox_val = nz_ddt[sbox_transition_idx % nz_ddt.size()]
            sbox_transition_idx /= nz_ddt.size()
            mul_acc(sboxes.data(), mat[16 * free_sbox:16 * (free_sbox + 1)], sbox_val)

        prev_gray = 0
        for i in range((<uint64_t> 1) << remaining_bits):
            gray = i ^ (i >> 1)

            first_set_bit = ffs(gray ^ prev_gray)
            if likely(first_set_bit > 0):
                xor_assign(sboxes.data(), &mat[16 * free_sboxes + first_set_bit - 1, 0], num_sboxes)
            prev_gray = gray

            prob_tmp = 1
            for sbox in sboxes:
                prob_tmp *= ddt[sbox]
                if prob_tmp == 0:
                    break

            if unlikely(prob_tmp > 0):
                result_tmp.log_prob = log2(prob_tmp) - num_sboxes * 8
                result_tmp.mask = gray << (free_sboxes * 16)

                for free_sbox in range(free_sboxes):
                    result_tmp.mask |= (<uint64_t> sboxes[free_sbox]) << (free_sbox * 16)

                result.push_back(result_tmp)

    with gil:
        pbar.set_description('{} tweakeys'.format(result.size()))
        pbar.update(limit - pbar.n)
        pbar.close()

    return result

def count_good_masks(const uint16_t[:, ::1] mat, const int16_t[:, ::1] ddt):
    cdef uint64_t[:] result_mask_view = None
    cdef float[:] result_prob_view = None
    cdef size_t i
    cdef const int16_t *ddt_ptr = &ddt[0, 0]
    cdef uint64_t num_good_masks

    num_good_masks = _count_good_masks(mat, ddt_ptr)
    return num_good_masks

@cache.cache_function
def find_good_masks(const uint16_t[:, ::1] mat, const int16_t[:, ::1] ddt):
    """
    Search for row-wise linear combinations of mat such that ddt.flatten(el) != 0
    for all 16-bit el in the linear combination.

    Each 16-bit element corresponds to one S-box transitions where the upper 8
    bits define the input difference and the lower 8 bits the output differnce,
    i.e., el = input_differnce << 8 | output_difference.a

    returns a structured np array where ['mask'] specifies which rows of mat
    shall be selected to get a probability of ['log_prob'].
    """
    cdef vector[Difference] vec
    cdef uint64_t[:] result_mask_view = None
    cdef float[:] result_prob_view = None
    cdef size_t i
    cdef const int16_t *ddt_ptr = &ddt[0, 0]

    vec = _find_good_masks(mat, ddt_ptr)

    diff_dtype = [('mask', np.uint64), ('log_prob', np.float32)]
    result = np.empty(vec.size(), dtype=diff_dtype)

    result_mask_view = result['mask']
    result_prob_view = result['log_prob']

    for i in range(vec.size()):
        result_mask_view[i] = vec[i].mask
        result_prob_view[i] = vec[i].log_prob

    result.sort(order='log_prob')
    return result[::-1]
