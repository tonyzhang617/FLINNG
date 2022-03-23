#ifndef _LSH_FUNCTIONS
#define _LSH_FUNCTIONS

#include <climits>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

uint64_t combine(uint64_t item1, uint64_t item2);

void single_densified_minhash(uint64_t *result, const uint64_t *point,
                              uint64_t point_len, uint64_t num_tables,
                              uint64_t hashes_per_table, uint8_t hash_range_pow,
                              uint32_t random_seed);

std::vector<uint64_t>
parallel_densified_minhash(const uint64_t *points, uint64_t num_points,
                           uint64_t point_dimension, uint64_t num_tables,
                           uint64_t hashes_per_table, uint8_t hash_range_pow,
                           uint32_t random_seed);

std::vector<uint64_t>
parallel_densified_minhash(const std::vector<std::vector<uint64_t>> &points,
                           uint64_t num_tables, uint64_t hashes_per_table,
                           uint8_t hash_range_pow, uint32_t random_seed);

std::vector<uint64_t> parallel_srp(const float *dense_data, uint64_t num_points,
                                   uint64_t data_dimension, int8_t *random_bits,
                                   uint64_t num_tables,
                                   uint64_t hashes_per_table);

std::vector<uint64_t> parallel_l2_lsh(const float *dense_data, uint64_t num_points,
                                      uint64_t data_dimension, int8_t *random_bits,
                                      uint64_t num_tables,
                                      uint64_t hashes_per_table,
                                      uint64_t sub_hash_bits = 2,
                                      uint64_t cutoff = 6);
#endif