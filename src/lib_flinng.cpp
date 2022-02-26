#include "lib_flinng.h"

BaseDenseFlinng32::BaseDenseFlinng32(uint64_t num_rows, uint64_t cells_per_row, uint64_t data_dimension,
                                     uint64_t num_hash_tables,
                                     uint64_t hashes_per_table, uint64_t hash_range) : internal_flinng(num_rows,
                                                                                                       cells_per_row,
                                                                                                       num_hash_tables,
                                                                                                       hash_range),
                                                                                       num_hash_tables(num_hash_tables),
                                                                                       hashes_per_table(
                                                                                           hashes_per_table),
                                                                                       data_dimension(data_dimension),
                                                                                       rand_bits(num_hash_tables *
                                                                                                 hashes_per_table *
                                                                                                 data_dimension) {

  for (uint64_t i = 0; i < rand_bits.size(); i++) {
    rand_bits[i] = (rand() % 2) * 2 - 1; // 50% chance either 1 or -1
  }
}

void BaseDenseFlinng32::addPoints(const std::vector<float> &points) {
  if (points.size() < data_dimension || points.size() % data_dimension != 0) {
    throw std::invalid_argument("The rows (each point) must be of dimension " +
                                std::to_string(data_dimension) +
                                ", and there must be at least 1 row.");
  }
  uint64_t num_points = points.size() / data_dimension;
  std::vector<uint64_t> hashes = getHashes(points.data(), num_points);
  internal_flinng.addPoints(hashes);
}

void BaseDenseFlinng32::addPoints(float *points, uint64_t num_points) {
  std::vector<uint64_t> hashes = getHashes(points, num_points);
  internal_flinng.addPoints(hashes);
}

void BaseDenseFlinng32::prepareForQueries() { internal_flinng.prepareForQueries(); }

std::vector<uint64_t> BaseDenseFlinng32::query(const std::vector<float> &queries, uint32_t top_k) {
  if (queries.size() < data_dimension || queries.size() % data_dimension != 0) {
    throw std::invalid_argument("The rows (each point) must be of dimension " +
                                std::to_string(data_dimension) +
                                ", and there must be at least 1 row.");
  }
  uint64_t num_queries = queries.size() / data_dimension;
  std::vector<uint64_t> hashes = getHashes(queries.data(), num_queries);
  std::vector<uint64_t> results = internal_flinng.query(hashes, top_k);
  return results;
}

std::vector<uint64_t> BaseDenseFlinng32::query(float *queries, uint64_t num_queries, uint32_t top_k) {
  std::vector<uint64_t> hashes = getHashes(queries, num_queries);
  std::vector<uint64_t> results = internal_flinng.query(hashes, top_k);
  return results;
}

DenseFlinng32::DenseFlinng32(uint64_t num_rows, uint64_t cells_per_row,
                             uint64_t data_dimension, uint64_t num_hash_tables, uint64_t hashes_per_table)
    : BaseDenseFlinng32(num_rows, cells_per_row, data_dimension, num_hash_tables, hashes_per_table,
                        1 << hashes_per_table) {

}

L2DenseFlinng32::L2DenseFlinng32(uint64_t num_rows, uint64_t cells_per_row,
                                 uint64_t data_dimension, uint64_t num_hash_tables,
                                 uint64_t hashes_per_table, uint64_t sub_hash_bits, uint64_t cutoff)
    : BaseDenseFlinng32(num_rows, cells_per_row, data_dimension, num_hash_tables, hashes_per_table,
                        1 << (hashes_per_table * sub_hash_bits)),
      sub_hash_bits(sub_hash_bits), cutoff(cutoff) {}

SparseFlinng32::SparseFlinng32(uint64_t num_rows, uint64_t cells_per_row,
                               uint64_t num_hash_tables, uint64_t hashes_per_table,
                               uint64_t hash_range_pow)
    : internal_flinng(num_rows, cells_per_row, num_hash_tables,
                      1 << hash_range_pow),
      num_hash_tables(num_hash_tables), hashes_per_table(hashes_per_table),
      hash_range_pow(hash_range_pow), seed(rand()) {}

void SparseFlinng32::addPointsSameDim(const uint64_t *points, uint64_t num_points, uint64_t point_dimension) {
  std::vector<uint64_t> hashes = getHashes(points, num_points, point_dimension);
  internal_flinng.addPoints(hashes);
}

void
SparseFlinng32::addPointsSameDim(const std::vector<uint64_t> &points, uint64_t num_points, uint64_t point_dimension) {
  std::vector<uint64_t> hashes = getHashes(points.data(), num_points, point_dimension);
  internal_flinng.addPoints(hashes);
}

void SparseFlinng32::addPoints(const std::vector<std::vector<uint64_t>> &data) {
  std::vector<uint64_t> hashes = getHashes(data);
  internal_flinng.addPoints(hashes);
}

std::vector<uint64_t> SparseFlinng32::hashPoints(const std::vector<std::vector<uint64_t>> &data) {
  return getHashes(data);
}

void SparseFlinng32::prepareForQueries() { internal_flinng.prepareForQueries(); }

std::vector<uint64_t> SparseFlinng32::query(const std::vector<std::vector<uint64_t>> &queries, uint64_t top_k) {
  std::vector<uint64_t> hashes = getHashes(queries);
  std::vector<uint64_t> results = internal_flinng.query(hashes, top_k);

  return results;
}

std::vector<uint64_t>
SparseFlinng32::querySameDim(const std::vector<uint64_t> &queries, uint64_t num_points, uint64_t point_dimension,
                             uint64_t top_k) {
  std::vector<uint64_t> hashes = getHashes(queries.data(), num_points, point_dimension);
  std::vector<uint64_t> results = internal_flinng.query(hashes, top_k);

  return results;
}
