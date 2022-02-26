#include <vector>
#include "Flinng.h"
#include "LshFunctions.h"

class BaseFlinng32 {

public:
  BaseFlinng32(uint64_t num_rows, uint64_t cells_per_row,
                uint64_t data_dimension, uint64_t num_hash_tables,
                uint64_t hashes_per_table, uint64_t hash_range)
      : internal_flinng(num_rows, cells_per_row, num_hash_tables, hash_range),
        num_hash_tables(num_hash_tables), hashes_per_table(hashes_per_table),
        data_dimension(data_dimension),
        rand_bits(num_hash_tables * hashes_per_table * data_dimension) {

    for (uint64_t i = 0; i < rand_bits.size(); i++) {
      rand_bits[i] = (rand() % 2) * 2 - 1; // 50% chance either 1 or -1
    }
  }

  void addPoints(const std::vector<float> &points) {
    if (points.size() < data_dimension || points.size() % data_dimension != 0) {
      throw std::invalid_argument("The rows (each point) must be of dimension " +
                                  std::to_string(data_dimension) +
                                  ", and there must be at least 1 row.");
    }
    uint64_t num_points = points.size() / data_dimension;
    std::vector<uint64_t> hashes = getHashes(points.data(), num_points);
    internal_flinng.addPoints(hashes);
  }

  void addPoints(float *points, uint64_t num_points) {
    std::vector<uint64_t> hashes = getHashes(points, num_points);
    internal_flinng.addPoints(hashes);
  }

  void prepareForQueries() { internal_flinng.prepareForQueries(); }

  std::vector<uint64_t> query(const std::vector<float> &queries, uint32_t top_k) {
    if (queries.size() < data_dimension || queries.size() % data_dimension != 0) {
      throw std::invalid_argument("The rows (each point) must be of dimension " +
                                  std::to_string(data_dimension) +
                                  ", and there must be at least 1 row.");
    }
    uint64_t num_queries = queries.size() / data_dimension;
    std::vector<uint64_t> hashes = getHashes(queries.data(), num_queries);
    std::vector<uint64_t> results = internal_flinng.query(hashes, top_k);
    return move(results);
  }

  std::vector<uint64_t> query(float *queries, uint64_t num_queries, uint32_t top_k) {
    std::vector<uint64_t> hashes = getHashes(queries, num_queries);
    std::vector<uint64_t> results = internal_flinng.query(hashes, top_k);
    return move(results);
  }

protected:
  Flinng internal_flinng;
  const uint64_t num_hash_tables, hashes_per_table, data_dimension;
  std::vector<int8_t> rand_bits;

  virtual std::vector<uint64_t> getHashes(const float *points, uint64_t num_points) = 0;
};

class DenseFlinng32: public BaseFlinng32 {
public:
  DenseFlinng32(uint64_t num_rows, uint64_t cells_per_row,
                uint64_t data_dimension, uint64_t num_hash_tables, uint64_t hashes_per_table)
      : BaseFlinng32(num_rows, cells_per_row, data_dimension, num_hash_tables, hashes_per_table, 1 << hashes_per_table) {

  }
protected:
  inline std::vector<uint64_t> getHashes(const float *points, uint64_t num_points) override {
    return move(parallel_srp(points, num_points, data_dimension, rand_bits.data(), num_hash_tables, hashes_per_table));
  }
};

class L2DenseFlinng32: public BaseFlinng32 {

public:
  L2DenseFlinng32(uint64_t num_rows, uint64_t cells_per_row,
                  uint64_t data_dimension, uint64_t num_hash_tables,
                  uint64_t hashes_per_table, uint64_t sub_hash_bits = 2, uint64_t cutoff = 6)
      : BaseFlinng32(num_rows, cells_per_row, data_dimension, num_hash_tables, hashes_per_table, 1 << (hashes_per_table * sub_hash_bits)),
        sub_hash_bits(sub_hash_bits), cutoff(cutoff) {}

protected:
  const uint64_t sub_hash_bits, cutoff;

  inline std::vector<uint64_t> getHashes(const float *points, uint64_t num_points) override {
    return move(parallel_l2_lsh(points, num_points, data_dimension,
                                rand_bits.data(), num_hash_tables, hashes_per_table,
                                sub_hash_bits, cutoff));
  }
};

class SparseFlinng32 {

public:
  SparseFlinng32(uint64_t num_rows, uint64_t cells_per_row,
                 uint64_t num_hash_tables, uint64_t hashes_per_table,
                 uint64_t hash_range_pow)
      : internal_flinng(num_rows, cells_per_row, num_hash_tables,
                        1 << hash_range_pow),
        num_hash_tables(num_hash_tables), hashes_per_table(hashes_per_table),
        hash_range_pow(hash_range_pow), seed(rand()) {}

  void addPointsSameDim(const uint64_t *points, uint64_t num_points, uint64_t point_dimension) {
    std::vector<uint64_t> hashes = getHashes(points, num_points, point_dimension);
    internal_flinng.addPoints(hashes);
  }

  void addPointsSameDim(const std::vector<uint64_t> &points, uint64_t num_points, uint64_t point_dimension) {
    std::vector<uint64_t> hashes = getHashes(points.data(), num_points, point_dimension);
    internal_flinng.addPoints(hashes);
  }

  void addPoints(const std::vector<std::vector<uint64_t>> &data) {
    std::vector<uint64_t> hashes = getHashes(data);
    internal_flinng.addPoints(hashes);
  }

  std::vector<uint64_t> hashPoints(const std::vector<std::vector<uint64_t>> &data) {
    return getHashes(data);
  }

  void prepareForQueries() { internal_flinng.prepareForQueries(); }

  std::vector<uint64_t> query(const std::vector<std::vector<uint64_t>> &queries, uint64_t top_k) {
    std::vector<uint64_t> hashes = getHashes(queries);
    std::vector<uint64_t> results = internal_flinng.query(hashes, top_k);

    return move(results);
  }

  std::vector<uint64_t> querySameDim(const std::vector<uint64_t> &queries, uint64_t num_points, uint64_t point_dimension, uint64_t top_k) {
    std::vector<uint64_t> hashes = getHashes(queries.data(), num_points, point_dimension);
    std::vector<uint64_t> results = internal_flinng.query(hashes, top_k);

    return move(results);
  }

protected:
  Flinng internal_flinng;
  const uint64_t num_hash_tables, hashes_per_table, hash_range_pow;
  const uint32_t seed;

  inline std::vector<uint64_t> getHashes(const uint64_t *points, uint64_t num_points, uint64_t point_dimension) {
    return move(parallel_densified_minhash(points, num_points, point_dimension, num_hash_tables, hashes_per_table, hash_range_pow, seed));
  }

  inline std::vector<uint64_t> getHashes(const std::vector<std::vector<uint64_t>> &data) {
    return move(parallel_densified_minhash(data, num_hash_tables, hashes_per_table, hash_range_pow, seed));
  }
};
