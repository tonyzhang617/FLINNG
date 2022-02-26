#include <vector>
#include "Flinng.h"
#include "LshFunctions.h"

class BaseDenseFlinng32 {

public:
  BaseDenseFlinng32(uint64_t num_rows, uint64_t cells_per_row,
                    uint64_t data_dimension, uint64_t num_hash_tables,
                    uint64_t hashes_per_table, uint64_t hash_range);

  void addPoints(const std::vector<float> &points);

  void addPoints(float *points, uint64_t num_points);

  void prepareForQueries();

  std::vector<uint64_t> query(const std::vector<float> &queries, uint32_t top_k);

  std::vector<uint64_t> query(float *queries, uint64_t num_queries, uint32_t top_k);

protected:
  Flinng internal_flinng;
  const uint64_t num_hash_tables, hashes_per_table, data_dimension;
  std::vector<int8_t> rand_bits;

  virtual std::vector<uint64_t> getHashes(const float *points, uint64_t num_points) = 0;
};

class DenseFlinng32 : public BaseDenseFlinng32 {
public:
  DenseFlinng32(uint64_t num_rows, uint64_t cells_per_row,
                uint64_t data_dimension, uint64_t num_hash_tables, uint64_t hashes_per_table);

protected:
  inline std::vector<uint64_t> getHashes(const float *points, uint64_t num_points) override {
    return parallel_srp(points, num_points, data_dimension, rand_bits.data(), num_hash_tables, hashes_per_table);
  }
};

class L2DenseFlinng32 : public BaseDenseFlinng32 {

public:
  L2DenseFlinng32(uint64_t num_rows, uint64_t cells_per_row,
                  uint64_t data_dimension, uint64_t num_hash_tables,
                  uint64_t hashes_per_table, uint64_t sub_hash_bits = 2, uint64_t cutoff = 6);

protected:
  const uint64_t sub_hash_bits, cutoff;

  inline std::vector<uint64_t> getHashes(const float *points, uint64_t num_points) override {
    return parallel_l2_lsh(points, num_points, data_dimension,
                           rand_bits.data(), num_hash_tables, hashes_per_table,
                           sub_hash_bits, cutoff);
  }
};

class SparseFlinng32 {

public:
  SparseFlinng32(uint64_t num_rows, uint64_t cells_per_row,
                 uint64_t num_hash_tables, uint64_t hashes_per_table,
                 uint64_t hash_range_pow);

  void addPointsSameDim(const uint64_t *points, uint64_t num_points, uint64_t point_dimension);

  void addPointsSameDim(const std::vector<uint64_t> &points, uint64_t num_points, uint64_t point_dimension);

  void addPoints(const std::vector<std::vector<uint64_t>> &data);

  std::vector<uint64_t> hashPoints(const std::vector<std::vector<uint64_t>> &data);

  void prepareForQueries();

  std::vector<uint64_t> query(const std::vector<std::vector<uint64_t>> &queries, uint64_t top_k);

  std::vector<uint64_t>
  querySameDim(const std::vector<uint64_t> &queries, uint64_t num_points, uint64_t point_dimension, uint64_t top_k);

protected:
  Flinng internal_flinng;
  const uint64_t num_hash_tables, hashes_per_table, hash_range_pow;
  const uint32_t seed;

  inline std::vector<uint64_t> getHashes(const uint64_t *points, uint64_t num_points, uint64_t point_dimension) {
    return parallel_densified_minhash(points, num_points, point_dimension, num_hash_tables, hashes_per_table,
                                      hash_range_pow, seed);
  }

  inline std::vector<uint64_t> getHashes(const std::vector<std::vector<uint64_t>> &data) {
    return parallel_densified_minhash(data, num_hash_tables, hashes_per_table, hash_range_pow, seed);
  }
};
