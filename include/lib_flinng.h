#pragma once

#include <chrono>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "Flinng.h"
#include "LshFunctions.h"
#include "io.h"

namespace flinng {

  void write_verify(void *ptr, size_t size, size_t count, FileIO &file);

  void read_verify(void *ptr, size_t size, size_t count, FileIO &file);

  class FlinngBuilder {

  public:
    uint64_t num_rows;
    uint64_t cells_per_row;
    uint64_t num_hash_tables;
    uint64_t hashes_per_table;
    uint64_t sub_hash_bits; //sub_hash_bits * hashes_per_table must be less than 32, otherwise segfault will happen
    uint64_t cut_off;


    FlinngBuilder(uint64_t num_rows = 3, uint64_t cells_per_row = (1 << 12),
                  uint64_t num_hash_tables = (1 << 9), uint64_t hashes_per_table = 14,
                  uint64_t sub_hash_bits = 2, uint64_t cut_off = 6)
        : num_rows(num_rows), cells_per_row(cells_per_row), num_hash_tables(num_hash_tables),
          hashes_per_table(hashes_per_table), sub_hash_bits(sub_hash_bits),
          cut_off(cut_off) {}
  };


  class BaseDenseFlinng32 {

  public:
    BaseDenseFlinng32(uint64_t num_rows, uint64_t cells_per_row,
                      uint64_t data_dimension, uint64_t num_hash_tables,
                      uint64_t hashes_per_table, uint64_t hash_range);

    static BaseDenseFlinng32 *from_index(const char *fname);

    void addPoints(const std::vector<float> &points);

    void addPoints(float *points, uint64_t num_points);

    void prepareForQueries();

    std::vector<uint64_t> query(const std::vector<float> &queries, uint32_t top_k);

    std::vector<uint64_t> query(float *queries, uint64_t num_queries, uint32_t top_k);

    void finalize_construction();

    void add(float *x, uint64_t num);

    void add_and_store(float *input, uint64_t num_items);

    void search(float *queries, unsigned n, unsigned k, long *ids);

    void search_with_distance(float *queries, unsigned n, unsigned k, long *ids, float *distances);

    void write_index(const char *fname);

    void fetch_descriptors(long id, float *desc);

  protected:
    BaseDenseFlinng32();

    Flinng internal_flinng;
    uint64_t num_hash_tables, hashes_per_table, data_dimension;
    std::vector<int8_t> rand_bits;

    std::vector<float> bases; /// database vectors, size ntotal * dimension

    void write_content_to_index(FileIO &index);

    void read_content_from_index(FileIO &index);

    static bool read_type_from_index(FileIO &index);

    virtual void write_type_to_index(FileIO &index) = 0;

    virtual void write_additional_content_to_index(FileIO &index) {}

    virtual void read_additional_content_from_index(FileIO &index) {}

    virtual float compute_distance(float *a, float *b) = 0;

    virtual std::vector<uint64_t> getHashes(const float *points, uint64_t num_points) = 0;
  };

  class DenseFlinng32 : public BaseDenseFlinng32 {
  public:
    DenseFlinng32();
    DenseFlinng32(uint64_t num_rows, uint64_t cells_per_row,
                  uint64_t data_dimension, uint64_t num_hash_tables, uint64_t hashes_per_table);

    DenseFlinng32(uint64_t data_dimension, FlinngBuilder *def = nullptr);
  protected:
    DenseFlinng32(uint64_t data_dimension, FlinngBuilder &&def);

    float compute_distance(float *a, float *b) override;

    inline std::vector<uint64_t> getHashes(const float *points, uint64_t num_points) override {
      return parallel_srp(points, num_points, data_dimension, rand_bits.data(), num_hash_tables, hashes_per_table);
    }

    void write_type_to_index(FileIO &index) override;
  };

  class L2DenseFlinng32 : public BaseDenseFlinng32 {

  public:
    L2DenseFlinng32();
    L2DenseFlinng32(uint64_t num_rows, uint64_t cells_per_row,
                    uint64_t data_dimension, uint64_t num_hash_tables,
                    uint64_t hashes_per_table, uint64_t sub_hash_bits = 2, uint64_t cutoff = 6);

    L2DenseFlinng32(uint64_t data_dimension, FlinngBuilder *def = nullptr);
  protected:
    uint64_t sub_hash_bits, cutoff;

    L2DenseFlinng32(uint64_t data_dimension, FlinngBuilder &&def);

    void write_additional_content_to_index(FileIO &index) override;

    void read_additional_content_from_index(FileIO &index) override;

    void write_type_to_index(FileIO &index) override;

    float compute_distance(float *a, float *b) override;

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

}; //end namespace flinng
