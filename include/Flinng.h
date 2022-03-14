#ifndef _FLING
#define _FLING

#include "LshFunctions.h"
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include "io.h"

// TODO: Add back 16 bit FLINNG, check input
// TODO: Reproduce experiments
// TODO: Add percent of srp used
class Flinng {

public:
  Flinng(uint64_t num_rows, uint64_t cells_per_row, uint64_t num_hashes,
         uint64_t hash_range)
      : num_rows(num_rows), cells_per_row(cells_per_row),
        num_hash_tables(num_hashes), hash_range(hash_range),
        inverted_flinng_index(hash_range * num_hashes),
        cell_membership(num_rows * cells_per_row) {}

  // All the hashes for point 1 come first, etc.
  // Size of hashes should be multiple of num_hash_tables
  void addPoints(const std::vector<uint64_t> &hashes) {

    uint64_t num_points = hashes.size() / num_hash_tables;
    std::vector<uint64_t> random_buckets(num_rows * num_points);
    for (uint64_t i = 0; i < num_rows * num_points; i++) {
      random_buckets[i] =
          (rand() % cells_per_row + cells_per_row) % cells_per_row +
          (i % num_rows) * cells_per_row;
    }

#pragma omp parallel for
    for (uint64_t table = 0; table < num_hash_tables; table++) {
      for (uint64_t point = 0; point < num_points; point++) {
        uint64_t hash = hashes[point * num_hash_tables + table];
        uint64_t hash_id = table * hash_range + hash;
        for (uint64_t row = 0; row < num_rows; row++) {
          inverted_flinng_index[hash_id].push_back(
              random_buckets[point * num_rows + row]);
        }
      }
    }

    for (uint64_t point = 0; point < num_points; point++) {
      for (uint64_t row = 0; row < num_rows; row++) {
        cell_membership[random_buckets[point * num_rows + row]].push_back(
            total_points_added + point);
      }
    }

    total_points_added += num_points;

    prepareForQueries();
  }

  void prepareForQueries() {
    for (uint64_t i = 0; i < inverted_flinng_index.size(); i++) {
      std::sort(inverted_flinng_index[i].begin(),
                inverted_flinng_index[i].end());
      inverted_flinng_index[i].erase(
          std::unique(inverted_flinng_index[i].begin(),
                      inverted_flinng_index[i].end()),
          inverted_flinng_index[i].end());
    }
  }

  // Again all the hashes for point 1 come first, etc.
  // Size of hashes should be multiple of num_hash_tables
  // Results are similarly ordered
  std::vector<uint64_t> query(const std::vector<uint64_t> &hashes, uint32_t top_k) {

    uint64_t num_queries = hashes.size() / num_hash_tables;
    std::vector<uint64_t> results(top_k * num_queries);

#pragma omp parallel for
    for (uint32_t query_id = 0; query_id < num_queries; query_id++) {

      std::vector<uint32_t> counts(num_rows * cells_per_row, 0);
      for (uint32_t rep = 0; rep < num_hash_tables; rep++) {
        const uint32_t index =
            hash_range * rep + hashes[num_hash_tables * query_id + rep];
        const uint32_t size = inverted_flinng_index[index].size();
        for (uint32_t small_index = 0; small_index < size; small_index++) {
          // This single line takes 80% of the time, around half for the move
          // and half for the add
          ++counts[inverted_flinng_index[index][small_index]];
        }
      }

      std::vector<uint32_t> sorted[num_hash_tables + 1];
      uint32_t size_guess = num_rows * cells_per_row / (num_hash_tables + 1);
      for (std::vector<uint32_t> &v: sorted) {
        v.reserve(size_guess);
      }

      for (uint32_t i = 0; i < num_rows * cells_per_row; ++i) {
        sorted[counts[i]].push_back(i);
      }

      if (num_rows > 2) {
        std::vector<uint8_t> num_counts(total_points_added, 0);
        uint32_t num_found = 0;
        for (int32_t rep = num_hash_tables; rep >= 0; --rep) {
          for (uint32_t bin: sorted[rep]) {
            for (uint32_t point: cell_membership[bin]) {
              if (++num_counts[point] == num_rows) {
                results[top_k * query_id + num_found] = point;
                if (++num_found == top_k) {
                  goto end_of_query;
                }
              }
            }
          }
        }
      } else {
        char *num_counts =
            (char *) calloc(total_points_added / 8 + 1, sizeof(char));
        uint32_t num_found = 0;
        for (int32_t rep = num_hash_tables; rep >= 0; --rep) {
          for (uint32_t bin: sorted[rep]) {
            for (uint32_t point: cell_membership[bin]) {
              if (num_counts[(point / 8)] & (1 << (point % 8))) {
                results[top_k * query_id + num_found] = point;
                if (++num_found == top_k) {
                  free(num_counts);
                  goto end_of_query;
                }
              } else {
                num_counts[(point / 8)] |= (1 << (point % 8));
              }
            }
          }
        }
      }
      end_of_query:;
    }

    return results;
  }

  uint64_t num_points_added() const {
    return total_points_added;
  }

  void write_content_to_index(flinng::FileIO &index) {
    flinng::write_verify(&num_rows, sizeof(num_rows), 1, index);
    flinng::write_verify(&cells_per_row, sizeof(cells_per_row), 1, index);
    flinng::write_verify(&num_hash_tables, sizeof(num_hash_tables), 1, index);
    flinng::write_verify(&hash_range, sizeof(hash_range), 1, index);
    flinng::write_verify(&total_points_added, sizeof(total_points_added), 1, index);

    size_t tmp = inverted_flinng_index.size();
    flinng::write_verify(&tmp, sizeof(size_t), 1, index);
    for (size_t i = 0; i < tmp; ++i) {
      size_t tmp2 = inverted_flinng_index[i].size();
      flinng::write_verify(&tmp2, sizeof(size_t), 1, index);
      flinng::write_verify(inverted_flinng_index[i].data(), sizeof(uint32_t), tmp2, index);
    }

    tmp = cell_membership.size();
    flinng::write_verify(&tmp, sizeof(size_t), 1, index);
    for (size_t i = 0; i < tmp; ++i) {
      size_t tmp2 = cell_membership[i].size();
      flinng::write_verify(&tmp2, sizeof(size_t), 1, index);
      flinng::write_verify(cell_membership[i].data(), sizeof(uint64_t), tmp2, index);
    }
  }

  void read_content_from_index(flinng::FileIO &index) {
    flinng::read_verify(&num_rows, sizeof(num_rows), 1, index);
    flinng::read_verify(&cells_per_row, sizeof(cells_per_row), 1, index);
    flinng::read_verify(&num_hash_tables, sizeof(num_hash_tables), 1, index);
    flinng::read_verify(&hash_range, sizeof(hash_range), 1, index);
    flinng::read_verify(&total_points_added, sizeof(total_points_added), 1, index);

    size_t tmp;
    flinng::read_verify(&tmp, sizeof(size_t), 1, index);
    inverted_flinng_index.resize(tmp);
    for (size_t i = 0; i < tmp; ++i) {
      size_t tmp2;
      flinng::read_verify(&tmp2, sizeof(size_t), 1, index);
      inverted_flinng_index[i].resize(tmp2);
      flinng::read_verify(inverted_flinng_index[i].data(), sizeof(uint32_t), tmp2, index);
    }

    flinng::read_verify(&tmp, sizeof(size_t), 1, index);
    cell_membership.resize(tmp);
    for (size_t i = 0; i < tmp; ++i) {
      size_t tmp2;
      flinng::read_verify(&tmp2, sizeof(size_t), 1, index);
      cell_membership[i].resize(tmp2);
      flinng::read_verify(cell_membership[i].data(), sizeof(uint64_t), tmp2, index);
    }
  }

private:
  uint64_t num_rows, cells_per_row, num_hash_tables, hash_range;
  uint64_t total_points_added = 0;
  std::vector<std::vector<uint32_t>> inverted_flinng_index;
  std::vector<std::vector<uint64_t>> cell_membership;
};

#endif