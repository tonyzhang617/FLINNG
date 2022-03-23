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
         uint64_t hash_range);

  // All the hashes for point 1 come first, etc.
  // Size of hashes should be multiple of num_hash_tables
  void addPoints(const std::vector<uint64_t> &hashes);

  void prepareForQueries();

  // Again all the hashes for point 1 come first, etc.
  // Size of hashes should be multiple of num_hash_tables
  // Results are similarly ordered
  std::vector<uint64_t> query(const std::vector<uint64_t> &hashes, uint32_t top_k);

  uint64_t num_points_added() const;

  void write_content_to_index(flinng::FileIO &index);

  void read_content_from_index(flinng::FileIO &index);

private:
  uint64_t num_rows, cells_per_row, num_hash_tables, hash_range;
  uint64_t total_points_added = 0;
  std::vector<std::vector<uint32_t>> inverted_flinng_index;
  std::vector<std::vector<uint64_t>> cell_membership;
};

#endif