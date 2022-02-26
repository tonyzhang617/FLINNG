#include <iostream>
#include <vector>
#include <random>
#include "lib_flinng.h"

using namespace std;

int main() {
  uint64_t data_dim = 10, dataset_size = 10000, query_size = 100;
  float dataset_std = 1.0f, query_std = 0.1f;
  uint64_t flinng_num_rows = 3, flinngs_cells_per_row =
      dataset_size / 100, flinng_hashes_per_table = 12, flinng_num_hash_tables = 10;

  srand(100);
  default_random_engine generator;
  normal_distribution<float> dataset_dist(0.0f, dataset_std);
  normal_distribution<float> query_dist(0.0f, query_std);
  uniform_int_distribution<> uni_dist(0, dataset_size - 1);

  vector<float> dataset(dataset_size * data_dim);
  for (uint64_t i = 0; i < dataset.size(); ++i) {
    dataset[i] = dataset_dist(generator);
  }

  vector<float> queries(query_size * data_dim);
  vector<int> gt(query_size);
  for (uint64_t i = 0; i < query_size; ++i) {
    int e = uni_dist(generator);
    for (uint64_t j = 0; j < data_dim; ++j) {
      queries[i * data_dim + j] = dataset[e * data_dim + j] + query_dist(generator);
    }
    gt[i] = e;
  }

  {
    DenseFlinng32 index(flinng_num_rows, flinngs_cells_per_row, data_dim, flinng_num_hash_tables,
                        flinng_hashes_per_table);
    index.addPoints(dataset);
    index.prepareForQueries();
    auto res = index.query(queries, 1);
    uint32_t c = 0;
    for (uint64_t i = 0; i < query_size; ++i) {
      c += static_cast<int>(res[i]) == gt[i];
    }
    cout << "Recall (Angular Similarity) = " << static_cast<float>(c) / query_size << endl;
  }

  {
    L2DenseFlinng32 index(flinng_num_rows, flinngs_cells_per_row, data_dim, flinng_num_hash_tables,
                          flinng_hashes_per_table);
    index.addPoints(dataset);
    index.prepareForQueries();
    auto res = index.query(queries, 1);
    uint32_t c = 0;
    for (uint64_t i = 0; i < query_size; ++i) {
      c += static_cast<int>(res[i]) == gt[i];
    }
    cout << "Recall (L2 Similarity) = " << static_cast<float>(c) / query_size << endl;
  }

  return 0;
}