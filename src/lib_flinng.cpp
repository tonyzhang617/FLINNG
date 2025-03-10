#include "lib_flinng.h"

static uint64_t power(const uint64_t base, const uint64_t exp) {
  uint64_t accu = 1;
  for (uint64_t e = 0; e < exp; ++e) {
    accu *= base;
  }
  return accu;
}

namespace flinng {
  void write_verify(void *ptr, size_t size, size_t count, FileIO &file) {
    size_t ret = fwrite(ptr, size, count, file.fp);
    if (ret != count) {
      std::cerr << "Error while writing to " << file.fname
                << " ret==" << ret << " != count==" << count << " errno: "
                << strerror(errno) << std::endl;
    }
  }

  void read_verify(void *ptr, size_t size, size_t count, FileIO &file) {
    size_t ret = fread(ptr, size, count, file.fp);
    if (ret != count) {
      std::cerr << "Error while reading " << file.fname
                << " ret==" << ret << " != count==" << count << " errno: "
                << strerror(errno) << std::endl;
    }
  }

  BaseDenseFlinng32::BaseDenseFlinng32(uint64_t num_rows, uint64_t cells_per_row, uint64_t data_dimension,
                                       uint64_t num_hash_tables,
                                       uint64_t hashes_per_table, uint64_t hash_range)
      : internal_flinng(num_rows,
                        cells_per_row,
                        num_hash_tables,
                        hash_range),
        num_hash_tables(num_hash_tables),
        hashes_per_table(hashes_per_table),
        data_dimension(data_dimension),
        rand_bits(num_hash_tables * hashes_per_table * data_dimension) {
    for (uint64_t i = 0; i < rand_bits.size(); i++) {
      rand_bits[i] = (rand() % 2) * 2 - 1; // 50% chance either 1 or -1
    }
  }

  BaseDenseFlinng32::BaseDenseFlinng32() : BaseDenseFlinng32(0, 0, 0, 0, 0, 0) {}

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

  void BaseDenseFlinng32::finalize_construction() {
    /**
     * place holder for when we need to do any post processing after
     * building the index, such as keeping maximum and normalizing dataset
     * for L2 metric
     */
    prepareForQueries();
  }

  void BaseDenseFlinng32::add(float *x, uint64_t num_points) {
    addPoints(x, num_points);
  }

  void BaseDenseFlinng32::add_and_store(float *input, uint64_t num_items) {
    add(input, num_items);
    bases.insert(bases.end(), input, input + num_items * data_dimension);
  }

  void BaseDenseFlinng32::search(float *queries, unsigned n, unsigned k, long *ids) {
    std::vector<uint64_t> results = query(queries, n, k);
    std::copy(results.begin(), results.end(), ids);
  }

  void BaseDenseFlinng32::search_with_distance(float *queries, unsigned n, unsigned k, long *ids, float *distances) {
    if (bases.size() / data_dimension != internal_flinng.num_points_added()) {
      std::cerr << "Dataset is not stored! Distance cannot be calculated. Invoke add_with_store() to store dataset."
                << std::endl;
      return;
    }

    search(queries, n, k, ids);

#pragma omp parallel for
    for (unsigned i = 0; i < n; i++) {
      for (unsigned j = 0; j < k; j++) {
        distances[i * k + j] = compute_distance(queries + data_dimension * i,
                                                bases.data() + data_dimension * ids[i * k + j]);
      }
    }
  }

  float DenseFlinng32::compute_distance(float *a, float *b) {
    float top = 0;
    float bottom_a = 0;
    float bottom_b = 0;
    for (uint64_t i = 0; i < data_dimension; ++i) {
      top += a[i] * b[i];
      bottom_a += a[i] * a[i];
      bottom_b += b[i] * b[i];
    }

    return 1 - top / (sqrtf(bottom_a) * sqrtf(bottom_b));
  }

  float L2DenseFlinng32::compute_distance(float *a, float *b) {
    float accu = 0;
    for (uint64_t i = 0; i < data_dimension; ++i) {
      float tmp = a[i] - b[i];
      accu += tmp * tmp;
    }

    return sqrtf(accu);
  }

  void BaseDenseFlinng32::fetch_descriptors(long id, float *desc) {
    std::copy(bases.begin() + id * data_dimension, bases.begin() + (id + 1) * data_dimension, desc);
  }

  BaseDenseFlinng32 * BaseDenseFlinng32::from_index(const char *fname) {
    FileIO idx_stream(fname);
    if (idx_stream.fp == NULL) {
      std::cerr << "Error occurred while opening index file for reading" << std::endl;
      return nullptr;
    }

    bool is_l2 = BaseDenseFlinng32::read_type_from_index(idx_stream);
    BaseDenseFlinng32 *obj;
    if (is_l2) {
      obj = new L2DenseFlinng32();
    } else {
      obj = new DenseFlinng32();
    }
    obj->read_content_from_index(idx_stream);
    obj->read_additional_content_from_index(idx_stream);

    return obj;
  }

  void BaseDenseFlinng32::write_content_to_index(FileIO &index) {
    internal_flinng.write_content_to_index(index);

    write_verify(&num_hash_tables, sizeof(num_hash_tables), 1, index);
    write_verify(&hashes_per_table, sizeof(hashes_per_table), 1, index);
    write_verify(&data_dimension, sizeof(data_dimension), 1, index);

    size_t tmp = rand_bits.size();
    write_verify(&tmp, sizeof(size_t), 1, index);
    write_verify(rand_bits.data(), sizeof(int8_t), rand_bits.size(), index);

    tmp = bases.size();
    write_verify(&tmp, sizeof(size_t), 1, index);
    write_verify(bases.data(), sizeof(float), bases.size(), index);
  }

  void BaseDenseFlinng32::read_content_from_index(FileIO &index) {
    internal_flinng.read_content_from_index(index);

    read_verify(&num_hash_tables, sizeof(num_hash_tables), 1, index);
    read_verify(&hashes_per_table, sizeof(hashes_per_table), 1, index);
    read_verify(&data_dimension, sizeof(data_dimension), 1, index);

    size_t tmp;
    read_verify(&tmp, sizeof(size_t), 1, index);
    rand_bits.resize(tmp);
    read_verify(rand_bits.data(), sizeof(int8_t), tmp, index);

    read_verify(&tmp, sizeof(size_t), 1, index);
    bases.resize(tmp);
    read_verify(bases.data(), sizeof(float), tmp, index);
  }


  void BaseDenseFlinng32::write_index(const char *fname) {
    FileIO idx_stream(fname, true);
    if (idx_stream.fp == NULL) {
      std::cerr << "Error occurred while opening index file for writing" << std::endl;
      return;
    }

    write_type_to_index(idx_stream);
    write_content_to_index(idx_stream);
    write_additional_content_to_index(idx_stream);
  }

  bool BaseDenseFlinng32::read_type_from_index(FileIO &index) {
    bool is_l2;
    read_verify(&is_l2, sizeof(bool), 1, index);
    return is_l2;
  }

  void DenseFlinng32::write_type_to_index(FileIO &index) {
    bool is_l2 = false;
    write_verify(&is_l2, sizeof(bool), 1, index);
  }

  void L2DenseFlinng32::write_type_to_index(FileIO &index) {
    bool is_l2 = true;
    write_verify(&is_l2, sizeof(bool), 1, index);
  }

  DenseFlinng32::DenseFlinng32()
    : DenseFlinng32(0, 0, 0, 0, 0) {

  }

  DenseFlinng32::DenseFlinng32(uint64_t num_rows, uint64_t cells_per_row,
                               uint64_t data_dimension, uint64_t num_hash_tables, uint64_t hashes_per_table)
      : BaseDenseFlinng32(num_rows, cells_per_row, data_dimension, num_hash_tables, hashes_per_table,
                          1 << hashes_per_table) {

  }

  DenseFlinng32::DenseFlinng32(uint64_t data_dimension, FlinngBuilder &&def)
      : BaseDenseFlinng32(def.num_rows, def.cells_per_row, data_dimension, def.num_hash_tables, def.hashes_per_table,
                          1 << def.hashes_per_table) {

  }

  DenseFlinng32::DenseFlinng32(uint64_t data_dimension, FlinngBuilder *def)
      : DenseFlinng32(data_dimension, def == nullptr ? FlinngBuilder() : *def) {

  }

  L2DenseFlinng32::L2DenseFlinng32(uint64_t num_rows, uint64_t cells_per_row,
                                   uint64_t data_dimension, uint64_t num_hash_tables,
                                   uint64_t hashes_per_table, uint64_t sub_hash_bits, uint64_t cutoff)
      : BaseDenseFlinng32(num_rows, cells_per_row, data_dimension, num_hash_tables, hashes_per_table,
                          power(1 << sub_hash_bits, hashes_per_table)),
        sub_hash_bits(sub_hash_bits), cutoff(cutoff) {}

  L2DenseFlinng32::L2DenseFlinng32(uint64_t data_dimension, FlinngBuilder &&def)
      : L2DenseFlinng32(def.num_rows, def.cells_per_row, data_dimension, def.num_hash_tables, def.hashes_per_table,
                        def.sub_hash_bits, def.cut_off) {}

  L2DenseFlinng32::L2DenseFlinng32(uint64_t data_dimension, FlinngBuilder *def)
      : L2DenseFlinng32(data_dimension, def == nullptr ? FlinngBuilder() : *def) {}

  L2DenseFlinng32::L2DenseFlinng32()
      : L2DenseFlinng32(0, 0, 0, 0, 0, 0, 0) {}

  void L2DenseFlinng32::write_additional_content_to_index(FileIO &index) {
    write_verify(&sub_hash_bits, sizeof(sub_hash_bits), 1, index);
    write_verify(&cutoff, sizeof(cutoff), 1, index);
  }

  void L2DenseFlinng32::read_additional_content_from_index(FileIO &index) {
    read_verify(&sub_hash_bits, sizeof(sub_hash_bits), 1, index);
    read_verify(&cutoff, sizeof(cutoff), 1, index);
  }

//todo add new APIs for SparseFlinng 

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

}; //end namespace flinng
