#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <cstring>
#include <vector>
#include "lib_flinng.h"

namespace flinng {
  struct FileIO {
    std::string fname;
    FILE *fp;

    FileIO(const char *fname, bool write = false)
        : fname(fname), fp(fopen(fname, write ? "wb" : "rb")) {}

    ~FileIO() {
      if (fp != nullptr) {
        fclose(fp);
      }
    }
  };

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
} //end namespace flinng