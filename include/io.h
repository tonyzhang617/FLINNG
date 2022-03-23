#pragma once

#include <iostream>
#include <string.h>

namespace flinng {
  struct FileIO {
    std::string fname;
    FILE *fp;

    FileIO(const char *fname, bool write = false);

    ~FileIO();
  };
} //end namespace flinng
