#include <iostream>
#include "io.h"

flinng::FileIO::FileIO(const char *fname, bool write)
    : fname(fname), fp(fopen(fname, write ? "wb" : "rb")) {}

flinng::FileIO::~FileIO() {
  if (fp != nullptr) {
    fclose(fp);
  }
}
