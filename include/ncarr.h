#pragma once

typedef struct {
  int    idx;
  int    time;
  bool   write = false;
  bool   write_v_time = false;
  size_t start[5];
  size_t count[5];
  size_t time_start[6];
  size_t time_count[6];
  int    dims[5];
  int    time_dims[6];
  int    ikx;
  int    iky;
  int    ns;
  void increment_ts(void) {time_start[0] += 1;}
} nca;
