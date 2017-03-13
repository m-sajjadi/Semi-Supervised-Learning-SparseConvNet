#pragma once
#include "Picture.h"
#include "Rng.h"
#include <vector>
#include <string>
#include "types.h"
#include "SpatiallySparseDataset.h"
#include <glob.h>

class Params {
public:
  Params(int nt, float lambda_mx, float lambda_ts, bool lab_aug, bool unlab_aug);
  int nt;
  float lambda_mx;
  float lambda_ts;
  
  bool lab_aug;
  bool unlab_aug;
};

class SpatiallySparseDataset {
  RNG rng;

public:
  Params *prms;
  std::string name;
  std::string header;
  std::vector<Picture *> pictures;
  int nFeatures;
  int nClasses;
  batchType type;
  void summary();
  void shuffle();
  SpatiallySparseDataset extractValidationSet(float p = 0.1);
  void subsetOfClasses(std::vector<int> activeClasses);
  SpatiallySparseDataset subset(int n);
  SpatiallySparseDataset balancedSubset(int n);
  void repeatSamples(int reps); // Make dataset seem n times bigger (i.e. for
                                // small datasets to avoid having v. small
                                // training epochs)
};

std::vector<std::string> globVector(const std::string &pattern);
