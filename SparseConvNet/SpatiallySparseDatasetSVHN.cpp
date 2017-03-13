#include "SpatiallySparseDatasetSVHN.h"
#include <vector>
#include <iostream>
#include <fstream>

void readSVHNFile(std::vector<Picture *> &characters, const char *filename) {
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  if (!file) {
    std::cout << "Cannot find " << filename << std::endl;
    exit(EXIT_FAILURE);
  }
  unsigned char label;  
  while (file.read((char *)&label, 1)) {
    OpenCVPicture *character = new OpenCVPicture(32, 32, 1, 128, label);
    unsigned char bitmap[1024];
    float *matData = ((float *)(character->mat.data));
    file.read((char *)bitmap, 1024);
    for (int y = 0; y < 32; y++) {
      for (int x = 0; x < 32; x++) {        
	matData[y * 32 + x] = bitmap[y * 32 + x];   	
      }
    }
    characters.push_back(character);
  }
  file.close();
}
SpatiallySparseDataset SVHNTrainSet() {
  SpatiallySparseDataset dataset;
  dataset.name = "SVHN train set";
  dataset.type = TRAINBATCH;
  dataset.nFeatures = 1;
  dataset.nClasses = 10;
  char filenameTrain[] = "Data/SVHN/train.bin";  
  readSVHNFile(dataset.pictures, filenameTrain);    
  return dataset;
}
SpatiallySparseDataset SVHNTestSet() {
  SpatiallySparseDataset dataset;
  dataset.name = "SVHN test set";
  dataset.type = TESTBATCH;
  dataset.nFeatures = 1;
  dataset.nClasses = 10;
  char filenameTest[] = "Data/SVHN/test.bin";
  readSVHNFile(dataset.pictures, filenameTest);
  return dataset;
}
