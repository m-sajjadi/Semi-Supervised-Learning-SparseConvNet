#include "SpatiallySparseDatasetNORB.h"
#include <vector>
#include <iostream>
#include <fstream>

void readNORBFile(std::vector<Picture *> &characters, const char *filename, const char *filename2) {
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  std::ifstream file2(filename2, std::ios::in | std::ios::binary);
  if (!file) {
    std::cout << "Cannot find " << filename << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!file2) {
    std::cout << "Cannot find " << filename2 << std::endl;
    exit(EXIT_FAILURE);
  }
  
  file.ignore(24);
  file2.ignore(20);
  
  int label;  
  
  // cv::Mat mt;
  // mt.create(108, 108, CV_32FC(2));
  
  while (file2.read((char *)&label, 4)) {     
    
    unsigned char bitmap[23328];    
    file.read((char *)bitmap, 23328);
   
    
    OpenCVPicture *character = new OpenCVPicture(54, 54, 2, 128, label);                
    float *matData = ((float *)(character->mat.data));            
    
    for (int x = 0; x < 54; x++) {
      for (int y = 0; y < 54; y++) {
        for (int c = 0; c < 2; c++) {
          matData[y * 108 + x * 2 + (1 - c)] = bitmap[c *  11664 + (2*y) * 108 + (2*x)];
        }
      }
    }       
    
    /*
    OpenCVPicture *character = new OpenCVPicture(48, 48, 2, 128, label);
    float *matData = ((float *)(mt.data));
    
    for (int x = 0; x < 108; x++) {
      for (int y = 0; y < 108; y++) {
        for (int c = 0; c < 2; c++) {
          matData[y * 216 + x * 2 + (1 - c)] = bitmap[c *  11664 + y * 108 + x];
        }
      }
    }
    cv::resize(mt, character->mat, character->mat.size(), 0, 0, CV_INTER_LINEAR);    
    */
    
    characters.push_back(character);      

  }  
  
  file.close();
  file2.close();
}
SpatiallySparseDataset NORBTrainSet() {
  SpatiallySparseDataset dataset;
  dataset.name = "NORB train set";
  dataset.type = TRAINBATCH;
  dataset.nFeatures = 2;
  dataset.nClasses = 6;
  
  
  char filenameFormat[] = "Data/NORB/norb-5x46789x9x18x6x2x108x108-training-0%d-dat.mat";
  char filenameFormat2[] = "Data/NORB/norb-5x46789x9x18x6x2x108x108-training-0%d-cat.mat";
  char filename[100];
  char filename2[100];
  for (int fileNumber = 1; fileNumber <= 2; fileNumber++) {
    sprintf(filename, filenameFormat, fileNumber);
    sprintf(filename2, filenameFormat2, fileNumber);
    readNORBFile(dataset.pictures, filename, filename2);    
  }    
  return dataset;
}
SpatiallySparseDataset NORBTestSet() {
  SpatiallySparseDataset dataset;
  dataset.name = "NORB test set";
  dataset.type = TESTBATCH;
  dataset.nFeatures = 2;
  dataset.nClasses = 6;
  
  
  char filenameFormat[] = "Data/NORB/norb-5x01235x9x18x6x2x108x108-testing-0%d-dat.mat";
  char filenameFormat2[] = "Data/NORB/norb-5x01235x9x18x6x2x108x108-testing-0%d-cat.mat";
  char filename[100];
  char filename2[100];
  for (int fileNumber = 1; fileNumber <= 2; fileNumber++) {
    sprintf(filename, filenameFormat, fileNumber);
    sprintf(filename2, filenameFormat2, fileNumber);
    readNORBFile(dataset.pictures, filename, filename2);    
  }    
  return dataset;
}
