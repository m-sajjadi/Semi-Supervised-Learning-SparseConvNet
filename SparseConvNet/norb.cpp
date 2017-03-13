#include "NetworkArchitectures.h"
#include "SpatiallySparseDatasetNORB.h"


int epoch = 0;
int cudaDevice = -1; // PCI bus ID, -1 for default GPU
int batchSize = 50;

// Total number of training samples: 58320
int n_labeled = 582;
int nt = 5;

float lambda_mx = 0.1;
float lambda_ts = 1;

bool lab_aug = false;
bool unlab_aug = false;

std::string Name="weights/norb";

Picture* OpenCVPicture::distort(RNG& rng, batchType type) {
  OpenCVPicture* pic=new OpenCVPicture(*this);  
  if (type==TRAINBATCH) {
    float
      c00=1, c01=0,  //2x2 identity matrix---starting point for calculating affine distortion matrix
      c10=0, c11=1;
    c00*=1+rng.uniform(-0.2,0.2); // x stretch
    c11*=1+rng.uniform(-0.2,0.2); // y stretch
    if (rng.randint(2)==0) c00*=-1; //Horizontal flip
    int r=rng.randint(3);
    float alpha=rng.uniform(-0.2,0.2);
    if (r==0) matrixMul2x2inPlace(c00,c01,c10,c11,1,0,alpha,1); //Slant
    if (r==1) matrixMul2x2inPlace(c00,c01,c10,c11,1,alpha,0,1); //Slant other way
    if (r==2) matrixMul2x2inPlace(c00,c01,c10,c11,cos(alpha),-sin(alpha),sin(alpha),cos(alpha)); //Rotate
    transformImage(pic->mat, backgroundColor, c00, c01, c10, c11);
    pic->jiggle(rng,16);
  }
  return pic;
}

Picture *OpenCVPicture::distort2(RNG &rng, batchType type) {
  OpenCVPicture *pic = new OpenCVPicture(*this);  
  return pic;
}

std::vector<int> find_index(std::vector<Picture *> &pics, int n_classes, int n_labeled, int nt){
  int no_imgs = pics.size();
  int no_per_cls = n_labeled/n_classes;  
  
  
  RNG rng;  
  std::vector<int> ind_lab;
  if (n_labeled == no_imgs){
    ind_lab = std::vector<int>(no_imgs);
    for (int i=0; i<ind_lab.size(); i++) ind_lab[i] = i;
    rng.vectorShuffle(ind_lab);
    
  } else {  
    std::vector<std::vector<int>> inds(n_classes);
    for (int i=0; i<pics.size(); i++){
      inds[pics[i]->label].push_back(i);
    }  
    
    for (int i=0; i<inds.size(); i++){
      rng.vectorShuffle(inds[i]);
    }
    
    ind_lab = std::vector<int>(no_per_cls*n_classes);
    for (int i=0; i<inds.size(); i++){
      for (int j=0; j<no_per_cls; j++){
	ind_lab[i*no_per_cls+j] = inds[i][j];
      }
    }
  }
  
  std::vector<int> lab_rep(no_imgs*nt);
  for (int i=0; i<lab_rep.size(); i++){
    int j = i % ind_lab.size();
    if (j == 0){
      rng.vectorShuffle(ind_lab);
    }
    lab_rep[i] = ind_lab[j];
  }
  
  std::vector<int> init_ind(no_imgs);
  for (int i=0; i<init_ind.size(); i++) init_ind[i] = i;
  rng.vectorShuffle(init_ind);    
  
  int no;
  if ((2*no_imgs*nt) % batchSize == 0) no = 2*no_imgs*nt;
  else no = (((2*no_imgs*nt)/batchSize)+1)*batchSize;  
    
  std::vector<int> train_ind(no, 0);
  int k = 0;
  int l = 0;
  for (int i=0; i<train_ind.size(); i++){
    int j = i % batchSize;    
    if (j < batchSize/2){
      train_ind[i] = init_ind[(l/nt) % init_ind.size()];
      l++;
    } else {
      train_ind[i] = lab_rep[k % lab_rep.size()];
      k++;
    }
  }    
  
  return train_ind;   
}

int main() {
  
  SpatiallySparseDataset trainSet = NORBTrainSet();
  SpatiallySparseDataset testSet = NORBTestSet();
  
  trainSet.prms = new Params(nt, lambda_mx, lambda_ts, lab_aug, unlab_aug);
  testSet.prms = new Params(nt, lambda_mx, lambda_ts, lab_aug, unlab_aug);
  
  std::vector<int> index_train;
  int n_classes = trainSet.nClasses;
  index_train = find_index(trainSet.pictures, n_classes, n_labeled, nt);
    
  std::vector<int> index_test(testSet.pictures.size());
  for (int i=0; i<index_test.size(); i++) index_test[i] = i;
  
  
  trainSet.summary();
  testSet.summary();

  ROFMPSparseConvNet cnn(
      2, 12 , 32 /* 32n units in the n-th hidden layer*/, powf(2, 0.3333),
      VLEAKYRELU, trainSet.nFeatures, trainSet.nClasses,
      0.1f /*dropout multiplier in the range [0,0.5] */, cudaDevice);
  if (epoch > 0)
    cnn.loadWeights(Name, epoch);
  for (epoch++; epoch <= 410; epoch++) { 
    std::cout << "epoch: " << epoch << " " << std::flush;
    cnn.processDataset(trainSet, index_train, batchSize, 0.001 * exp(-0.01 * epoch), 0.99);
    if (epoch % 20 == 0)
      cnn.saveWeights(Name, epoch);

    if (epoch % 1 == 0)
      cnn.processDatasetRepeatTest(testSet, index_test, batchSize / 2, 3);
  }
  cnn.processDatasetRepeatTest(testSet, index_test, batchSize / 2, 100);
}
