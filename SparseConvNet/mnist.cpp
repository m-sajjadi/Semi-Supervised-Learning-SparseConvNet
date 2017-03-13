#include "SparseConvNet.h"
#include "NetworkArchitectures.h"
#include "SpatiallySparseDatasetMnist.h"
#include <sstream>


int epoch = 0;
int cudaDevice = -1; //PCI bus ID, -1 for default GPU
int batchSize = 100;

// Total number of training samples: 60000
int n_labeled = 100;
int nt = 5;

float lambda_mx = 0.1;
float lambda_ts = 1;

bool lab_aug = false;
bool unlab_aug = false;


std::string Name="weights/mnist";                 


Picture* OpenCVPicture::distort(RNG& rng, batchType type) {
  OpenCVPicture* pic=new OpenCVPicture(*this);
    
  if (type==TRAINBATCH){    
    float c00 = 1, c01 = 0, c10 = 0, c11 = 1;
    c00 *= 1 + rng.uniform(-0.2, 0.2); // x stretch
    c11 *= 1 + rng.uniform(-0.2, 0.2); // y stretch
    int r = rng.randint(3);
    float alpha = rng.uniform(-0.2, 0.2);
    if (r == 0) // Slant
      matrixMul2x2inPlace(c00, c01, c10, c11, 1, 0, alpha, 1);
    if (r == 1) // Slant
      matrixMul2x2inPlace(c00, c01, c10, c11, 1, alpha, 0, 1);
    if (r == 2) // Rotate
      matrixMul2x2inPlace(c00, c01, c10, c11, cos(alpha), -sin(alpha),
                          sin(alpha), cos(alpha));
    pic->affineTransform(c00, c01, c10, c11);
    /* pic->jiggle(rng, 16); */
    pic->jiggle(rng, 2);
    
  }    
  return pic;
}

Picture *OpenCVPicture::distort2(RNG &rng, batchType type) {
  OpenCVPicture *pic = new OpenCVPicture(*this);    
  return pic;
}

class CNN : public SparseConvNet {
public:
  CNN (int dimension, int nInputFeatures, int nClasses, float p=0.0f, int cudaDevice=-1, int nTop=1);
};
CNN::CNN
(int dimension, int nInputFeatures, int nClasses, float p, int cudaDevice, int nTop)
  : SparseConvNet(dimension,nInputFeatures, nClasses, cudaDevice, nTop) {
  int l=0;
  
  addLeNetLayerPOFMP(32*(++l),2,1,2,powf(2,0.5),VLEAKYRELU,0);
  addLeNetLayerPOFMP(32*(++l),2,1,2,powf(2,0.5),VLEAKYRELU,0);
  addLeNetLayerPOFMP(32*(++l),2,1,2,powf(2,0.5),VLEAKYRELU,0);
  addLeNetLayerPOFMP(32*(++l),2,1,2,powf(2,0.5),VLEAKYRELU,0.1);
  addLeNetLayerPOFMP(32*(++l),2,1,2,powf(2,0.5),VLEAKYRELU,0.2);
  addLeNetLayerPOFMP(32*(++l),2,1,2,powf(2,0.5),VLEAKYRELU,0.3);
  addLeNetLayerMP   (32*(++l),2,1,1,1,          VLEAKYRELU,0.4);
  addLeNetLayerMP   (32*(++l),1,1,1,1,          VLEAKYRELU,0.5); 
  
  addSoftmaxLayer();
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

      SpatiallySparseDataset trainSet=MnistTrainSet();
      SpatiallySparseDataset testSet=MnistTestSet();
            
      trainSet.prms = new Params(nt, lambda_mx, lambda_ts, lab_aug, unlab_aug);
      testSet.prms = new Params(nt, lambda_mx, lambda_ts, lab_aug, unlab_aug);
      
      std::vector<int> index_train;
      int n_classes = trainSet.nClasses;
      index_train = find_index(trainSet.pictures, n_classes, n_labeled, nt);
      
      std::vector<int> index_test(testSet.pictures.size());
      for (int i=0; i<index_test.size(); i++) index_test[i] = i;
      
      
      trainSet.summary();
      testSet.summary();
      CNN cnn(2,trainSet.nFeatures,trainSet.nClasses,0.0f,cudaDevice);
      //DeepCNet cnn(2,5,32,VLEAKYRELU,trainSet.nFeatures,trainSet.nClasses,0.0f,cudaDevice);

      if (epoch>0)
	cnn.loadWeights(Name,epoch);
      for (epoch++;epoch<=400;epoch++) {
	std::cout <<"epoch: " << epoch << " " << std::flush;
	//This is the original learning rate:
	cnn.processDataset(trainSet, index_train, batchSize,0.003*exp(-0.01 * epoch)); 	
	
	if (epoch%1==0) {      
	  cnn.processDataset(testSet, index_test, batchSize);
	}
	if (epoch%40==0) {
	  cnn.saveWeights(Name,epoch);
	}
      }    
  
}
