#include "SoftmaxClassifier.h"
#include <iostream>
#include <vector>
#include <cassert>
#include "utilities.h"
#include <cfloat>

__global__ void dDerivativeOfCostWRTpreSoftmaxTopLevelWeights(
    int batchSize, float *topDelta, float *topGrid, int *labels, int N) {
  for (int k = batchSize/2; k < batchSize; k++) {    
    for (int i = threadIdx.x; i < N; i += NTHREADS) {      
      topDelta[k * N + i] = topGrid[k * N + i] - (i == labels[k]);
    }
  }
  for (int k = 0; k < batchSize/2; k++) {    
    for (int i = threadIdx.x; i < N; i += NTHREADS) {      
      topDelta[k * N + i] = 0;
    }
  }  
}

__global__ void mx_backprop(int batchSize, float* out_data, float* in_data,
    int N, float coe) {
  for (int index = threadIdx.x; index < batchSize*N/2; index += NTHREADS) {      
    
    int ind = (index / N) * N;
    float p = 1;
    for (int i = 0; i<N; i++){
	if ((ind+i) != index ){
	    p = p * (1 - in_data[ ind + i ]);
	}
    }

    float t1 = p;
    
    float p0;
    float t2 = 0;    
    for (int i = 0; i<N; i++){
	if ((ind+i) != index ){
	    p0 = 1;
	    for (int j = 0; j<N; j++){
		if ( (ind+j) != index && j != i ){
		    p0 = p0 * (1 - in_data[ ind + j ]);
		}
	    }
	  
	    t2 = t2 + p0 * in_data[ ind + i ];	  
	}
    }
    t2 = -t2;
    float v = (t1 + t2);  
        
    out_data[index] = -coe*v;    
    
  }
  for (int index = batchSize*N/2+threadIdx.x; index < batchSize*N; index += NTHREADS) {      
    out_data[index] = 0;	
  }
}

__global__ void ts_forward(int batchSize, float* in_data,
    int nt, int N, float* out_data) {  
  for (int index = threadIdx.x; index < batchSize*N*nt/2; index += NTHREADS) {          
    int ind1 = (nt*N)*(index /(nt*nt*N)) + index % (nt*N);
    int ind2 = N*(index /(N*nt)) + index % N;    
    out_data[index] = in_data[ind1] - in_data[ind2];
  }
  for (int index = batchSize*N*nt/2+threadIdx.x; index < batchSize*N*nt; index += NTHREADS) {          
    out_data[index] = 0;
  }
}

__global__ void ts_backprop(int batchSize, float* in_data,
    int nt, int N, float* out_data, float coe) {  
  for (int index = threadIdx.x; index < batchSize*N/2; index += NTHREADS) {     
    int ind1 = (nt*N)*(index /N) + index % N;    
    for (int i=0; i<nt; i++){
      out_data[index] = out_data[index] - coe*in_data[ind1 + N*i];       
    }
  }
}

__global__ void Softmaxb(int batchSize, float* dE_dx_l, float* y_l, float* dE_dy_l,
    int N) {
  for (int index = threadIdx.x; index < batchSize*N/2; index += NTHREADS) {  
    int tx = index % N;
    int ty = N * (index / N);
    
    float v = 0;
    for (int j = 0; j < N; j++) {
	v += dE_dy_l[j + ty] * ((j == tx) - y_l[j + ty]);
    }
    v *= y_l[index];        
    dE_dx_l[index] = dE_dx_l[index] + v;     
  } 
}

void SoftmaxClassifier(SpatiallySparseBatchInterface &input,
                       SpatiallySparseBatch &batch, int nTop,
                       cudaMemStream &memStream, Params *prms) {
  // Assume no dropout in the output layer! nClasses:=input.nFeatures.
  assert(batch.batchSize == input.nSpatialSites);
  assert(input.nFeatures == input.featuresPresent.size());  
        
  int nt = prms->nt;  
  float lambda_mx = prms->lambda_mx;
  float lambda_ts = prms->lambda_ts;  
  
  vectorCUDA<float> tmp1;
  tmp1.resize(input.nSpatialSites * input.featuresPresent.size());        
  
  if (batch.type ==
      TRAINBATCH) { // Begin backprop. Top layer: d Cost / d SoftmaxInput    
    input.sub->dfeatures.resize(input.nSpatialSites * input.featuresPresent.size());
	
    dDerivativeOfCostWRTpreSoftmaxTopLevelWeights
            << <1, NTHREADS, 0, memStream.stream>>>
        (batch.batchSize, input.sub->dfeatures.dPtr(),
         input.sub->features.dPtr(), batch.labels.dPtr(), input.nFeatures);
    
    mx_backprop<< <1, NTHREADS, 0, memStream.stream>>>
        (batch.batchSize, tmp1.dPtr(),
         input.sub->features.dPtr(), input.nFeatures, lambda_mx);
    
  }  
  
  vectorCUDA<float> tmp2;
  tmp2.resize(input.nSpatialSites * nt * input.featuresPresent.size());
  
  
  ts_forward<< <1, NTHREADS, 0, memStream.stream>>>
      (batch.batchSize, input.sub->features.dPtr(), nt, input.nFeatures, tmp2.dPtr());    
  
  if (batch.type == TRAINBATCH){

    ts_backprop<< <1, NTHREADS, 0, memStream.stream>>>
        (batch.batchSize, tmp2.dPtr(), nt, input.nFeatures, tmp1.dPtr(), lambda_ts ); 
	    
    Softmaxb<< <1, NTHREADS, 0, memStream.stream>>>
        (batch.batchSize, input.sub->dfeatures.dPtr(),
         input.sub->features.dPtr(), tmp1.dPtr(), input.nFeatures);
	
  }  
  
  input.sub->features.copyToCPUAsync(memStream);
  batch.labels.copyToCPUAsync(memStream);
  tmp2.copyToCPUAsync(memStream);  
    
    
  for (int i=0; i < batch.batchSize*input.nFeatures; i++){    
    if (isnan(input.sub->features.hVector()[i])){
      std::cout << "Found NaN\n";
      exit(EXIT_FAILURE);
    }
  }
 
  
  float *probs = &input.sub->features.hVector()[0];
  for (int i = 0; i < batch.batchSize; ++i)
    batch.probabilities.push_back(std::vector<float>(
        probs + i * input.nFeatures, probs + (i + 1) * input.nFeatures));
  for (int i = 0; i < batch.batchSize; i++)
    batch.predictions.push_back(vectorTopIndices(batch.probabilities[i], nTop));

  if (batch.type != UNLABELEDBATCH) {
    batch.mistakes += batch.batchSize;
    for (int i = 0; i < batch.batchSize; i++) {
      /* batch.negativeLogLikelihood -=
          log(max(batch.probabilities[i][batch.labels.hVector()[i]], 1.0e-15)); */
      for (int j = 0; j < nTop; j++) {
        if (batch.predictions[i][j] == batch.labels.hVector()[i]) {
          batch.mistakes--;
        }
      }
    }
  }  
  
  if (batch.type == TRAINBATCH){
    float cost_ts = 0;  
    for (int i = 0; i < input.nSpatialSites * nt * input.featuresPresent.size()/2; i++){
      cost_ts = cost_ts + tmp2.hVector()[i]*tmp2.hVector()[i];
    }
    
    float cost_mx = 0;
    float p;
    for (int i = 0; i < batch.batchSize/2; i++){ 
      for (int j = 0; j < input.nFeatures; j++){
	p = 1;
	for (int k = 0; k < input.nFeatures; k++){
	  if (k != j){
	    p = p * (1 - batch.probabilities[i][k]);      
	  }
	}
	p = p * batch.probabilities[i][j];
	cost_mx = cost_mx + p;
      }
    }
    
    batch.negativeLogLikelihood += cost_ts;
  }
  
  input.sub->features.copyToGPUAsync(memStream);
  cudaCheckError();
}
