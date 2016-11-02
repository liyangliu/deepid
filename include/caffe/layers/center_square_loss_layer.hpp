#ifndef CAFFE_CENTER_SQUARE_LOSS_LAYER_HPP_
#define CAFFE_CENTER_SQUARE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class CenterSquareLossLayer : public LossLayer<Dtype> {
 public:
  explicit CenterSquareLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CenterSquareLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return -1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;
  int K_;
  int N_;
  Dtype margin_;
  
  Blob<Dtype> distance_;
  Blob<Dtype> variation_sum_;
  Blob<Dtype> mat_; // yi != yj && ||xi - cyi||^2 - ||xi - cyj||^2 + a > 0
  Blob<Dtype> distance_mat_; // cyi - cyj
  Blob<Dtype> If_; // [I, I, ..., I]: K * (M * K)
  Blob<Dtype> Ic_; // [1, 1, ..., 1]: M
  Blob<Dtype> Ia_; // [1, 1, ..., 1]: M * M
  Blob<Dtype> Ib_; // [1, 1, ... 1, 0, 0, ..., 0; 0, 0, ..., 0, 1, 1, ..., 1, 0, 0, ..., 0; ...; 0, 0, ..., 0, 1, 1, ..., 1]: M * (M * M)
  Blob<Dtype> Im_; // [1, 1, ..., 1]: K
};

}  // namespace caffe

#endif  // CAFFE_CENTER_SQUARE_LOSS_LAYER_HPP_
