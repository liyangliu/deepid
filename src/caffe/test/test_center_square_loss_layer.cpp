#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/center_square_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class CenterSquareLossLayerTest: public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CenterSquareLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(100, 2, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(100, 1, 1, 1)),
        blob_top_data_(new Blob<Dtype>()) {
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    for (int i = 0; i < 100; i++) {
      blob_bottom_label_->mutable_cpu_data()[i] = i / 10;
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_data_);
  }
  virtual ~CenterSquareLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_data_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_data_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CenterSquareLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(CenterSquareLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CenterSquareLossParameter* center_square_loss_param =
      layer_param.mutable_center_square_loss_param();
  center_square_loss_param->set_num_output(10);
  center_square_loss_param->set_margin(0.2);
  center_square_loss_param->mutable_center_filler()->set_type("gaussian");
  CenterSquareLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

} // namespace caffe
