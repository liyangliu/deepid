#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/kps_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class KpsLayerTest: public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  KpsLayerTest()
      : blob_bottom_data_(new Blob<Dtype>()),
        blob_bottom_kps_(new Blob<Dtype>()),
        blob_top_data_(new Blob<Dtype>()) {
    blob_bottom_data_->Reshape(3, 3, 12, 8);
    blob_bottom_kps_->Reshape(3, 2*5, 1, 1);
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    int i = 0;
    int channels = blob_bottom_kps_->channels();
    blob_bottom_kps_->mutable_cpu_data()[2 + channels*i] = 2.2;
    blob_bottom_kps_->mutable_cpu_data()[3 + channels*i] = 4.2;
    i = 1;
    blob_bottom_kps_->mutable_cpu_data()[2 + channels*i] = 13.0;
    blob_bottom_kps_->mutable_cpu_data()[3 + channels*i] = 19.0;
    i = 2;
    blob_bottom_kps_->mutable_cpu_data()[2 + channels*i] = 8.0;
    blob_bottom_kps_->mutable_cpu_data()[3 + channels*i] = 12.0;

    blob_bottom_vec_.push_back(blob_bottom_kps_);
    blob_top_vec_.push_back(blob_top_data_);
  }
  virtual ~KpsLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_kps_;
    delete blob_top_data_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_kps_;
  Blob<Dtype>* const blob_top_data_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(KpsLayerTest, TestDtypesAndDevices);

TYPED_TEST(KpsLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  KpsParameter* kps_param =
      layer_param.mutable_kps_param();
  kps_param->set_kps_height(6);
  kps_param->set_kps_width(4);
  kps_param->set_kps_scale(0.5);
  kps_param->set_kps_id(1);
  KpsLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

} // namespace caffe
