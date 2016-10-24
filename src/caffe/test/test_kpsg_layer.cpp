#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/kpsg_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class KpsgLayerTest: public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  KpsgLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(3, 3, 12, 8)),
        blob_bottom_kpsg_(new Blob<Dtype>(3, 10, 1, 1)),
        blob_top_data_(new Blob<Dtype>()) {
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    int i = 0;
    blob_bottom_kpsg_->mutable_cpu_data()[0 + 10*i] = 1.1;
    blob_bottom_kpsg_->mutable_cpu_data()[1 + 10*i] = 2.1;
    blob_bottom_kpsg_->mutable_cpu_data()[2 + 10*i] = 6.0;
    blob_bottom_kpsg_->mutable_cpu_data()[3 + 10*i] = 2.2;
    blob_bottom_kpsg_->mutable_cpu_data()[4 + 10*i] = 4.2;
    blob_bottom_kpsg_->mutable_cpu_data()[5 + 10*i] = 6.3;
    blob_bottom_kpsg_->mutable_cpu_data()[6 + 10*i] = 1.2;
    blob_bottom_kpsg_->mutable_cpu_data()[7 + 10*i] = 8.7;
    blob_bottom_kpsg_->mutable_cpu_data()[8 + 10*i] = 6.5;
    blob_bottom_kpsg_->mutable_cpu_data()[9 + 10*i] = 9.5;
    i = 1;
    blob_bottom_kpsg_->mutable_cpu_data()[0 + 10*i] = 1.1;
    blob_bottom_kpsg_->mutable_cpu_data()[1 + 10*i] = 2.1;
    blob_bottom_kpsg_->mutable_cpu_data()[2 + 10*i] = 5.0;
    blob_bottom_kpsg_->mutable_cpu_data()[3 + 10*i] = 6.0;
    blob_bottom_kpsg_->mutable_cpu_data()[4 + 10*i] = 3.5;
    blob_bottom_kpsg_->mutable_cpu_data()[5 + 10*i] = 4.5;
    blob_bottom_kpsg_->mutable_cpu_data()[6 + 10*i] = 7.1;
    blob_bottom_kpsg_->mutable_cpu_data()[7 + 10*i] = 8.2;
    blob_bottom_kpsg_->mutable_cpu_data()[8 + 10*i] = 9.6;
    blob_bottom_kpsg_->mutable_cpu_data()[9 + 10*i] = 10.2;
    i = 2;
    blob_bottom_kpsg_->mutable_cpu_data()[0 + 10*i] = 4.5;
    blob_bottom_kpsg_->mutable_cpu_data()[1 + 10*i] = 5.5;
    blob_bottom_kpsg_->mutable_cpu_data()[2 + 10*i] = 5.4;
    blob_bottom_kpsg_->mutable_cpu_data()[3 + 10*i] = 6.2;
    blob_bottom_kpsg_->mutable_cpu_data()[4 + 10*i] = 8.0;
    blob_bottom_kpsg_->mutable_cpu_data()[5 + 10*i] = 12.0;
    blob_bottom_kpsg_->mutable_cpu_data()[6 + 10*i] = 3.6;
    blob_bottom_kpsg_->mutable_cpu_data()[7 + 10*i] = 9.3;
    blob_bottom_kpsg_->mutable_cpu_data()[8 + 10*i] = 4.1;
    blob_bottom_kpsg_->mutable_cpu_data()[9 + 10*i] = 8.0;

    blob_bottom_vec_.push_back(blob_bottom_kpsg_);
    blob_top_vec_.push_back(blob_top_data_);
  }
  virtual ~KpsgLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_kpsg_;
    delete blob_top_data_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_kpsg_;
  Blob<Dtype>* const blob_top_data_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(KpsgLayerTest, TestDtypesAndDevices);

TYPED_TEST(KpsgLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  KpsgParameter* kpsg_param =
      layer_param.mutable_kpsg_param();
  kpsg_param->set_kpsg_height(6);
  kpsg_param->set_kpsg_width(4);
  KpsgLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

} // namespace caffe
