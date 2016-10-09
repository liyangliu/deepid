#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/kps_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void KpsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  KpsParameter kps_param = this->layer_param_.kps_param();
  num_points_ = bottom[1]->channels()/2;
  kps_height_ = kps_param.kps_height();
  kps_width_ = kps_param.kps_width();
  kps_id_ = kps_param.kps_id();
  kps_scale_ = kps_param.kps_scale();
  LOG(INFO) << "Kps scale: " << kps_scale_;
}


template <typename Dtype>
void KpsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(num_, channels_, kps_height_, kps_width_);
}


template <typename Dtype>
void KpsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_kps = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < num_; n++) {
      const int kps_x = round(bottom_kps[2*kps_id_]*kps_scale_);
      const int kps_y = round(bottom_kps[2*kps_id_+1]*kps_scale_);
      int hstart = kps_y - kps_height_ / 2;
      int hend = hstart + kps_height_ - 1;
      int wstart = kps_x - kps_width_ / 2;
      int wend = wstart + kps_width_ - 1;
      if (hstart < 0) {
        hstart = 0;
        hend = hstart + kps_height_ - 1;
      }
      else if (hend > height_ - 1) {
        hend = height_ - 1;
        hstart = hend - (kps_height_ - 1);
      }
      if (wstart < 0) {
        wstart = 0;
        wend = wstart + kps_width_ - 1;
      }
      else if (wend > width_ - 1) {
        wend = width_ - 1;
        wstart = wend - (kps_width_ - 1);
      }
      CHECK_GE(wstart, 0);
      CHECK_LT(wend, width_);
      CHECK_GE(hstart, 0);
      CHECK_LT(hend, height_);
      CHECK_EQ(wend - wstart, kps_width_ - 1);
      CHECK_EQ(hend - hstart, kps_height_ - 1);
      for (int c = 0; c < channels_; c++) {
          for (int kh = 0; kh < kps_height_; kh++) {
              for (int kw = 0; kw < kps_width_; kw++) {
                  const int top_index = kh * kps_width_ + kw;
                  const int bottom_index = (kh + hstart) * width_ + kw + wstart;
                  top_data[top_index] = bottom_data[bottom_index];
              }
          }
          top_data += top[0]->offset(0, 1);
          bottom_data += bottom[0]->offset(0, 1);
      }
      bottom_kps += bottom[1]->offset(1);
    }
}


template <typename Dtype>
void KpsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* bottom_kps = bottom[1]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  const int count = bottom[0]->count();
  caffe_set(count, Dtype(0.), bottom_diff);
  for (int n = 0; n < num_; n++) {
      const int kps_x = round(bottom_kps[2*kps_id_]*kps_scale_);
      const int kps_y = round(bottom_kps[2*kps_id_+1]*kps_scale_);
      int hstart = kps_y - kps_height_ / 2;
      int hend = hstart + kps_height_ - 1;
      int wstart = kps_x - kps_width_ / 2;
      int wend = wstart + kps_width_ - 1;
      if (hstart < 0) {
        hstart = 0;
        hend = hstart + kps_height_ - 1;
      }
      else if (hend > height_ - 1) {
        hend = height_ - 1;
        hstart = hend - (kps_height_ - 1);
      }
      if (wstart < 0) {
        wstart = 0;
        wend = wstart + kps_width_ - 1;
      }
      else if (wend > width_ - 1) {
        wend = width_ - 1;
        wstart = wend - (kps_width_ - 1);
      }
      CHECK_GE(wstart, 0);
      CHECK_LT(wend, width_);
      CHECK_GE(hstart, 0);
      CHECK_LT(hend, height_);
      CHECK_EQ(wend - wstart, kps_width_ - 1);
      CHECK_EQ(hend - hstart, kps_height_ - 1);
      for (int c = 0; c < channels_; c++) {
          for (int h = hstart; h <= hend; h++) {
              for (int w = wstart; w <= wend; w++) {
                const int top_index = (h - hstart) * kps_width_ + w - wstart;
                const int bottom_index = h * width_ + w;
                bottom_diff[bottom_index] = top_diff[top_index];
              }
          }
          top_diff += top[0]->offset(0, 1);
          bottom_diff += bottom[0]->offset(0, 1);
      }
      bottom_kps += bottom[1]->offset(1);
  }
}


#ifdef CPU_ONLY
STUB_GPU(KpsLayer);
#endif

INSTANTIATE_CLASS(KpsLayer);
REGISTER_LAYER_CLASS(Kps);

}  // namespace caffe
