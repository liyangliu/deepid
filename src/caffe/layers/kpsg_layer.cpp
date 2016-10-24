#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/kpsg_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void KpsgLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  KpsgParameter kpsg_param = this->layer_param_.kpsg_param();
  num_points_ = bottom[1]->channels()/2;
  kpsg_height_ = kpsg_param.kpsg_height();
  kpsg_width_ = kpsg_param.kpsg_width();
  kpsg_scale_ = kpsg_param.kpsg_scale();
  LOG(INFO) << "Kpsg scale: " << kpsg_scale_;
}


template <typename Dtype>
void KpsgLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(num_, channels_*num_points_, kpsg_height_, kpsg_width_);
}


template <typename Dtype>
void KpsgLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_kpsg = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < num_; n++) {
      for (int kpsg_id = 0; kpsg_id < num_points_; kpsg_id ++) {
          const int kpsg_x = round(bottom_kpsg[2*kpsg_id]*kpsg_scale_);
          const int kpsg_y = round(bottom_kpsg[2*kpsg_id+1]*kpsg_scale_);
          int hstart = kpsg_y - kpsg_height_ / 2;
          int hend = hstart + kpsg_height_ - 1;
          int wstart = kpsg_x - kpsg_width_ / 2;
          int wend = wstart + kpsg_width_ - 1;
          if (hstart < 0) {
            hstart = 0;
            hend = hstart + kpsg_height_ - 1;
          }
          else if (hend > height_ - 1) {
            hend = height_ - 1;
            hstart = hend - (kpsg_height_ - 1);
          }
          if (wstart < 0) {
            wstart = 0;
            wend = wstart + kpsg_width_ - 1;
          }
          else if (wend > width_ - 1) {
            wend = width_ - 1;
            wstart = wend - (kpsg_width_ - 1);
          }
          CHECK_GE(wstart, 0);
          CHECK_LT(wend, width_);
          CHECK_GE(hstart, 0);
          CHECK_LT(hend, height_);
          CHECK_EQ(wend - wstart, kpsg_width_ - 1);
          CHECK_EQ(hend - hstart, kpsg_height_ - 1);
          for (int c = 0; c < channels_; c++) {
              for (int kh = 0; kh < kpsg_height_; kh++) {
                  for (int kw = 0; kw < kpsg_width_; kw++) {
                      const int top_index = kh * kpsg_width_ + kw;
                      const int bottom_index = (kh + hstart) * width_ + kw + wstart;
                      top_data[top_index] = bottom_data[bottom_index];
                  }
              }
              top_data += top[0]->offset(0, 1);
              bottom_data += bottom[0]->offset(0, 1);
          }
          bottom_data -= bottom[0]->offset(0, channels_);
        }
      bottom_kpsg += bottom[1]->offset(1);
  }
}


template <typename Dtype>
void KpsgLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* bottom_kpsg = bottom[1]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  const int count = bottom[0]->count();
  caffe_set(count, Dtype(0.), bottom_diff);
  for (int n = 0; n < num_; n++) {
      for (int kpsg_id = 0; kpsg_id < num_points_; kpsg_id ++) {
          const int kpsg_x = round(bottom_kpsg[2*kpsg_id]*kpsg_scale_);
          const int kpsg_y = round(bottom_kpsg[2*kpsg_id+1]*kpsg_scale_);
          int hstart = kpsg_y - kpsg_height_ / 2;
          int hend = hstart + kpsg_height_ - 1;
          int wstart = kpsg_x - kpsg_width_ / 2;
          int wend = wstart + kpsg_width_ - 1;
          if (hstart < 0) {
            hstart = 0;
            hend = hstart + kpsg_height_ - 1;
          }
          else if (hend > height_ - 1) {
            hend = height_ - 1;
            hstart = hend - (kpsg_height_ - 1);
          }
          if (wstart < 0) {
            wstart = 0;
            wend = wstart + kpsg_width_ - 1;
          }
          else if (wend > width_ - 1) {
            wend = width_ - 1;
            wstart = wend - (kpsg_width_ - 1);
          }
          CHECK_GE(wstart, 0);
          CHECK_LT(wend, width_);
          CHECK_GE(hstart, 0);
          CHECK_LT(hend, height_);
          CHECK_EQ(wend - wstart, kpsg_width_ - 1);
          CHECK_EQ(hend - hstart, kpsg_height_ - 1);
          for (int c = 0; c < channels_; c++) {
              for (int h = hstart; h <= hend; h++) {
                  for (int w = wstart; w <= wend; w++) {
                    const int top_index = (h - hstart) * kpsg_width_ + w - wstart;
                    const int bottom_index = h * width_ + w;
                    bottom_diff[bottom_index] += top_diff[top_index];
                  }
              }
              top_diff += top[0]->offset(0, 1);
              bottom_diff += bottom[0]->offset(0, 1);
          }
          bottom_diff -= bottom[0]->offset(0, channels_);
      }
      bottom_kpsg += bottom[1]->offset(1);
  }
}


#ifdef CPU_ONLY
STUB_GPU(KpsgLayer);
#endif

INSTANTIATE_CLASS(KpsgLayer);
REGISTER_LAYER_CLASS(Kpsg);

}  // namespace caffe
