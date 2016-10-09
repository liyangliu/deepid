#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/kps_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {


template <typename Dtype>
__global__ void KpsForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height, const int width,
    const Dtype* bottom_kps, const int num_points, const int kps_id, const int kps_height, const int kps_width,
    const Dtype kps_scale, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int kw = index % kps_width;
    const int kh = (index / kps_width) % kps_height;
    const int c = (index / kps_width / kps_height) % channels;
    const int n = index / kps_width / kps_height / channels;
    bottom_kps += n * 2 * num_points;
    const int kps_x = round(bottom_kps[2 * kps_id] * kps_scale);
    const int kps_y = round(bottom_kps[2 * kps_id + 1] * kps_scale);
    int hstart = kps_y - kps_height / 2;
    int hend = hstart + kps_height - 1;
    int wstart = kps_x - kps_width / 2;
    int wend = wstart + kps_width - 1;
    if (hstart < 0) {
        hstart = 0;
        hend = hstart + kps_height - 1;
    }
    else if (hend > height - 1) {
        hend = height - 1;
        hstart = hend - (kps_height - 1);
    }
    if (wstart < 0) {
        wstart = 0;
        wend = wstart + kps_width - 1;
    }
    else if (wend > width - 1) {
        wend = width - 1;
        wstart = wend - (kps_width - 1);
    }
    bottom_data += (n * channels + c) * height * width;
    const int bottom_index = (hstart + kh) * width + wstart + kw;
    top_data[index] = bottom_data[bottom_index];
  }
}


template <typename Dtype>
void KpsLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_kps = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = top[0]->count();
  KpsForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, num_, channels_, height_, width_,
      bottom_kps, num_points_, kps_id_, kps_height_, kps_width_, kps_scale_, top_data);
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void KpsBackward(const int nthreads, const Dtype* top_diff,
    const int num, const int channels, const int height, const int width,
    const Dtype* bottom_kps, const int num_points, const int kps_id, const int kps_height, const int kps_width,
    const Dtype kps_scale, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    bottom_kps += n * 2 * num_points;
    const int kps_x = round(bottom_kps[2 * kps_id] * kps_scale);
    const int kps_y = round(bottom_kps[2 * kps_id + 1] * kps_scale);
    int hstart = kps_y - kps_height / 2;
    int hend = hstart + kps_height - 1;
    int wstart = kps_x - kps_width / 2;
    int wend = wstart + kps_width - 1;
    if (hstart < 0) {
        hstart = 0;
        hend = hstart + kps_height - 1;
    }
    else if (hend > height - 1) {
        hend = height - 1;
        hstart = hend - (kps_height - 1);
    }
    if (wstart < 0) {
        wstart = 0;
        wend = wstart + kps_width - 1;
    }
    else if (wend > width - 1) {
        wend = width - 1;
        wstart = wend - (kps_width - 1);
    }
    if (w >= wstart && w<= wend && h >= hstart && h <= hend) {
        top_diff += (n * channels + c) * kps_height * kps_width;
        const int top_index = (h - hstart) * kps_width + w - wstart;
        bottom_diff[index] = top_diff[top_index];
    }
  }
}


template <typename Dtype>
void KpsLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* bottom_kps = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  KpsBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, num_, channels_, height_, width_,
      bottom_kps, num_points_, kps_id_, kps_height_, kps_width_, kps_scale_, bottom_diff);
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(KpsLayer);


}  // namespace caffe
