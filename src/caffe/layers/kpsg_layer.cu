#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/kpsg_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {


template <typename Dtype>
__global__ void KpsgForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height, const int width,
    const Dtype* bottom_kpsg, const int num_points, const int kpsg_height, const int kpsg_width,
    const Dtype kpsg_scale, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int kw = index % kpsg_width;
    const int kh = (index / kpsg_width) % kpsg_height;
    const int kc = (index / kpsg_width / kpsg_height) % (num_points * channels);
    const int n = index / kpsg_width / kpsg_height / (num_points * channels);
    bottom_kpsg += n * 2 * num_points;
    const int kpsg_id = kc / channels;
    const int c = kc % channels;
    const int kpsg_x = round(bottom_kpsg[2 * kpsg_id] * kpsg_scale);
    const int kpsg_y = round(bottom_kpsg[2 * kpsg_id + 1] * kpsg_scale);
    int hstart = kpsg_y - kpsg_height / 2;
    int hend = hstart + kpsg_height - 1;
    int wstart = kpsg_x - kpsg_width / 2;
    int wend = wstart + kpsg_width - 1;
    if (hstart < 0) {
        hstart = 0;
        hend = hstart + kpsg_height - 1;
    }
    else if (hend > height - 1) {
        hend = height - 1;
        hstart = hend - (kpsg_height - 1);
    }
    if (wstart < 0) {
        wstart = 0;
        wend = wstart + kpsg_width - 1;
    }
    else if (wend > width - 1) {
        wend = width - 1;
        wstart = wend - (kpsg_width - 1);
    }
    bottom_data += (n * channels + c) * height * width;
    const int bottom_index = (hstart + kh) * width + wstart + kw;
    top_data[index] = bottom_data[bottom_index];
  }
}


template <typename Dtype>
void KpsgLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_kpsg = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = top[0]->count();
  KpsgForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, num_, channels_, height_, width_,
      bottom_kpsg, num_points_, kpsg_height_, kpsg_width_, kpsg_scale_, top_data);
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void KpsgBackward(const int nthreads, const Dtype* top_diff,
    const int num, const int channels, const int height, const int width,
    const Dtype* bottom_kpsg, const int num_points, const int kpsg_height, const int kpsg_width,
    const Dtype kpsg_scale, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    bottom_kpsg += n * 2 * num_points;
    top_diff += (n * (num_points * channels) + c) * kpsg_height * kpsg_width;
    for (int kpsg_id = 0; kpsg_id < num_points; kpsg_id ++) {
        const int kpsg_x = round(bottom_kpsg[2 * kpsg_id] * kpsg_scale);
        const int kpsg_y = round(bottom_kpsg[2 * kpsg_id + 1] * kpsg_scale);
        int hstart = kpsg_y - kpsg_height / 2;
        int hend = hstart + kpsg_height - 1;
        int wstart = kpsg_x - kpsg_width / 2;
        int wend = wstart + kpsg_width - 1;
        if (hstart < 0) {
            hstart = 0;
            hend = hstart + kpsg_height - 1;
        }
        else if (hend > height - 1) {
            hend = height - 1;
            hstart = hend - (kpsg_height - 1);
        }
        if (wstart < 0) {
            wstart = 0;
            wend = wstart + kpsg_width - 1;
        }
        else if (wend > width - 1) {
            wend = width - 1;
            wstart = wend - (kpsg_width - 1);
        }
        if (w >= wstart && w<= wend && h >= hstart && h <= hend) {
            const int top_index = (h - hstart) * kpsg_width + w - wstart;
            bottom_diff[index] += top_diff[top_index];
        }
        top_diff += channels * kpsg_height * kpsg_width;
    }
  }
}


template <typename Dtype>
void KpsgLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* bottom_kpsg = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  KpsgBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, num_, channels_, height_, width_,
      bottom_kpsg, num_points_, kpsg_height_, kpsg_width_, kpsg_scale_, bottom_diff);
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(KpsgLayer);


}  // namespace caffe
