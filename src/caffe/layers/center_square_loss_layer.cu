/*#include <vector>*/

/*#include "caffe/filler.hpp"*/
/*#include "caffe/layers/center_loss_layer.hpp"*/
/*#include "caffe/util/math_functions.hpp"*/

/*namespace caffe {*/

/*template <typename Dtype>*/
/*__global__ void Compute_distance_cy_data_gpu(int nthreads, const int K, const Dtype* bottom,*/
		  /*const Dtype* label, const Dtype* center, Dtype* distance, Dtype* cy) {*/
  /*CUDA_KERNEL_LOOP(index, nthreads) {*/
    /*int m = index / K;*/
    /*int k = index % K;*/
    /*const int label_value = static_cast<int>(label[m]);*/
    /*// distance(i) = x(i) - c_{y(i)}*/
    /*distance[index] = bottom[index] - center[label_value * K + k];*/
    /*// c_y(i) = c_{y(i)}*/
    /*cy[index] = center[label_value * K + k];*/
  /*}*/
/*}*/

/*template <typename Dtype>*/
/*__global__ void Compute_lij_data_gpu(int nthreads, const int M, const Dtype* label,*/
              /*Dtype* lij) {*/
  /*CUDA_KERNEL_LOOP(index, nthreads) {*/
    /*int i = index / M;*/
    /*int j = index % M;*/
    /*const int label_value_i = static_cast<int>(label[i]);*/
    /*const int label_value_j = static_cast<int>(label[j]);*/
    /*if (label_value_i == label_value_j) {*/
      /*lij[index] = 1;*/
    /*}*/
    /*else {*/
      /*lij[index] = -1;*/
    /*}*/
  /*}*/
/*}*/

/*template <typename Dtype>*/
/*__global__ void Compute_mat_data_gpu(int nthreads, const Dtype* pa,*/
              /*Dtype* mat) {*/
  /*CUDA_KERNEL_LOOP(index, nthreads) {*/
    /*if (pa[index] > 0) {*/
      /*mat[index] = 1;*/
    /*}*/
    /*else {*/
      /*mat[index] = 0;*/
    /*}*/
  /*}*/
/*}*/

/*template <typename Dtype>*/
/*__global__ void Compute_center_diff_gpu(int nthreads, const int M, const int K, */
        /*const Dtype* label, const Dtype* distance, Dtype* variation_sum, */
        /*Dtype* center_diff) {*/
  /*CUDA_KERNEL_LOOP(index, nthreads) {*/
    /*int count = 0;*/
    /*for (int m = 0; m < M; m++) {*/
      /*const int label_value = static_cast<int>(label[m]);*/
      /*if (label_value == index) {*/
        /*count++;*/
        /*for (int k = 0; k < K; k++) {*/
          /*variation_sum[index * K + k] -= distance[m * K + k];*/
        /*}*/
      /*}*/
    /*}*/
    /*for (int k = 0; k < K; k++) {*/
      /*center_diff[index * K + k] = variation_sum[index * K + k] /(count + (Dtype)1.);*/
    /*}*/
  /*}*/
/*}*/


/*template <typename Dtype>*/
/*void CenterLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,*/
    /*const vector<Blob<Dtype>*>& top) {*/
  /*int nthreads = M_ * K_;*/
  /*Blob<Dtype> cy;*/
  /*cy.ReshapeLike(*bottom[0]);*/
  /*Compute_distance_cy_data_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),*/
      /*CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, bottom[0]->gpu_data(), bottom[1]->gpu_data(),*/
                                /*this->blobs_[0]->gpu_data(), distance_.mutable_gpu_data(), cy.mutable_gpu_data());*/
  /*nthreads = M_ * M_; */
  /*Compute_lij_data_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),*/
      /*CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, bottom[1]->gpu_data(),*/
                                /*lij_.mutable_gpu_data());*/
  /*Blob<Dtype> xi, cyj;*/
  /*xi.ReshapeLike(distance_mat_);*/
  /*cyj.ReshapeLike(distance_mat_);*/
  /*// cyj = [cy; cy; ...; cy]*/
  /*caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, M_ * K_, 1, 1, Ic_.gpu_data(), cy.gpu_data(), 0., cyj.mutable_gpu_data());*/
  /*// xi = [x0; x0; ...; x0; x1; x1; ...; x1; x(M - 1); x(M - 1); ...; x(M - 1)]*/
  /*caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, M_ * K_, K_, 1, bottom[0]->gpu_data(), If_.gpu_data(), 0., xi.mutable_gpu_data());*/
  /*// xi - cyj*/
  /*caffe_gpu_sub(distance_mat_.count(), xi.gpu_data(), cyj.gpu_data(), distance_mat_.mutable_gpu_data());*/
  /*Blob<Dtype> distance_square;*/
  /*distance_square.ReshapeLike(distance_mat_);*/
  /*// (xi - cyj) .* (xi - cyj)*/
  /*caffe_gpu_powx(distance_mat_.count(), distance_mat_.gpu_data(), Dtype(2), distance_square.mutable_gpu_data());*/
  /*Blob<Dtype> pa;*/
  /*pa.ReshapeLike(mat_);*/
  /*// (xi - cyj)T(xi - cyj)*/
  /*caffe_gpu_gemv<Dtype>(CblasNoTrans, M_ * M_, K_, 1., distance_square.gpu_data(), Im_.gpu_data(), 0., distance_norm_mat_.mutable_gpu_data());*/
  /*// ||xi - cyj||*/
  /*caffe_gpu_powx(mat_.count(), distance_norm_mat_.gpu_data(), Dtype(0.5), distance_norm_mat_.mutable_gpu_data());*/
  /*// ||xi - cyj|| - 1*/
  /*caffe_gpu_sub(mat_.count(), distance_norm_mat_.gpu_data(), Ia_.gpu_data(), pa.mutable_gpu_data());*/
  /*// lij(||xi - cyj|| - 1)*/
  /*caffe_gpu_mul(mat_.count(), lij_.gpu_data(), pa.gpu_data(), pa.mutable_gpu_data());*/
  /*// lij(||xi - cyj|| - 1) + alpha*/
  /*caffe_gpu_axpy(mat_.count(), margin_, Ia_.gpu_data(), pa.mutable_gpu_data());*/

  /*Compute_mat_data_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),*/
      /*CAFFE_CUDA_NUM_THREADS>>>(nthreads, pa.gpu_data(),*/
                                /*mat_.mutable_gpu_data());*/
  /*Dtype dot;*/
  /*// caffe_gpu_dot(M_ * K_, distance_.gpu_data(), distance_.gpu_data(), &dot);*/
  /*caffe_gpu_dot(M_ * M_, mat_.gpu_data(), pa.gpu_data(), &dot);*/
  /*Dtype loss = dot / M_ / Dtype(2);*/
  /*top[0]->mutable_gpu_data()[0] = loss;*/
/*}*/

/*template <typename Dtype>*/
/*void CenterLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,*/
    /*const vector<bool>& propagate_down,*/
    /*const vector<Blob<Dtype>*>& bottom) {*/
  /*int nthreads = N_;*/
  /*caffe_gpu_set(N_ * K_, (Dtype)0., variation_sum_.mutable_cpu_data());*/
  /*Compute_center_diff_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),*/
      /*CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, K_, bottom[1]->gpu_data(), distance_.gpu_data(), */
                                /*variation_sum_.mutable_cpu_data(), this->blobs_[0]->mutable_gpu_diff());*/

  /*if (propagate_down[0]) {*/
    /*// caffe_gpu_scale(M_ * K_, top[0]->cpu_diff()[0] / M_, */
                             /*// distance_.gpu_data(), bottom[0]->mutable_gpu_diff());*/
    /*Blob<Dtype> scale;*/
    /*scale.ReshapeLike(mat_);*/
    /*// scale: lij / ||xi - cyj||*/
    /*caffe_gpu_div(mat_.count(), lij_.gpu_data(), distance_norm_mat_.gpu_data(), scale.mutable_gpu_data());*/
    /*// caffe_copy(mat_.count(), lij_.cpu_data(), scale.mutable_cpu_data());*/
    /*// Iij * lij / ||xi - cyj||*/
    /*caffe_gpu_mul(mat_.count(), mat_.gpu_data(), scale.gpu_data(), scale.mutable_gpu_data());*/
    /*Blob<Dtype> matI;*/
    /*matI.ReshapeLike(distance_mat_);*/
    /*// [mat_ * scale, mat_ * scale, ..., mat_ * scale]*/
    /*caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_ * M_, K_, 1, 1, scale.gpu_data(), Im_.gpu_data(), 0., matI.mutable_gpu_data());*/
    /*// (xi - cyj) .* [mat_ * scale, mat_ * scale, ..., mat_ * scale]*/
    /*caffe_gpu_mul(distance_mat_.count(), distance_mat_.gpu_data(), matI.gpu_data(), matI.mutable_gpu_data());*/
    /*caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, M_ * M_, 1, Ib_.gpu_data(), matI.gpu_data(), 0., bottom[0]->mutable_gpu_diff());*/
    /*caffe_gpu_scal(M_ * K_, top[0]->gpu_diff()[0] / (2 * M_), bottom[0]->mutable_gpu_diff());*/
  /*}*/
  /*if (propagate_down[1]) {*/
    /*LOG(FATAL) << this->type()*/
               /*<< " Layer cannot backpropagate to label inputs.";*/
  /*}*/
/*}*/

/*INSTANTIATE_LAYER_GPU_FUNCS(CenterLossLayer);*/

/*}  // namespace caffe*/

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/center_square_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CenterSquareLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void CenterSquareLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(CenterSquareLossLayer);

}
