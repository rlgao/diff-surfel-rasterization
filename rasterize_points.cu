#include <climits>
/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

#define CHECK_INPUT(x)											\
	AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
	// AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
	auto lambda = [&t](size_t N) {
		t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
	};
	return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansNearTieImpl(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug, int depth_policy, float depth_epsilon,
 const torch::Tensor& trace_pixels, torch::Tensor& trace_ids,
 torch::Tensor& trace_records, torch::Tensor& trace_counts, const torch::Tensor& stable_ids, double relative_span, bool accumulate_weight,
 const std::vector<int64_t>& limits = {}, torch::Tensor* stats_output = nullptr)
{
  TORCH_CHECK(depth_policy == 0 || depth_policy == 1, "invalid depth policy");
  TORCH_CHECK(depth_policy == 0 || (std::isfinite(depth_epsilon) && depth_epsilon > 0),
              "invalid depth conditioning epsilon");
  const int trace_n = trace_pixels.size(0);
  const int trace_capacity = trace_ids.size(1);
  TORCH_CHECK(trace_n <= 8 && trace_capacity <= 1024, "trace bound exceeded");
  TORCH_CHECK(trace_pixels.is_cuda() && trace_pixels.is_contiguous()
              && trace_pixels.scalar_type() == torch::kInt32 && trace_pixels.dim() == 2
              && trace_pixels.size(1) == 2, "trace pixels must be CUDA int32 Nx2");
  TORCH_CHECK(trace_ids.is_cuda() && trace_ids.is_contiguous()
              && trace_ids.scalar_type() == torch::kInt64 && trace_ids.size(0) == trace_n
              && trace_records.is_cuda() && trace_records.is_contiguous()
              && trace_records.scalar_type() == torch::kFloat32
              && trace_records.sizes() == torch::IntArrayRef({trace_n, trace_capacity, 18})
              && trace_counts.is_cuda() && trace_counts.is_contiguous()
              && trace_counts.scalar_type() == torch::kInt32
              && trace_counts.sizes() == torch::IntArrayRef({trace_n, 8}), "invalid trace outputs");
  if (trace_n) {
    TORCH_CHECK(trace_capacity > 0 && trace_pixels.select(1,0).min().item<int>() >= 0
                && trace_pixels.select(1,0).max().item<int>() < image_width
                && trace_pixels.select(1,1).min().item<int>() >= 0
                && trace_pixels.select(1,1).max().item<int>() < image_height,
                "trace pixel outside image");
  }
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
	AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  CHECK_INPUT(background);
  CHECK_INPUT(means3D);
  CHECK_INPUT(colors);
  CHECK_INPUT(opacity);
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(transMat_precomp);
  CHECK_INPUT(viewmatrix);
  CHECK_INPUT(projmatrix);
  CHECK_INPUT(sh);
  CHECK_INPUT(campos);

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_others = torch::full({accumulate_weight ? 8 : 7, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  torch::Tensor workspaceBuffer;
  if (!limits.empty()) workspaceBuffer = torch::empty({0}, options.device(device));
  torch::Tensor statistics;
  if (!limits.empty()) {
    if (stats_output) { stats_output->resize_({8}); stats_output->zero_(); statistics=*stats_output; }
    else statistics = torch::zeros({8}, means3D.options().dtype(torch::kInt64));
  }
  CudaRasterizer::CombinedSettings settings;
  if (!limits.empty()) {
    settings = {limits[3], static_cast<uint64_t>(limits[4]), static_cast<int>(limits[5]),
                static_cast<int>(limits[6]), resizeFunctional(workspaceBuffer), statistics.data_ptr<int64_t>()};
  }
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
	  }

	  rendered = CudaRasterizer::Rasterizer::forward(
		geomFunc,
		binningFunc,
		imgFunc,
		P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		transMat_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_others.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		debug, depth_policy, depth_epsilon, trace_n, trace_capacity,
        trace_pixels.data_ptr<int>(), trace_ids.data_ptr<int64_t>(),
        trace_records.data_ptr<float>(), trace_counts.data_ptr<int>(),
        stable_ids.defined() ? stable_ids.data_ptr<int64_t>() : nullptr, relative_span,
        accumulate_weight ? out_others.data_ptr<float>() + 7*H*W : nullptr,
        limits.empty() ? nullptr : &settings);
  }
  if (stats_output) { stats_output->resize_({8}); stats_output->copy_(statistics); }
  if (accumulate_weight && P == 0) {
    out_color = background.view({3,1,1}).expand({3,H,W}).contiguous();
    if (statistics.defined()) {
      statistics[2].fill_(static_cast<int64_t>((W+15)/16)*((H+15)/16));
      statistics[6].fill_(limits[6]); statistics[7].fill_(1);
    }
  }
  return std::make_tuple(rendered, out_color, out_others, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardImpl(
	 const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_others,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug, int depth_policy, float depth_epsilon, bool combined)
{

  CHECK_INPUT(background);
  CHECK_INPUT(means3D);
  CHECK_INPUT(radii);
  CHECK_INPUT(colors);
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(transMat_precomp);
  CHECK_INPUT(viewmatrix);
  CHECK_INPUT(projmatrix);
  CHECK_INPUT(sh);
  CHECK_INPUT(campos);
  CHECK_INPUT(binningBuffer);
  CHECK_INPUT(imageBuffer);
  CHECK_INPUT(geomBuffer);

  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dnormal = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dtransMat = torch::zeros({P, 9}, means3D.options());
  torch::Tensor accumulated_transform = combined ? torch::zeros({P, 9}, means3D.options().dtype(torch::kFloat64)) : torch::Tensor();
  torch::Tensor accumulated_centre = combined ? torch::zeros({P, 2}, means3D.options().dtype(torch::kFloat64)) : torch::Tensor();
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 2}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  transMat_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_others.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dnormal.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dtransMat.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  debug, depth_policy, depth_epsilon, combined ? accumulated_transform.data_ptr<double>() : nullptr,
      combined ? accumulated_centre.data_ptr<double>() : nullptr);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dtransMat, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}

#include "cuda_rasterizer/rasterizer_impl.h"
#include "cuda_rasterizer/forward.h"
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> TraceDepthCUDA(
 int P,int R,int W,int H,torch::Tensor geom,torch::Tensor binning,torch::Tensor image,
 torch::Tensor matrices,torch::Tensor pixels,int capacity,int policy,float epsilon) {
 TORCH_CHECK(P>0 && R>=0 && W>0 && H>0,"invalid trace buffers");
 TORCH_CHECK(pixels.is_cuda() && pixels.scalar_type()==torch::kInt32 && pixels.dim()==2
             && pixels.size(1)==2 && pixels.is_contiguous(),"trace pixels must be CUDA int32 Nx2");
 int N=pixels.size(0);
 TORCH_CHECK(N>0 && N<=8 && capacity>0 && capacity<=1024,"trace bound exceeded");
 TORCH_CHECK(pixels.select(1,0).min().item<int>()>=0 && pixels.select(1,0).max().item<int>()<W
             && pixels.select(1,1).min().item<int>()>=0 && pixels.select(1,1).max().item<int>()<H,
             "trace pixel outside image");
 char* g=reinterpret_cast<char*>(geom.data_ptr<uint8_t>());char* b=reinterpret_cast<char*>(binning.data_ptr<uint8_t>());char* i=reinterpret_cast<char*>(image.data_ptr<uint8_t>());
 auto gs=CudaRasterizer::GeometryState::fromChunk(g,P);
 auto bs=CudaRasterizer::BinningState::fromChunk(b,R);
 auto is=CudaRasterizer::ImageState::fromChunk(i,W*H);
 auto out_ids=torch::full({N,capacity},-1,pixels.options().dtype(torch::kInt64));
 auto records=torch::zeros({N,capacity,16},pixels.options().dtype(torch::kFloat32));
 auto counts=torch::zeros({N,2},pixels.options());
 FORWARD::trace(N,capacity,pixels.data_ptr<int>(),W,H,is.ranges,bs.point_list,gs.means2D,
   matrices.numel() ? matrices.data_ptr<float>():gs.transMat,gs.normal_opacity,policy,epsilon,
   out_ids.data_ptr<int64_t>(),records.data_ptr<float>(),counts.data_ptr<int>());
 return std::make_tuple(out_ids,records,counts);
}

// Read existing single-Gaussian geometry fields; no kernel or buffer-layout change.
#include "cuda_rasterizer/rasterizer_impl.h"
#include <vector>
std::vector<torch::Tensor> DiagnosticGeometryState(
        const torch::Tensor& geom, const torch::Tensor& radii) {
    TORCH_CHECK(geom.is_cuda() && geom.is_contiguous() &&
        geom.scalar_type() == torch::kByte && geom.dim() == 1, "invalid geometry buffer");
    TORCH_CHECK(radii.is_cuda() && radii.device() == geom.device() && radii.is_contiguous() &&
        radii.scalar_type() == torch::kInt32 && radii.numel() == 1 && radii.item<int>() > 0,
        "geometry readback requires one visible Gaussian");
    TORCH_CHECK(geom.numel() >= CudaRasterizer::required<CudaRasterizer::GeometryState>(1),
        "short geometry buffer");
    char* ptr = reinterpret_cast<char*>(geom.data_ptr<uint8_t>());
    auto state = CudaRasterizer::GeometryState::fromChunk(ptr, 1);
    auto f = geom.options().dtype(torch::kFloat32);
    auto i = geom.options().dtype(torch::kInt32);
    return {torch::from_blob(state.means2D, {1,2}, f).clone(),
        torch::from_blob(state.transMat, {1,3,3}, f).clone(),
        torch::from_blob(state.normal_opacity, {1,4}, f).clone(), radii.clone(),
        torch::from_blob(state.tiles_touched, {1}, i).clone()};
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug, int depth_policy, float depth_epsilon,
 const torch::Tensor& trace_pixels, torch::Tensor& trace_ids,
 torch::Tensor& trace_records, torch::Tensor& trace_counts) {
 return RasterizeGaussiansNearTieImpl(background, means3D, colors, opacity, scales, rotations, scale_modifier, transMat_precomp, viewmatrix, projmatrix, tan_fovx, tan_fovy, image_height, image_width, sh, degree, campos, prefiltered, debug, depth_policy, depth_epsilon, trace_pixels, trace_ids, trace_records, trace_counts, torch::Tensor(), 0.0, false);
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansNearTieCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug, int depth_policy, float depth_epsilon,
 const torch::Tensor& trace_pixels, torch::Tensor& trace_ids,
 torch::Tensor& trace_records, torch::Tensor& trace_counts, const torch::Tensor& stable_ids, double relative_span) {
    TORCH_CHECK(means3D.size(0)>0 && means3D.size(0)<=400 && image_height>0 && image_width>0
                && image_height<=512 && image_width<=512, "near-tie prototype bounds exceeded");
    TORCH_CHECK(stable_ids.is_cuda() && stable_ids.device()==means3D.device()
                && stable_ids.scalar_type()==torch::kInt64 && stable_ids.is_contiguous()
                && stable_ids.dim()==1 && stable_ids.numel()==means3D.size(0), "invalid stable IDs");
    TORCH_CHECK(std::isfinite(relative_span) && relative_span>=0 && relative_span<1,
                "invalid relative near-tie span");
    return RasterizeGaussiansNearTieImpl(background, means3D, colors, opacity, scales, rotations, scale_modifier, transMat_precomp, viewmatrix, projmatrix, tan_fovx, tan_fovy, image_height, image_width, sh, degree, campos, prefiltered, debug, depth_policy, depth_epsilon, trace_pixels, trace_ids, trace_records, trace_counts, stable_ids, relative_span, false);
}

std::vector<torch::Tensor> DiagnosticOrderState(
    int P, int R, int W, int H, const torch::Tensor& geom,
    const torch::Tensor& binning, const torch::Tensor& image, const torch::Tensor& radii) {
    const int tile_count=((W+15)/16)*((H+15)/16);
    TORCH_CHECK(P>0 && W>0 && H>0 && W<=512 && H<=512 && R>=0 && static_cast<int64_t>(R)<=static_cast<int64_t>(P)*tile_count, "invalid registered state dimensions");
    TORCH_CHECK(geom.is_cuda() && binning.is_cuda() && image.is_cuda() && radii.is_cuda(),
                "state buffers must be CUDA");
    char* g=reinterpret_cast<char*>(geom.data_ptr<uint8_t>());
    char* b=reinterpret_cast<char*>(binning.data_ptr<uint8_t>());
    char* i=reinterpret_cast<char*>(image.data_ptr<uint8_t>());
    auto gs=CudaRasterizer::GeometryState::fromChunk(g,P);
    auto bs=CudaRasterizer::BinningState::fromChunk(b,R);
    auto is=CudaRasterizer::ImageState::fromChunk(i,W*H);
    auto f=geom.options().dtype(torch::kFloat32);
    auto u=geom.options().dtype(torch::kInt32);
    auto l=geom.options().dtype(torch::kInt64);
    return {
        torch::from_blob(gs.depths,{P},f).clone(),
        torch::from_blob(gs.transMat,{P,3,3},f).clone(),
        torch::from_blob(gs.means2D,{P,2},f).clone(),
        torch::from_blob(gs.normal_opacity,{P,4},f).clone(), radii.clone(),
        torch::from_blob(gs.tiles_touched,{P},u).clone(),
        torch::from_blob(gs.point_offsets,{P},u).clone(),
        torch::from_blob(bs.point_list_keys,{R},l).clone(),
        torch::from_blob(bs.point_list,{R},u).clone(),
        torch::from_blob(is.ranges,{tile_count,2},u).clone(),
        torch::from_blob(is.accum_alpha,{H,W},f).clone(),
        torch::from_blob(is.n_contrib,{H,W},u).clone()
    };
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansWeightCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug, int depth_policy, float depth_epsilon,
 const torch::Tensor& trace_pixels, torch::Tensor& trace_ids,
 torch::Tensor& trace_records, torch::Tensor& trace_counts, const torch::Tensor& stable_ids, double relative_span, bool grouping_enabled) {
    TORCH_CHECK(means3D.size(0)>0 && means3D.size(0)<=400 && image_height>0 && image_width>0
                && image_height<=512 && image_width<=512, "near-tie prototype bounds exceeded");
    TORCH_CHECK(stable_ids.is_cuda() && stable_ids.device()==means3D.device()
                && stable_ids.scalar_type()==torch::kInt64 && stable_ids.is_contiguous()
                && stable_ids.dim()==1 && stable_ids.numel()==means3D.size(0), "invalid stable IDs");
    TORCH_CHECK(std::isfinite(relative_span) && relative_span>=0 && relative_span<1,
                "invalid relative near-tie span");
    return RasterizeGaussiansNearTieImpl(background, means3D, colors, opacity, scales, rotations, scale_modifier, transMat_precomp, viewmatrix, projmatrix, tan_fovx, tan_fovy, image_height, image_width, sh, degree, campos, prefiltered, debug, depth_policy, depth_epsilon, trace_pixels, trace_ids, trace_records, trace_counts, grouping_enabled ? stable_ids : torch::Tensor(), relative_span, true);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
	 const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_others,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug, int depth_policy, float depth_epsilon) { return RasterizeGaussiansBackwardImpl(background, means3D, radii, colors, scales, rotations, scale_modifier, transMat_precomp, viewmatrix, projmatrix, tan_fovx, tan_fovy, dL_dout_color, dL_dout_others, sh, degree, campos, geomBuffer, R, binningBuffer, imageBuffer, debug, depth_policy, depth_epsilon, false); }

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansCombinedBackwardCUDA(
	 const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_others,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug, int depth_policy, float depth_epsilon) { return RasterizeGaussiansBackwardImpl(background, means3D, radii, colors, scales, rotations, scale_modifier, transMat_precomp, viewmatrix, projmatrix, tan_fovx, tan_fovy, dL_dout_color, dL_dout_others, sh, degree, campos, geomBuffer, R, binningBuffer, imageBuffer, debug, depth_policy, depth_epsilon, true); }

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCombinedCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug, int depth_policy, float depth_epsilon,
 const torch::Tensor& trace_pixels, torch::Tensor& trace_ids,
 torch::Tensor& trace_records, torch::Tensor& trace_counts, const torch::Tensor& stable_ids, double relative_span, const std::vector<int64_t>& limits, torch::Tensor& stats_output) {
 TORCH_CHECK(limits.size()==7 && limits[0]>0 && limits[1]>0 && limits[2]>0
             && limits[3]>0 && limits[3]<=INT_MAX && limits[4]>0 && limits[5]>0
             && (limits[6]==0 || limits[6]==1), "invalid combined renderer limits");
 TORCH_CHECK(means3D.size(0)<=limits[0] && means3D.size(0)<=INT_MAX
             && image_height>0 && image_width>0 && image_height<=limits[1]
             && image_width<=limits[2], "combined population/image admission failed");
 TORCH_CHECK(stable_ids.is_cuda() && stable_ids.device()==means3D.device()
             && stable_ids.scalar_type()==torch::kInt64 && stable_ids.is_contiguous()
             && stable_ids.dim()==1 && stable_ids.numel()==means3D.size(0), "invalid combined stable IDs");
 TORCH_CHECK(std::isfinite(relative_span) && relative_span>=0 && relative_span<1,"invalid combined near-tie span");
 if (stable_ids.numel()) {
   TORCH_CHECK(stable_ids.min().item<int64_t>()>=0,"combined IDs must be nonnegative");
   auto sorted = std::get<0>(stable_ids.sort());
   if (sorted.numel()>1) TORCH_CHECK((sorted.slice(0,1)!=sorted.slice(0,0,-1)).all().item<bool>(),"combined IDs must be unique");
 }
 return RasterizeGaussiansNearTieImpl(background, means3D, colors, opacity, scales, rotations, scale_modifier, transMat_precomp, viewmatrix, projmatrix, tan_fovx, tan_fovy, image_height, image_width, sh, degree, campos, prefiltered, debug, depth_policy, depth_epsilon, trace_pixels, trace_ids, trace_records, trace_counts, stable_ids, relative_span, true, limits, &stats_output);
}
