#include <torch/extension.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "aclnn_as_py_if_backward.h"
#include "aclnn_as_py_if_forward.h"
#include "aclnn_as_py_lif_backward.h"
#include "aclnn_as_py_lif_forward.h"
#include "aclnn_as_py_plif_backward.h"
#include "aclnn_as_py_plif_forward.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace {

class AclTensorHandle {
 public:
  explicit AclTensorHandle(const at::Tensor& tensor) {
    std::vector<int64_t> dimensions(
        tensor.sizes().begin(), tensor.sizes().end());
    std::vector<int64_t> strides(
        tensor.strides().begin(), tensor.strides().end());
    tensor_ = aclCreateTensor(
        dimensions.data(),
        dimensions.size(),
        ACL_FLOAT,
        strides.data(),
        0,
        ACL_FORMAT_ND,
        dimensions.data(),
        dimensions.size(),
        tensor.data_ptr());
    TORCH_CHECK(tensor_ != nullptr, "aclCreateTensor failed");
  }

  AclTensorHandle(const AclTensorHandle&) = delete;
  AclTensorHandle& operator=(const AclTensorHandle&) = delete;

  ~AclTensorHandle() {
    if (tensor_ != nullptr) {
      aclDestroyTensor(tensor_);
    }
  }

  const aclTensor* get() const { return tensor_; }

 private:
  aclTensor* tensor_ = nullptr;
};

void check_aclnn(aclnnStatus status, const char* operation) {
  TORCH_CHECK(status == 0, operation, " failed with ACLNN status ", status);
}

void check_tensor(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(
      tensor.device().type() == c10::DeviceType::PrivateUse1,
      name,
      " must be an NPU tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be FP32");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.storage_offset() == 0, name, " must have storage offset zero");
}

void check_sequence(const at::Tensor& tensor, const char* name) {
  check_tensor(tensor, name);
  TORCH_CHECK(tensor.dim() >= 2, name, " must be [T, N, ...]");
  TORCH_CHECK(tensor.size(0) > 0, name, " time dimension must be non-empty");
  TORCH_CHECK(
      tensor.numel() / tensor.size(0) % 8 == 0,
      name,
      " flattened time-step size must be a multiple of 8");
}

at::Tensor make_workspace(const at::Tensor& reference, uint64_t size) {
  if (size == 0) {
    return {};
  }
  return at::empty(
      {static_cast<int64_t>(size)},
      reference.options().dtype(at::kByte));
}

}  // namespace

std::vector<at::Tensor> if_forward(
    const at::Tensor& x_seq,
    const at::Tensor& v_init,
    double v_threshold,
    double v_reset,
    bool hard_reset) {
  check_sequence(x_seq, "x_seq");
  check_tensor(v_init, "v_init");
  TORCH_CHECK(v_init.device() == x_seq.device(), "v_init must be on the same NPU");
  TORCH_CHECK(
      v_init.sizes().equals(x_seq.sizes().slice(1)),
      "v_init shape mismatch");

  c10_npu::NPUGuard device_guard(x_seq.device());
  at::Tensor spike_seq = at::empty_like(x_seq);
  at::Tensor v_seq = at::empty_like(x_seq);
  at::Tensor v_final = at::empty_like(v_init);
  at::Tensor h_seq = at::empty_like(x_seq);

  auto x_acl = std::make_shared<AclTensorHandle>(x_seq);
  auto v_init_acl = std::make_shared<AclTensorHandle>(v_init);
  auto spike_acl = std::make_shared<AclTensorHandle>(spike_seq);
  auto v_seq_acl = std::make_shared<AclTensorHandle>(v_seq);
  auto v_final_acl = std::make_shared<AclTensorHandle>(v_final);
  auto h_seq_acl = std::make_shared<AclTensorHandle>(h_seq);

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_aclnn(
      aclnnAsPyIfForwardGetWorkspaceSize(
          x_acl->get(),
          v_init_acl->get(),
          v_threshold,
          v_reset,
          hard_reset,
          spike_acl->get(),
          v_seq_acl->get(),
          v_final_acl->get(),
          h_seq_acl->get(),
          &workspace_size,
          &executor),
      "aclnnAsPyIfForwardGetWorkspaceSize");
  TORCH_CHECK(executor != nullptr, "ACLNN returned a null forward executor");

  const auto stream =
      c10_npu::getCurrentNPUStream(x_seq.device().index()).stream(false);
  at::Tensor workspace = make_workspace(x_seq, workspace_size);
  at_npu::native::OpCommand::RunOpApiV2(
      "aclnnAsPyIfForward",
      [workspace,
       workspace_size,
       executor,
       stream,
       x_acl,
       v_init_acl,
       spike_acl,
       v_seq_acl,
       v_final_acl,
       h_seq_acl]() mutable -> int {
        void* workspace_pointer =
            workspace.defined() ? workspace.data_ptr() : nullptr;
        return aclnnAsPyIfForward(
            workspace_pointer, workspace_size, executor, stream);
      });
  return {spike_seq, v_seq, v_final, h_seq};
}

std::vector<at::Tensor> if_backward(
    const at::Tensor& h_seq,
    const at::Tensor& spike_seq,
    const at::Tensor& grad_spike_seq,
    const at::Tensor& grad_v_seq,
    const at::Tensor& grad_v_final,
    double v_threshold,
    double v_reset,
    bool hard_reset,
    bool detach_reset,
    double surrogate_alpha) {
  check_sequence(h_seq, "h_seq");
  check_sequence(spike_seq, "spike_seq");
  check_sequence(grad_spike_seq, "grad_spike_seq");
  check_sequence(grad_v_seq, "grad_v_seq");
  check_tensor(grad_v_final, "grad_v_final");
  TORCH_CHECK(spike_seq.sizes().equals(h_seq.sizes()), "spike_seq shape mismatch");
  TORCH_CHECK(
      grad_spike_seq.sizes().equals(h_seq.sizes()),
      "grad_spike_seq shape mismatch");
  TORCH_CHECK(grad_v_seq.sizes().equals(h_seq.sizes()), "grad_v_seq shape mismatch");
  TORCH_CHECK(
      grad_v_final.sizes().equals(h_seq.sizes().slice(1)),
      "grad_v_final shape mismatch");
  for (const at::Tensor* tensor :
       {&spike_seq, &grad_spike_seq, &grad_v_seq, &grad_v_final}) {
    TORCH_CHECK(tensor->device() == h_seq.device(), "backward tensor device mismatch");
  }

  c10_npu::NPUGuard device_guard(h_seq.device());
  at::Tensor grad_x_seq = at::empty_like(h_seq);
  at::Tensor grad_v_init = at::empty_like(grad_v_final);

  auto h_acl = std::make_shared<AclTensorHandle>(h_seq);
  auto spike_acl = std::make_shared<AclTensorHandle>(spike_seq);
  auto grad_spike_acl = std::make_shared<AclTensorHandle>(grad_spike_seq);
  auto grad_v_seq_acl = std::make_shared<AclTensorHandle>(grad_v_seq);
  auto grad_v_final_acl = std::make_shared<AclTensorHandle>(grad_v_final);
  auto grad_x_acl = std::make_shared<AclTensorHandle>(grad_x_seq);
  auto grad_v_init_acl = std::make_shared<AclTensorHandle>(grad_v_init);

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_aclnn(
      aclnnAsPyIfBackwardGetWorkspaceSize(
          h_acl->get(),
          spike_acl->get(),
          grad_spike_acl->get(),
          grad_v_seq_acl->get(),
          grad_v_final_acl->get(),
          v_threshold,
          v_reset,
          hard_reset,
          detach_reset,
          surrogate_alpha,
          grad_x_acl->get(),
          grad_v_init_acl->get(),
          &workspace_size,
          &executor),
      "aclnnAsPyIfBackwardGetWorkspaceSize");
  TORCH_CHECK(executor != nullptr, "ACLNN returned a null backward executor");

  const auto stream =
      c10_npu::getCurrentNPUStream(h_seq.device().index()).stream(false);
  at::Tensor workspace = make_workspace(h_seq, workspace_size);
  at_npu::native::OpCommand::RunOpApiV2(
      "aclnnAsPyIfBackward",
      [workspace,
       workspace_size,
       executor,
       stream,
       h_acl,
       spike_acl,
       grad_spike_acl,
       grad_v_seq_acl,
       grad_v_final_acl,
       grad_x_acl,
       grad_v_init_acl]() mutable -> int {
        void* workspace_pointer =
            workspace.defined() ? workspace.data_ptr() : nullptr;
        return aclnnAsPyIfBackward(
            workspace_pointer, workspace_size, executor, stream);
      });
  return {grad_x_seq, grad_v_init};
}

std::vector<at::Tensor> lif_forward(
    const at::Tensor& x_seq,
    const at::Tensor& v_init,
    double v_threshold,
    double v_reset,
    bool hard_reset,
    double tau,
    bool decay_input) {
  check_sequence(x_seq, "x_seq");
  check_tensor(v_init, "v_init");
  TORCH_CHECK(tau > 1.0, "tau must be greater than 1");
  TORCH_CHECK(v_init.device() == x_seq.device(), "v_init must be on the same NPU");
  TORCH_CHECK(
      v_init.sizes().equals(x_seq.sizes().slice(1)),
      "v_init shape mismatch");

  c10_npu::NPUGuard device_guard(x_seq.device());
  at::Tensor spike_seq = at::empty_like(x_seq);
  at::Tensor v_seq = at::empty_like(x_seq);
  at::Tensor v_final = at::empty_like(v_init);
  at::Tensor h_seq = at::empty_like(x_seq);

  auto x_acl = std::make_shared<AclTensorHandle>(x_seq);
  auto v_init_acl = std::make_shared<AclTensorHandle>(v_init);
  auto spike_acl = std::make_shared<AclTensorHandle>(spike_seq);
  auto v_seq_acl = std::make_shared<AclTensorHandle>(v_seq);
  auto v_final_acl = std::make_shared<AclTensorHandle>(v_final);
  auto h_seq_acl = std::make_shared<AclTensorHandle>(h_seq);

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_aclnn(
      aclnnAsPyLifForwardGetWorkspaceSize(
          x_acl->get(),
          v_init_acl->get(),
          v_threshold,
          v_reset,
          hard_reset,
          tau,
          decay_input,
          spike_acl->get(),
          v_seq_acl->get(),
          v_final_acl->get(),
          h_seq_acl->get(),
          &workspace_size,
          &executor),
      "aclnnAsPyLifForwardGetWorkspaceSize");
  TORCH_CHECK(executor != nullptr, "ACLNN returned a null LIF forward executor");

  const auto stream =
      c10_npu::getCurrentNPUStream(x_seq.device().index()).stream(false);
  at::Tensor workspace = make_workspace(x_seq, workspace_size);
  at_npu::native::OpCommand::RunOpApiV2(
      "aclnnAsPyLifForward",
      [workspace,
       workspace_size,
       executor,
       stream,
       x_acl,
       v_init_acl,
       spike_acl,
       v_seq_acl,
       v_final_acl,
       h_seq_acl]() mutable -> int {
        void* workspace_pointer =
            workspace.defined() ? workspace.data_ptr() : nullptr;
        return aclnnAsPyLifForward(
            workspace_pointer, workspace_size, executor, stream);
      });
  return {spike_seq, v_seq, v_final, h_seq};
}

std::vector<at::Tensor> lif_backward(
    const at::Tensor& h_seq,
    const at::Tensor& spike_seq,
    const at::Tensor& grad_spike_seq,
    const at::Tensor& grad_v_seq,
    const at::Tensor& grad_v_final,
    double v_threshold,
    double v_reset,
    bool hard_reset,
    bool detach_reset,
    double surrogate_alpha,
    double tau,
    bool decay_input) {
  check_sequence(h_seq, "h_seq");
  check_sequence(spike_seq, "spike_seq");
  check_sequence(grad_spike_seq, "grad_spike_seq");
  check_sequence(grad_v_seq, "grad_v_seq");
  check_tensor(grad_v_final, "grad_v_final");
  TORCH_CHECK(tau > 1.0, "tau must be greater than 1");
  TORCH_CHECK(spike_seq.sizes().equals(h_seq.sizes()), "spike_seq shape mismatch");
  TORCH_CHECK(
      grad_spike_seq.sizes().equals(h_seq.sizes()),
      "grad_spike_seq shape mismatch");
  TORCH_CHECK(grad_v_seq.sizes().equals(h_seq.sizes()), "grad_v_seq shape mismatch");
  TORCH_CHECK(
      grad_v_final.sizes().equals(h_seq.sizes().slice(1)),
      "grad_v_final shape mismatch");
  for (const at::Tensor* tensor :
       {&spike_seq, &grad_spike_seq, &grad_v_seq, &grad_v_final}) {
    TORCH_CHECK(tensor->device() == h_seq.device(), "backward tensor device mismatch");
  }

  c10_npu::NPUGuard device_guard(h_seq.device());
  at::Tensor grad_x_seq = at::empty_like(h_seq);
  at::Tensor grad_v_init = at::empty_like(grad_v_final);

  auto h_acl = std::make_shared<AclTensorHandle>(h_seq);
  auto spike_acl = std::make_shared<AclTensorHandle>(spike_seq);
  auto grad_spike_acl = std::make_shared<AclTensorHandle>(grad_spike_seq);
  auto grad_v_seq_acl = std::make_shared<AclTensorHandle>(grad_v_seq);
  auto grad_v_final_acl = std::make_shared<AclTensorHandle>(grad_v_final);
  auto grad_x_acl = std::make_shared<AclTensorHandle>(grad_x_seq);
  auto grad_v_init_acl = std::make_shared<AclTensorHandle>(grad_v_init);

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_aclnn(
      aclnnAsPyLifBackwardGetWorkspaceSize(
          h_acl->get(),
          spike_acl->get(),
          grad_spike_acl->get(),
          grad_v_seq_acl->get(),
          grad_v_final_acl->get(),
          v_threshold,
          v_reset,
          hard_reset,
          detach_reset,
          surrogate_alpha,
          tau,
          decay_input,
          grad_x_acl->get(),
          grad_v_init_acl->get(),
          &workspace_size,
          &executor),
      "aclnnAsPyLifBackwardGetWorkspaceSize");
  TORCH_CHECK(executor != nullptr, "ACLNN returned a null LIF backward executor");

  const auto stream =
      c10_npu::getCurrentNPUStream(h_seq.device().index()).stream(false);
  at::Tensor workspace = make_workspace(h_seq, workspace_size);
  at_npu::native::OpCommand::RunOpApiV2(
      "aclnnAsPyLifBackward",
      [workspace,
       workspace_size,
       executor,
       stream,
       h_acl,
       spike_acl,
       grad_spike_acl,
       grad_v_seq_acl,
       grad_v_final_acl,
       grad_x_acl,
       grad_v_init_acl]() mutable -> int {
        void* workspace_pointer =
            workspace.defined() ? workspace.data_ptr() : nullptr;
        return aclnnAsPyLifBackward(
            workspace_pointer, workspace_size, executor, stream);
      });
  return {grad_x_seq, grad_v_init};
}

std::vector<at::Tensor> plif_forward(
    const at::Tensor& x_seq,
    const at::Tensor& v_init,
    const at::Tensor& reciprocal_tau,
    double v_threshold,
    double v_reset,
    bool hard_reset,
    bool decay_input) {
  check_sequence(x_seq, "x_seq");
  check_tensor(v_init, "v_init");
  check_tensor(reciprocal_tau, "reciprocal_tau");
  TORCH_CHECK(reciprocal_tau.numel() == 1, "reciprocal_tau must be scalar");
  TORCH_CHECK(v_init.device() == x_seq.device(), "v_init must be on the same NPU");
  TORCH_CHECK(
      reciprocal_tau.device() == x_seq.device(),
      "reciprocal_tau must be on the same NPU");
  TORCH_CHECK(
      v_init.sizes().equals(x_seq.sizes().slice(1)),
      "v_init shape mismatch");

  c10_npu::NPUGuard device_guard(x_seq.device());
  at::Tensor spike_seq = at::empty_like(x_seq);
  at::Tensor v_seq = at::empty_like(x_seq);
  at::Tensor v_final = at::empty_like(v_init);
  at::Tensor h_seq = at::empty_like(x_seq);
  at::Tensor v_prev_seq = at::empty_like(x_seq);

  auto x_acl = std::make_shared<AclTensorHandle>(x_seq);
  auto v_init_acl = std::make_shared<AclTensorHandle>(v_init);
  auto reciprocal_tau_acl = std::make_shared<AclTensorHandle>(reciprocal_tau);
  auto spike_acl = std::make_shared<AclTensorHandle>(spike_seq);
  auto v_seq_acl = std::make_shared<AclTensorHandle>(v_seq);
  auto v_final_acl = std::make_shared<AclTensorHandle>(v_final);
  auto h_seq_acl = std::make_shared<AclTensorHandle>(h_seq);
  auto v_prev_seq_acl = std::make_shared<AclTensorHandle>(v_prev_seq);

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_aclnn(
      aclnnAsPyPlifForwardGetWorkspaceSize(
          x_acl->get(),
          v_init_acl->get(),
          reciprocal_tau_acl->get(),
          v_threshold,
          v_reset,
          hard_reset,
          decay_input,
          spike_acl->get(),
          v_seq_acl->get(),
          v_final_acl->get(),
          h_seq_acl->get(),
          v_prev_seq_acl->get(),
          &workspace_size,
          &executor),
      "aclnnAsPyPlifForwardGetWorkspaceSize");
  TORCH_CHECK(executor != nullptr, "ACLNN returned a null PLIF forward executor");

  const auto stream =
      c10_npu::getCurrentNPUStream(x_seq.device().index()).stream(false);
  at::Tensor workspace = make_workspace(x_seq, workspace_size);
  at_npu::native::OpCommand::RunOpApiV2(
      "aclnnAsPyPlifForward",
      [workspace,
       workspace_size,
       executor,
       stream,
       x_acl,
       v_init_acl,
       reciprocal_tau_acl,
       spike_acl,
       v_seq_acl,
       v_final_acl,
       h_seq_acl,
       v_prev_seq_acl]() mutable -> int {
        void* workspace_pointer =
            workspace.defined() ? workspace.data_ptr() : nullptr;
        return aclnnAsPyPlifForward(
            workspace_pointer, workspace_size, executor, stream);
      });
  return {spike_seq, v_seq, v_final, h_seq, v_prev_seq};
}

std::vector<at::Tensor> plif_backward(
    const at::Tensor& x_seq,
    const at::Tensor& v_prev_seq,
    const at::Tensor& h_seq,
    const at::Tensor& spike_seq,
    const at::Tensor& grad_spike_seq,
    const at::Tensor& grad_v_seq,
    const at::Tensor& grad_v_final,
    const at::Tensor& reciprocal_tau,
    double v_threshold,
    double v_reset,
    bool hard_reset,
    bool detach_reset,
    double surrogate_alpha,
    bool decay_input) {
  check_sequence(x_seq, "x_seq");
  check_sequence(v_prev_seq, "v_prev_seq");
  check_sequence(h_seq, "h_seq");
  check_sequence(spike_seq, "spike_seq");
  check_sequence(grad_spike_seq, "grad_spike_seq");
  check_sequence(grad_v_seq, "grad_v_seq");
  check_tensor(grad_v_final, "grad_v_final");
  check_tensor(reciprocal_tau, "reciprocal_tau");
  TORCH_CHECK(reciprocal_tau.numel() == 1, "reciprocal_tau must be scalar");
  for (const at::Tensor* tensor :
       {&v_prev_seq, &h_seq, &spike_seq, &grad_spike_seq, &grad_v_seq}) {
    TORCH_CHECK(tensor->sizes().equals(x_seq.sizes()), "PLIF sequence shape mismatch");
  }
  TORCH_CHECK(
      grad_v_final.sizes().equals(x_seq.sizes().slice(1)),
      "grad_v_final shape mismatch");
  for (const at::Tensor* tensor :
       {&v_prev_seq, &h_seq, &spike_seq, &grad_spike_seq, &grad_v_seq,
        &grad_v_final, &reciprocal_tau}) {
    TORCH_CHECK(tensor->device() == x_seq.device(), "PLIF tensor device mismatch");
  }

  c10_npu::NPUGuard device_guard(x_seq.device());
  at::Tensor grad_x_seq = at::empty_like(x_seq);
  at::Tensor grad_v_init = at::empty_like(grad_v_final);
  at::Tensor grad_reciprocal_tau_partial = at::empty_like(grad_v_final);

  auto x_acl = std::make_shared<AclTensorHandle>(x_seq);
  auto v_prev_acl = std::make_shared<AclTensorHandle>(v_prev_seq);
  auto h_acl = std::make_shared<AclTensorHandle>(h_seq);
  auto spike_acl = std::make_shared<AclTensorHandle>(spike_seq);
  auto grad_spike_acl = std::make_shared<AclTensorHandle>(grad_spike_seq);
  auto grad_v_seq_acl = std::make_shared<AclTensorHandle>(grad_v_seq);
  auto grad_v_final_acl = std::make_shared<AclTensorHandle>(grad_v_final);
  auto reciprocal_tau_acl = std::make_shared<AclTensorHandle>(reciprocal_tau);
  auto grad_x_acl = std::make_shared<AclTensorHandle>(grad_x_seq);
  auto grad_v_init_acl = std::make_shared<AclTensorHandle>(grad_v_init);
  auto grad_reciprocal_tau_partial_acl =
      std::make_shared<AclTensorHandle>(grad_reciprocal_tau_partial);

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_aclnn(
      aclnnAsPyPlifBackwardGetWorkspaceSize(
          x_acl->get(),
          v_prev_acl->get(),
          h_acl->get(),
          spike_acl->get(),
          grad_spike_acl->get(),
          grad_v_seq_acl->get(),
          grad_v_final_acl->get(),
          reciprocal_tau_acl->get(),
          v_threshold,
          v_reset,
          hard_reset,
          detach_reset,
          surrogate_alpha,
          decay_input,
          grad_x_acl->get(),
          grad_v_init_acl->get(),
          grad_reciprocal_tau_partial_acl->get(),
          &workspace_size,
          &executor),
      "aclnnAsPyPlifBackwardGetWorkspaceSize");
  TORCH_CHECK(executor != nullptr, "ACLNN returned a null PLIF backward executor");

  const auto stream =
      c10_npu::getCurrentNPUStream(x_seq.device().index()).stream(false);
  at::Tensor workspace = make_workspace(x_seq, workspace_size);
  at_npu::native::OpCommand::RunOpApiV2(
      "aclnnAsPyPlifBackward",
      [workspace,
       workspace_size,
       executor,
       stream,
       x_acl,
       v_prev_acl,
       h_acl,
       spike_acl,
       grad_spike_acl,
       grad_v_seq_acl,
       grad_v_final_acl,
       reciprocal_tau_acl,
       grad_x_acl,
       grad_v_init_acl,
       grad_reciprocal_tau_partial_acl]() mutable -> int {
        void* workspace_pointer =
            workspace.defined() ? workspace.data_ptr() : nullptr;
        return aclnnAsPyPlifBackward(
            workspace_pointer, workspace_size, executor, stream);
      });
  return {grad_x_seq, grad_v_init, grad_reciprocal_tau_partial};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("if_forward", &if_forward, "AsPy IF forward (NPU)");
  module.def("if_backward", &if_backward, "AsPy IF backward (NPU)");
  module.def("lif_forward", &lif_forward, "AsPy LIF forward (NPU)");
  module.def("lif_backward", &lif_backward, "AsPy LIF backward (NPU)");
  module.def("plif_forward", &plif_forward, "AsPy PLIF forward (NPU)");
  module.def("plif_backward", &plif_backward, "AsPy PLIF backward (NPU)");
}
