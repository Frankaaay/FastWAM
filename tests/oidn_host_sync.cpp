// Diagnostic replacement for SAPIEN 3.0.0b1's OIDN synchronization only.
// Based on sapien-vulkan-2 d764f627e784bcd4fc5a6271d211bea6744270b4,
// src/renderer/denoiser_oidn.cpp. Compile against that revision's private
// denoiser.h and the installed wheel's public headers. Not a production switch.
#include "denoiser.h"
#include "svulkan2/core/context.h"
#include <cstdio>
#include <stdexcept>

namespace svulkan2::renderer {

void DenoiserOidn::denoise(core::Image &color, core::Image *albedo,
                           core::Image *normal) {
  if (color.getFormat() != vk::Format::eR32G32B32A32Sfloat ||
      (albedo && albedo->getFormat() != vk::Format::eR32G32B32A32Sfloat) ||
      (normal && normal->getFormat() != vk::Format::eR32G32B32A32Sfloat)) {
    throw std::runtime_error("denoiser only supports R32G32B32A32Sfloat format");
  }
  static unsigned calls = 0;
  bool trace = calls++ < 8;
  if (trace) std::fprintf(stderr, "OIDN_HOST_SYNC input %u\n", calls);
  mCommandBufferIn->reset();
  mCommandBufferIn->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  color.recordCopyToBuffer(mCommandBufferIn.get(), mInputBuffer->getVulkanBuffer(),
                          0, mInputBuffer->getSize(), {0, 0, 0}, color.getExtent());
  if (useAlbedo() && albedo) {
    albedo->recordCopyToBuffer(mCommandBufferIn.get(), mAlbedoBuffer->getVulkanBuffer(),
                             0, mAlbedoBuffer->getSize(), {0, 0, 0}, color.getExtent());
  }
  if (useNormal() && normal) {
    normal->recordCopyToBuffer(mCommandBufferIn.get(), mNormalBuffer->getVulkanBuffer(),
                             0, mNormalBuffer->getSize(), {0, 0, 0}, color.getExtent());
  }
  mCommandBufferIn->end();
  auto &queue = core::Context::Get()->getQueue();
  if (queue.submitAndWait(mCommandBufferIn.get()) != vk::Result::eSuccess) {
    throw std::runtime_error("OIDN input Vulkan copy failed");
  }
  if (trace) std::fprintf(stderr, "OIDN_HOST_SYNC filter %u\n", calls);
  mFilter.execute();
  char const *error;
  if (mDevice.getError(error) != oidn::Error::None) {
    throw std::runtime_error(error ? error : "OIDN filter failed");
  }
  if (trace) std::fprintf(stderr, "OIDN_HOST_SYNC output %u\n", calls);
  mCommandBufferOut->reset();
  mCommandBufferOut->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  color.recordCopyFromBuffer(mCommandBufferOut.get(), mInputBuffer->getVulkanBuffer(),
                            0, mInputBuffer->getSize(), {0, 0, 0}, color.getExtent());
  mCommandBufferOut->end();
  if (queue.submitAndWait(mCommandBufferOut.get()) != vk::Result::eSuccess) {
    throw std::runtime_error("OIDN output Vulkan copy failed");
  }
  if (trace) std::fprintf(stderr, "OIDN_HOST_SYNC done %u\n", calls);
}

} // namespace svulkan2::renderer
