#pragma once

namespace ffs_depth {

// Registers the FFSGWCVolume TensorRT plugin creator in the global registry.
// Safe to call more than once.
bool registerFFSGWCPlugin();

}  // namespace ffs_depth

// C ABI wrapper for loading/registering the plugin from Python via ctypes.
extern "C" bool ffs_register_gwc_plugin();
