#include "hip_graph_modifier.hpp"
#include "cl_kernel.h"
#include "hip/hip_runtime_api.h"
#include "hip_graph_fuse_recorder.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "hip_graph_internal.hpp"
#include "platform/kernel.hpp"
#include <yaml-cpp/yaml.h>


namespace {
void rtrim(std::string& str) { str.erase(std::find(str.begin(), str.end(), '\0'), str.end()); }

void loadExternalSymbols(const std::vector<std::pair<std::string, std::string>>& fusedGroups) {
  HIP_INIT_VOID();
  for (auto& [symbolName, imagePath] : fusedGroups) {
    hip::PlatformState::instance().loadExternalSymbol(symbolName, imagePath);
  }
}

dim3 max(dim3& one, dim3& two) {
  return dim3(std::max(one.x, two.x), std::max(one.y, two.y), std::max(one.z, two.z));
}

class FusionGroup : public hip::GraphNode {
 public:
  FusionGroup() : hip::GraphNode(hipGraphNodeTypeKernel) {
    semaphore_ = hip::PlatformState::instance().getSemaphore();
  }
  virtual ~FusionGroup() { delete fusedNode_; };
  void addNode(hip::GraphKernelNode* node) { fusee_.push_back(node); }
  std::vector<hip::GraphKernelNode*>& getNodes() { return fusee_; }
  hip::GraphKernelNode* getHead() { return fusee_.empty() ? nullptr : fusee_.front(); }
  hip::GraphKernelNode* getTail() { return fusee_.empty() ? nullptr : fusee_.back(); }

  hip::GraphKernelNode* getFusedNode() { return fusedNode_; }
  void dumpExtra(hipKernelNodeParams nodeParams);
  void dumpDescriptor(amd::Kernel* kernel);
  char* clk_value_type_to_string(clk_value_type_t type);
  void insertAlignKernarg(size_t alignment, uint8_t* kernarg, size_t kernargSize, size_t* kernargs_combined_size);
  void generateNode(void* functionHandle);
  hipKernelNodeParams* getNodeParams() { return &fusedNodeParams_; }

 private:
  hip::GraphNode* clone() const override { return nullptr; }
  std::vector<hip::GraphKernelNode*> fusee_{};
  hip::GraphKernelNode* fusedNode_;
  hipKernelNodeParams fusedNodeParams_{};
  std::vector<void*> kernelArgs_{};
  std::vector<uint8_t> gemmKernelArgs_{};
  std::vector<void*> hiddenkernelArgs_{};
  void* semaphore_{};
};

char* FusionGroup::clk_value_type_to_string(clk_value_type_t type) {  
    switch (type) {  
        case T_VOID: return "T_VOID";  
        case T_CHAR: return "T_CHAR";  
        case T_SHORT: return "T_SHORT";  
        case T_INT: return "T_INT";  
        case T_LONG: return "T_LONG";  
        case T_FLOAT: return "T_FLOAT";  
        case T_DOUBLE: return "T_DOUBLE";  
        case T_POINTER: return "T_POINTER";  
        case T_CHAR2: return "T_CHAR2";  
        case T_CHAR3: return "T_CHAR3";  
        case T_CHAR4: return "T_CHAR4";  
        case T_CHAR8: return "T_CHAR8";  
        case T_CHAR16: return "T_CHAR16";  
        case T_SHORT2: return "T_SHORT2";  
        case T_SHORT3: return "T_SHORT3";  
        case T_SHORT4: return "T_SHORT4";  
        case T_SHORT8: return "T_SHORT8";  
        case T_SHORT16: return "T_SHORT16";  
        case T_INT2: return "T_INT2";  
        case T_INT3: return "T_INT3";  
        case T_INT4: return "T_INT4";  
        case T_INT8: return "T_INT8";  
        case T_INT16: return "T_INT16";  
        case T_LONG2: return "T_LONG2";  
        case T_LONG3: return "T_LONG3";  
        case T_LONG4: return "T_LONG4";  
        case T_LONG8: return "T_LONG8";  
        case T_LONG16: return "T_LONG16";  
        case T_FLOAT2: return "T_FLOAT2";  
        case T_FLOAT3: return "T_FLOAT3";  
        case T_FLOAT4: return "T_FLOAT4";  
        case T_FLOAT8: return "T_FLOAT8";  
        case T_FLOAT16: return "T_FLOAT16";  
        case T_DOUBLE2: return "T_DOUBLE2";  
        case T_DOUBLE3: return "T_DOUBLE3";  
        case T_DOUBLE4: return "T_DOUBLE4";  
        case T_DOUBLE8: return "T_DOUBLE8";  
        case T_DOUBLE16: return "T_DOUBLE16";  
        case T_SAMPLER: return "T_SAMPLER";  
        case T_SEMA: return "T_SEMA";  
        case T_STRUCT: return "T_STRUCT";  
        case T_QUEUE: return "T_QUEUE";  
        case T_PAD: return "T_PAD";  
        default: return "UNKNOWN";  
    }  
}  

void FusionGroup::dumpDescriptor(amd::Kernel* kernel) {
  for (int i = 0; i < kernel->signature().numParameters(); i++) {
    const amd::KernelParameterDescriptor& desc = kernel->signature().at(i);
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "amd::KernelParameterDescriptor - typeName_: %s", desc.typeName_.c_str());
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "amd::KernelParameterDescriptor - name_: %s", desc.name_.c_str());
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "amd::KernelParameterDescriptor - type_: %s", clk_value_type_to_string(desc.type_));
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "amd::KernelParameterDescriptor - size_: %zu", desc.size_);
  }
}

void FusionGroup::dumpExtra(hipKernelNodeParams nodeParams) {
  ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "hipKernelNodeParams::extra[0] -> HIP_LAUNCH_PARAM_BUFFER_POINTER: %p", (void*)nodeParams.extra[0]);
  ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "hipKernelNodeParams::extra[2] -> HIP_LAUNCH_PARAM_BUFFER_SIZE: %p", (void*)nodeParams.extra[2]);
  ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "hipKernelNodeParams::extra[4] -> HIP_LAUNCH_PARAM_END: %p", (void*)nodeParams.extra[4]);
  ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "hipKernelNodeParams::extra[3] -> KERNARG SIZE: %zu", *((size_t*)nodeParams.extra[3]));
  uint8_t* kernargs = static_cast<uint8_t*>(nodeParams.extra[1]);
  size_t k_size = *((size_t*)nodeParams.extra[3]);
  ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "hipKernelNodeParams::extra[1] -> KERNARG BUFFER: ");
  for (int i = 0; i < k_size; i++) {
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "%02x ", kernargs[i]);
  }
}

void FusionGroup::insertAlignKernarg(size_t alignment, uint8_t* kernarg, size_t kernargSize, size_t* kernargs_combined_size) {
  // Calculate the padding needed to align to alignment bytes
  size_t padding = (alignment - (kernargSize % alignment)) % alignment;
  // Insert the new data
  gemmKernelArgs_.insert(gemmKernelArgs_.end(), kernarg, kernarg + kernargSize);
  // Insert padding bytes
  gemmKernelArgs_.insert(gemmKernelArgs_.end(), padding, 0);
  // Update combined kernel arg size to account for the extra memory used to align to alignment size
  *kernargs_combined_size += padding;
}

void FusionGroup::generateNode(void* functionHandle) {
  fusedNodeParams_.blockDim = dim3(0, 0, 0);
  fusedNodeParams_.gridDim = dim3(0, 0, 0);
  fusedNodeParams_.sharedMemBytes = 0;
  fusedNodeParams_.func = functionHandle;
  size_t kernargs_combined_size = 0;

  // Allocate the extra buffer one time at the beginning
  fusedNodeParams_.extra = (void**)malloc(5 * sizeof(void*));
  if (fusedNodeParams_.extra == nullptr) {
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "FusionGroup::generateNode() - HIP error out of memory for allocating extra!");
  }

  for (auto* node : fusee_) {
    auto nodeParams = hip::GraphFuseRecorder::getKernelNodeParams(node);
    fusedNodeParams_.blockDim = max(fusedNodeParams_.blockDim, nodeParams.blockDim);
    fusedNodeParams_.gridDim = max(fusedNodeParams_.gridDim, nodeParams.gridDim);
    fusedNodeParams_.sharedMemBytes = std::max(fusedNodeParams_.sharedMemBytes, nodeParams.sharedMemBytes);

    auto* kernel = hip::GraphFuseRecorder::getDeviceKernel(nodeParams);
    const auto numKernelArgsAndHidden = kernel->signature().numParametersAll();
    const auto numKernelArgs = kernel->signature().numParameters();
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "# Kernel args + hidden: %d", numKernelArgsAndHidden);
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "# Kernel args: %d", numKernelArgs);

    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "FusionGroup::generateNode() - Dumping out descriptors for node in fusee_!");
    dumpDescriptor(kernel);
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "FusionGroup::generateNode() - Dumping out nodeParams->extra for node in fusee_!");
    dumpExtra(nodeParams);

    // Populate from the extra struct if kernel args are not passed from  kernelParams
    if (nodeParams.kernelParams == nullptr) {
      // 'extra' is a struct that contains the following info: 
      // {
        // HIP_LAUNCH_PARAM_BUFFER_POINTER, kernargs,
        // HIP_LAUNCH_PARAM_BUFFER_SIZE, &kernargs_size,
        // HIP_LAUNCH_PARAM_END 
      // }
      fusedNodeParams_.extra[0] = nodeParams.extra[0]; // HIP_LAUNCH_PARAM_BUFFER_POINTER
      fusedNodeParams_.extra[2] = nodeParams.extra[2]; // HIP_LAUNCH_PARAM_BUFFER_SIZE
      fusedNodeParams_.extra[4] = nodeParams.extra[4]; // HIP_LAUNCH_PARAM_END
      kernargs_combined_size += *((size_t*)nodeParams.extra[3]); // += kernargs_size
      // We cast it back to uint8_t since kernargs are actually stored as this: 
      // https://github.com/ROCm/hipBLASLt/blob/33e633fe2a270dd5c3a8b0e4ed12147f77d32761/tensilelite/Tensile/Source/lib/include/Tensile/KernelArguments.hpp#L227
      uint8_t* kernargs = static_cast<uint8_t*>(nodeParams.extra[1]);
      insertAlignKernarg(8, kernargs, *((size_t*)nodeParams.extra[3]), &kernargs_combined_size);
    } else {
      for (size_t i = 0; i < numKernelArgsAndHidden; ++i) {
        if (kernel->signature().at(i).info_.hidden_) hiddenkernelArgs_.push_back(nodeParams.kernelParams[i]);
        else kernelArgs_.push_back(nodeParams.kernelParams[i]);
      }
    }
  }

  size_t semaphoreSize = sizeof(semaphore_);
  kernargs_combined_size += semaphoreSize;
  uint8_t* semaphoreBytePtr = reinterpret_cast<uint8_t*>(&semaphore_);
  gemmKernelArgs_.insert(gemmKernelArgs_.end(), semaphoreBytePtr, semaphoreBytePtr + semaphoreSize);
  fusedNodeParams_.extra[1] = malloc(kernargs_combined_size);
  if (fusedNodeParams_.extra[1] == nullptr) {
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "FusionGroup::generateNode() - HIP error out of memory for allocating extra[1]: kernargs!");
  }
  fusedNodeParams_.extra[3] = malloc(sizeof(void*));
  if (fusedNodeParams_.extra[3] == nullptr) {
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "FusionGroup::generateNode() - HIP error out of memory for allocating extra[3]: kernargs_size!");
  }
  *((size_t*)fusedNodeParams_.extra[3]) = kernargs_combined_size;
  ::memcpy(fusedNodeParams_.extra[1], gemmKernelArgs_.data(), kernargs_combined_size);

  // TODO: In the near future we have to figure out how to mix kernel args coming from `extra` buffer and the `kernelParams` buffer
  // The commented code below deals with kernargs coming from kernelParams path
  // kernelArgs_.push_back(&semaphore_);
  // kernelArgs_.reserve(kernelArgs_.size() + hiddenkernelArgs_.size());
  // kernelArgs_.insert(kernelArgs_.end(), hiddenkernelArgs_.begin(), hiddenkernelArgs_.end());

  // for hipblaslt gemm kernels that use hipExtModuleLaunchKernel()
  fusedNodeParams_.kernelParams = nullptr;
  // fusedNodeParams_.extra = kernelArgs_.data();

  ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "FusionGroup::generateNode() - Dumping out fusedNodeParams->extra after merging kernargs!");
  dumpExtra(fusedNodeParams_);

  hipError_t status = hip::GraphKernelNode::validateKernelParams(&fusedNodeParams_);
  if (hipSuccess != status) {
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "`validateKernelParams` during fusion");
  }
  fusedNode_ = new hip::GraphKernelNode(&fusedNodeParams_, nullptr);
  
  auto fusedParams = hip::GraphFuseRecorder::getKernelNodeParams(fusedNode_);
  auto kernel = hip::GraphFuseRecorder::getDeviceKernel(fusedParams);
  ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "FusionGroup::generateNode() - Dumping out descriptors for each kernarg in fused kernel!");
  dumpDescriptor(kernel);
}
}  // namespace


namespace hip {
bool GraphModifier::isSubstitutionStateQueried_{false};
bool GraphModifier::isSubstitutionSwitchedOn_{false};
GraphModifier::CounterType GraphModifier::instanceCounter_{};
hip::ExternalCOs::SymbolTableType GraphModifier::symbolTable_{};
std::vector<GraphModifier::GraphDescription> GraphModifier::descriptions_{};

bool GraphModifier::isInputOk() {
  auto* env = getenv("AMD_FUSION_MANIFEST");
  if (env == nullptr) {
    std::stringstream msg;
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS,
            "fusion manifest is not specified; cannot proceed fusion substitution");
    return false;
  }

  std::string manifestPathName(env);
  std::filesystem::path filePath(manifestPathName);
  if (!std::filesystem::exists(filePath)) {
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "cannot open fusion manifest file: %s",
            manifestPathName.c_str());
    return false;
  }

  std::vector<std::pair<std::string, std::string>> fusedGroups{};
  std::vector<std::vector<std::vector<size_t>>> executionOrders{};
  try {
    const auto& manifest = YAML::LoadFile(manifestPathName);
    const auto& graphs = manifest["graphs"];
    for (const auto& graph : graphs) {
      GraphModifier::GraphDescription descr{};
      for (const auto& group : graph["groups"]) {
        const auto& groupName = group["name"].as<std::string>();
        const auto& location = group["location"].as<std::string>();
        fusedGroups.push_back(std::make_pair(groupName, location));
        descr.groupSymbols_.push_back(groupName);
      }
      std::vector<std::vector<size_t>> order;
      for (const auto& sequence : graph["executionOrder"]) {
        order.push_back(sequence.as<std::vector<size_t>>());
      }
      descr.executionGroups_ = std::move(order);
      descriptions_.push_back(descr);
    }
  } catch (const YAML::ParserException& ex) {
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "error while parsing fusion manifest: %s", ex.what());
    return false;
  } catch (const std::runtime_error& ex) {
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "error while parsing fusion manifest: %s", ex.what());
    return false;
  }

  loadExternalSymbols(fusedGroups);
  GraphModifier::symbolTable_ = hip::PlatformState::instance().getExternalSymbolTable();

  auto isOk = hip::PlatformState::instance().initSemaphore();
  if (!isOk) {
    return false;
  }

  ClPrint(amd::LOG_INFO, amd::LOG_ALWAYS, "graph fuse substitution is enabled");
  return true;
}

bool GraphModifier::isSubstitutionOn() {
  static amd::Monitor lock_;
  amd::ScopedLock lock(lock_);
  if (!isSubstitutionStateQueried_) {
    isSubstitutionSwitchedOn_ = GraphModifier::isInputOk();
    isSubstitutionStateQueried_ = true;
  }
  return isSubstitutionSwitchedOn_;
}

void GraphModifier::Finalizer::operator()(size_t* instanceCounter) const {
  for (auto& description : GraphModifier::descriptions_) {
    for (auto& group : description.fusionGroups_) {
      delete group;
    }
  }
  delete instanceCounter;
}

GraphModifier::GraphModifier(hip::Graph*& graph) : graph_(graph) {
  amd::ScopedLock lock(fclock_);
  if (GraphModifier::instanceCounter_ == nullptr) {
    GraphModifier::instanceCounter_ = GraphModifier::CounterType{new size_t};
    *(GraphModifier::instanceCounter_) = 0;
  }
  instanceId_ = (*GraphModifier::instanceCounter_)++;
}

GraphModifier::~GraphModifier() {}

bool GraphModifier::check() {
  size_t numNodes{0};
  const auto& executionGroups_ = currDescription.executionGroups_;
  std::for_each(executionGroups_.begin(), executionGroups_.end(),
                [&numNodes](auto& group) { numNodes += group.size(); });
  const auto& originalGraphNodes = graph_->GetNodes();
  return numNodes == originalGraphNodes.size();
}

std::vector<hip::GraphNode*> GraphModifier::collectNodes(const std::vector<Node>& originalNodes) {
  size_t nodeCounter{0};
  std::vector<hip::GraphNode*> nodes{};
  const auto& executionGroups = currDescription.executionGroups_;
  for (const auto& group : executionGroups) {
    if (group.size() == 1) {
      const auto nodeNumber = group[0];
      guarantee(nodeNumber == nodeCounter, "the execution order must be correct");
      nodes.push_back(originalNodes[nodeCounter]);
      ++nodeCounter;
    } else {
      FusionGroup* fusionGroup = new FusionGroup{};
      currDescription.fusionGroups_.emplace_back(fusionGroup);

      for (const auto& nodeNumber : group) {
        guarantee(nodeNumber == nodeCounter, "the execution order must be correct");
        auto originalNode = originalNodes[nodeNumber];
        auto* kernelNode = dynamic_cast<hip::GraphKernelNode*>(originalNode);
        fusionGroup->addNode(kernelNode);
        ++nodeCounter;
      }
      nodes.push_back(fusionGroup);
    }
  }
  return nodes;
}

void GraphModifier::generateFusedNodes() {
  for (size_t groupId = 0; groupId < currDescription.fusionGroups_.size(); ++groupId) {
    std::string groupKey = currDescription.groupSymbols_.at(groupId);
    auto [funcHandle, fusedKernel] = GraphModifier::symbolTable_.at(groupKey);
    std::string out = "";
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "GOT HERE passed fetch from symbolTable %s", out.c_str());
    auto* group = dynamic_cast<FusionGroup*>(currDescription.fusionGroups_.at(groupId));
    group->generateNode(funcHandle);
  }
}

void GraphModifier::performCortSubstitution(const std::vector<Node>& originalNodes) {
  // This section is for debug purposes, wanted dump out what's inside the symbolTable and see if any symbols are missing
  ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[GraphModifier]: Dumping out contents of symbolTable_");
  for (const auto& entry : GraphModifier::symbolTable_) {
    const std::string& key = entry.first;
    void* voidPtr = entry.second.first;
    hip::Function* funcPtr = entry.second.second;
    ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[GraphModifier]: key: %s", key.c_str());
    ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[GraphModifier]: void* address: %p", voidPtr);
    if (funcPtr) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[GraphModifier]: hip::Function* address: %p", funcPtr);
    } else {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[GraphModifier]: hip::Function* is null");
    }
  }
  ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[GraphModifier]: Finished dumping contents of symbolTable_");

  // Substitution begins here
  for (const auto& kernelNode: originalNodes) {
    const auto type = kernelNode->GetType();
    if (type == hipGraphNodeTypeKernel) {
      // Get kernel symbol name and hipKernelNodeParams
      hipKernelNodeParams params = GraphFuseRecorder::getKernelNodeParams(kernelNode);
      amd::Kernel* kernel = GraphFuseRecorder::getDeviceKernel(params);
      std::string symbolName = kernel->name();
      rtrim(symbolName);
      try {
        // Get rearranged func handle by symbol name from the external symbol table  
        auto [cortFuncHandle, cortOriginalKernel] = GraphModifier::symbolTable_.at(symbolName);  
        // Modify original hipKernelNodeParams' funcHandle with the rearranged funcHandle  
        params.func = cortFuncHandle;  
        // Now set the modified hipKernelNodeParams as the current used hipKernelNodeParams by the original kernel node  
        auto* graphKernelNode = dynamic_cast<GraphKernelNode*>(kernelNode);  
        if (graphKernelNode == nullptr) {  
          ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[GraphModifier]: Failed to convert a graph node to `GraphKernelNode`");  
          continue;  
        }
        hipError_t status = graphKernelNode->SetParams(&params);  
        if (hipSuccess != status) {  
          ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[GraphModifier]: Failed to set rearranged hipKernelNodeParams for kernel - %s", symbolName.c_str());  
        }
        ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[GraphModifier]: Successfully set the rearranged function handle for kernel - %s", symbolName.c_str());
      } catch (const std::exception& e) {  
        ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[GraphModifier]: Exception caught during symbol table lookup for symbol - %s: %s", symbolName.c_str(), e.what());  
      }  
    }
  }
}

void GraphModifier::run() {
  amd::ScopedLock lock(fclock_);
  currDescription = descriptions_[instanceId_];

  // auto isOk = check();
  // if (!isOk) {
  //   ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "[GraphModifier]: Rearranged code object binary substitution %zu failed consistency check", instanceId_);
  //   return;
  // }

  // 1. Get original graph nodes
  const auto& originalGraphNodes = graph_->GetNodes();

  // 2. Modify original graph nodes with kernel codes coming from the rearranged code object
  performCortSubstitution(originalGraphNodes);

  // auto nodes = collectNodes(originalGraphNodes);
  // std::string out = "";
  // ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "GOT HERE collectNodes() passed %s", out.c_str());
  // generateFusedNodes();
  // ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "GOT HERE generateFusedNodes() passed %s", out.c_str());

  // for (size_t i = 0; i < currDescription.fusionGroups_.size(); ++i) {
  //   auto* group = dynamic_cast<FusionGroup*>(currDescription.fusionGroups_.at(i));
  //   auto* fusedNode = group->getFusedNode();

  //   auto* groupHead = group->getHead();
  //   if (groupHead) {
  //     const auto& dependencies = groupHead->GetDependencies();
  //     std::vector<Node> additionalEdges{fusedNode};
  //     for (const auto& dependency : dependencies) {
  //       dependency->RemoveUpdateEdge(groupHead);
  //       dependency->AddEdge(fusedNode);
  //     }
  //   }

  //   auto* groupTail = group->getTail();
  //   if (groupTail) {
  //     const auto& edges = groupTail->GetEdges();
  //     for (const auto& edge : edges) {
  //       groupTail->RemoveUpdateEdge(edge);
  //       fusedNode->AddEdge(edge);
  //     }
  //   }

  //   auto& fusee = group->getNodes();
  //   for (auto node : fusee) {
  //     graph_->RemoveNode(node);
  //   }
  //   graph_->AddNode(fusedNode);
  // }
  ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "[GraphModifier]: Passed rearranged code object binary substitution!");
}
}  // namespace hip
