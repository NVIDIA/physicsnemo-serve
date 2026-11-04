#pragma once

#include <functional>
#include <memory>
#include <string>

namespace physicsnemo::workflow::v1 {

// Construct, use and destroy on one dedicated native thread. This facade keeps
// SDK tensors, backend objects and device pointers entirely inside C++.
class WorkflowHost {
 public:
  explicit WorkflowHost(const std::string& config_json);
  ~WorkflowHost();
  WorkflowHost(const WorkflowHost&) = delete;
  WorkflowHost& operator=(const WorkflowHost&) = delete;

  std::string self_test(std::function<bool()> cancelled = {});
  std::string run(const std::string& request_json,
                  std::function<bool()> cancelled = {});

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace physicsnemo::workflow::v1
