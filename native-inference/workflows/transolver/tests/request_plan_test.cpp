#include "transolver_workflow/request_plan.hpp"
#include "transolver_workflow/runner.hpp"
#include "transolver_workflow/startup_check.hpp"
#include "physicsnemo/inference/v1/self_test.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
template <typename F> void rejects(F&& f, const std::string& fragment) {
  try { f(); } catch (const std::invalid_argument& error) {
    check(std::string(error.what()).find(fragment) != std::string::npos,
          "wrong error: " + std::string(error.what()));
    return;
  }
  throw std::runtime_error("expected rejection: " + fragment);
}
}

int main() {
  using namespace transolver_workflow;
  try {
    const std::vector<std::int64_t> shape{1, 2, 4};
    std::vector<float> startup_values(8, 0.0F);
    validate_startup_output(shape, shape, startup_values);
    startup_values[3] = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { validate_startup_output(shape, shape, startup_values); }, "finite");
    startup_values[3] = 0.0F;
    rejects([&] { validate_startup_output({1, 2, 5}, shape, startup_values); }, "shape");
    startup_values.pop_back();
    rejects([&] { validate_startup_output(shape, shape, startup_values); }, "element count");
    const std::vector<float> actual{1, 2, 3};
    const std::vector<float> expected{1, 2, 4};
    rejects([&] { physicsnemo::inference::v1::compare_self_test_float32(actual, expected, 1e-5, 1e-6); },
            "self_test output mismatch");
    check(physicsnemo::inference::v1::compare_self_test_float32(actual, actual, 0, 0).max_absolute_error == 0,
          "identical golden values must have zero error");
    rejects([&] { physicsnemo::inference::v1::compare_self_test_float32(actual, actual, -1, 0); },
            "tolerances");
    rejects([&] { physicsnemo::inference::v1::compare_self_test_float32(actual, startup_values, 0, 0); },
            "element count");
    bool external_cancelled = false;
    CancellationToken token([&] { return external_cancelled; });
    check(!token.cancelled(), "fresh callback cancellation");
    external_cancelled = true;
    check(token.cancelled(), "external cancellation callback");
    external_cancelled = false;
    token.cancel();
    check(token.cancelled(), "explicit cancellation survives callback reset");
    check(plan_blocks(59999, 60000) == std::vector<InferenceBlock>{{0, 59999}}, "59999 grouping");
    check(plan_blocks(60000, 60000) == std::vector<InferenceBlock>{{0, 60000}}, "60000 grouping");
    rejects([] { (void)plan_blocks(60001, 60000); }, "final block contains one point");
    check(plan_blocks(60011, 60000) == std::vector<InferenceBlock>{{0, 60000}, {60000, 11}}, "60011 grouping");
    check(plan_blocks(120000, 60000) == std::vector<InferenceBlock>{{0, 60000}, {60000, 60000}}, "120000 grouping");
    const auto tail = plan_blocks(120011, 60000);
    check(tail == std::vector<InferenceBlock>{{0, 60000}, {60000, 60000}, {120000, 11}}, "120011 grouping");
    preflight_blocks(tail, {60000, 11});
    preflight_blocks(tail, {-1});
    rejects([&] { preflight_blocks(tail, {60000}); }, "11-point block");
    rejects([] { (void)plan_blocks(0, 60000); }, "at least two mesh locations");
    rejects([] { (void)plan_blocks(10, 1); }, "block_size must be at least two");
    check(provider_case_seed(0, "run_1") == 2260079506302488169LL, "provider case seed differs for (0, run_1)");
    check(provider_case_seed(42, "run_1") == 1615895419752914611LL, "provider case seed differs for (42, run_1)");
    check(provider_case_seed(42, "run_11") == 3962535327529237276LL, "provider case seed differs for (42, run_11)");
    check(provider_case_seed(-1, "") == 6638654126240681726LL, "empty case seed");
    check(provider_case_seed(9223372036854775807LL, std::string(100, 'x')) == 2791673934283558317LL,
          "multi-block SHA256 seed");
    check(provider_case_seed(42, "café/车") == 6909229864957396684LL, "UTF8 case seed");
    std::cout << "Transolver request contract passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
