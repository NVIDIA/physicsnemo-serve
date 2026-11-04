#pragma once

#if defined(_WIN32) && defined(PNMS_TEST_CUSTOM_EXPORTS)
#define PNMS_TEST_CUSTOM_API __declspec(dllexport)
#elif defined(_WIN32)
#define PNMS_TEST_CUSTOM_API __declspec(dllimport)
#else
#define PNMS_TEST_CUSTOM_API
#endif

namespace physicsnemo::workflow::v1::test {
PNMS_TEST_CUSTOM_API int initializations();
PNMS_TEST_CUSTOM_API int destructions();
PNMS_TEST_CUSTOM_API int wrong_threads();
}
