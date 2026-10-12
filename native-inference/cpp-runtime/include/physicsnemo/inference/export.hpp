#pragma once

// Most SDK functions are exported by CMake's WINDOWS_EXPORT_ALL_SYMBOLS.
// A polymorphic base with inline construction also needs an explicit import of
// its vtable when an optional backend subclasses it from a separate DLL.
#if defined(_WIN32) && defined(PNMIR_SHARED)
#  if defined(pnmir_EXPORTS)
#    define PNMIR_API __declspec(dllexport)
#  else
#    define PNMIR_API __declspec(dllimport)
#  endif
#else
#  define PNMIR_API
#endif
