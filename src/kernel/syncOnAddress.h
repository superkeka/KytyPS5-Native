#ifndef EMULATOR_INCLUDE_EMULATOR_KERNEL_SYNC_ON_ADDRESS_H_
#define EMULATOR_INCLUDE_EMULATOR_KERNEL_SYNC_ON_ADDRESS_H_

#include "common/abi.h"
#include "common/common.h"

#include <chrono>

namespace Libs::LibKernel::SyncOnAddress {

using signal_poll_func_t = void (*)();

int Wait32(volatile uint32_t* address, uint32_t expected, const uint32_t* timeout_micros,
           signal_poll_func_t signal_poll = nullptr);
int Wait64(volatile uint64_t* address, uint64_t expected, const uint32_t* timeout_micros,
           signal_poll_func_t signal_poll = nullptr);
int Wait64(volatile uint64_t* address, uint64_t expected, std::chrono::nanoseconds timeout,
           signal_poll_func_t signal_poll = nullptr);
int Wake(volatile void* address, int32_t count);
// True when a thread may be sleeping in Wait32/Wait64 on `address` (or an address sharing its
// counter): a store made before this call is seen by any waiter that is not counted yet, so a caller
// that publishes a value and finds no waiter can skip the wake syscall.
bool HasWaiters(volatile void* address);

} // namespace Libs::LibKernel::SyncOnAddress

#endif /* EMULATOR_INCLUDE_EMULATOR_KERNEL_SYNC_ON_ADDRESS_H_ */
