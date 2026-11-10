#include "graphics/native_gpu/nativeAcbNullGate.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <random>
#include <vector>

using Libs::Graphics::NativeGpu::AcbNullGate;

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "NativeGpuAcbNullGateTests: failed: %s\n", message);
		std::abort();
	}
}

void TestNothingAhead() {
	AcbNullGate    gate;
	const uint64_t compute = gate.EnqueueCompute();
	Check(gate.ComputeMayStart(compute), "a compute buffer with no null before it starts at once");
	AcbNullGate    none;
	const uint64_t null = none.EnqueueNull();
	Check(none.NullMayGo(null), "a null with no compute buffer before it goes at once");
}

void TestComputeWaitsForEarlierNull() {
	AcbNullGate    gate;
	const uint64_t null    = gate.EnqueueNull();
	const uint64_t compute = gate.EnqueueCompute();
	Check(gate.NullMayGo(null), "the null owes nothing to a later compute buffer");
	Check(!gate.ComputeMayStart(compute), "the compute buffer does not pass the null before it");
	gate.NullPassed();
	Check(gate.ComputeMayStart(compute), "it starts once the null is passed");
}

void TestNullWaitsForEarlierCompute() {
	AcbNullGate    gate;
	const uint64_t first  = gate.EnqueueCompute();
	const uint64_t second = gate.EnqueueCompute();
	const uint64_t null   = gate.EnqueueNull();
	Check(gate.ComputeMayStart(first) && gate.ComputeMayStart(second), "both are ahead of the null");
	Check(!gate.NullMayGo(null), "nothing handed over yet");
	gate.ComputeHanded();
	Check(!gate.NullMayGo(null), "one of two handed over");
	gate.ComputeHanded();
	Check(gate.NullMayGo(null), "both handed over");
}

void TestAlternating() {
	AcbNullGate    gate;
	const uint64_t c1 = gate.EnqueueCompute();
	const uint64_t n1 = gate.EnqueueNull();
	const uint64_t c2 = gate.EnqueueCompute();
	const uint64_t n2 = gate.EnqueueNull();
	Check(gate.ComputeMayStart(c1) && !gate.NullMayGo(n1), "first the compute buffer");
	gate.ComputeHanded();
	Check(gate.NullMayGo(n1) && !gate.ComputeMayStart(c2), "then the null");
	gate.NullPassed();
	Check(gate.ComputeMayStart(c2) && !gate.NullMayGo(n2), "then the second compute buffer");
	gate.ComputeHanded();
	Check(gate.NullMayGo(n2), "then the second null");
}

// Any order of enqueues and steps of the two consumers: both finish, and neither kind overtakes the other.
void TestRandomInterleavings() {
	struct Item {
		uint64_t stamp;
		size_t   index;
	};
	for (uint32_t seed = 1; seed <= 2000; seed++) {
		std::mt19937      rng(seed);
		AcbNullGate       gate;
		std::deque<Item>  computes, nulls;
		const size_t      total = 8 + (rng() % 32);
		std::vector<char> done(total, 0);
		std::vector<char> is_null(total, 0);
		size_t            enqueued = 0;
		size_t            finished = 0;
		auto              others_before_done = [&](size_t index, bool null) {
			for (size_t i = 0; i < index; i++) {
				if ((is_null[i] != 0) != null && done[i] == 0) {
					return false;
				}
			}
			return true;
		};
		for (uint32_t step = 0; finished < total; step++) {
			Check(step < 1000000, "the gate never deadlocks");
			switch (rng() % 3) {
				case 0:
					if (enqueued < total) {
						const bool null   = (rng() % 3) == 0;
						is_null[enqueued] = null ? 1 : 0;
						if (null) {
							nulls.push_back({gate.EnqueueNull(), enqueued});
						} else {
							computes.push_back({gate.EnqueueCompute(), enqueued});
						}
						enqueued++;
					}
					break;
				case 1:
					if (!computes.empty() && gate.ComputeMayStart(computes.front().stamp)) {
						Check(others_before_done(computes.front().index, false), "a compute buffer passed a null before it");
						done[computes.front().index] = 1;
						computes.pop_front();
						gate.ComputeHanded();
						finished++;
					}
					break;
				default:
					if (!nulls.empty() && gate.NullMayGo(nulls.front().stamp)) {
						Check(others_before_done(nulls.front().index, true), "a null passed a compute buffer before it");
						done[nulls.front().index] = 1;
						nulls.pop_front();
						gate.NullPassed();
						finished++;
					}
					break;
			}
		}
	}
}

} // namespace

int main() {
	TestNothingAhead();
	TestComputeWaitsForEarlierNull();
	TestNullWaitsForEarlierCompute();
	TestAlternating();
	TestRandomInterleavings();
	std::fprintf(stderr, "NativeGpuAcbNullGateTests: all passed\n");
	return 0;
}
